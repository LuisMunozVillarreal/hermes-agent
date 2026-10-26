"""Adversarial chunk transfers: no partial dispatch or leaked spools."""
import base64
import json
import os
import select
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from types import SimpleNamespace

import pytest

from agent.isolated_channel import Channel
from agent.isolated_protocol import MAX_FRAME, send
from agent import isolated_transport as transport


@pytest.mark.parametrize('fault', ['stale', 'sequence', 'identity', 'direction', 'total',
    'oversize', 'short', 'base64', 'version', 'missing', 'timeout', 'disconnect', 'cancel'])
def test_bad_partial_transfer_closes_spool_without_dispatch(monkeypatch, fault):
    files = []
    original = transport.tempfile.TemporaryFile
    def tracked(*args, **kwargs):
        file = original(*args, **kwargs)
        files.append(file)
        return file
    monkeypatch.setattr(transport, 'tempfile', SimpleNamespace(TemporaryFile=tracked))
    monkeypatch.setattr(transport, 'TRANSFER_TIMEOUT', 0.2)
    a, b = socket.socketpair()
    calls = []
    channel = Channel(a, 'nonce', lambda *args: calls.append(args))
    first = {'v': 1, 'generation': 'nonce', 'chunk': 'a'*32, 'reply_chunk': False,
             'sequence': 0, 'size': MAX_FRAME + 123,
             'data': base64.b64encode(b'x' * transport.CHUNK_BYTES).decode()}
    second = dict(first, sequence=1)
    changes = {'stale': {'generation': 'old'}, 'sequence': {'sequence': 0},
        'identity': {'chunk': 'b'*32}, 'direction': {'reply_chunk': True},
        'total': {'size': transport.MAX_MESSAGE + 1},
        'oversize': {'data': base64.b64encode(b'x'*(transport.CHUNK_BYTES + 1)).decode()},
        'short': {'data': ''}, 'base64': {'data': '?!'}, 'version': {'v': 2}}
    try:
        send(b, first)
        if fault in changes:
            send(b, dict(second, **changes[fault]))
        elif fault == 'missing':
            second.pop('data')
            send(b, second)
        elif fault == 'disconnect':
            b.close()
        elif fault == 'cancel':
            channel.close()
        assert channel.closed.wait(3)
        channel._thread.join(3)
        assert not channel._thread.is_alive()
        assert not calls
        assert all(file.closed for file in files)
        if fault != 'cancel':
            assert files, 'partial transfer did not exercise spooling'
    finally:
        channel.close()
        b.close()


def test_blocked_large_send_is_cancellable_and_releases_spool(monkeypatch):
    files = []
    original = transport.tempfile.TemporaryFile
    def tracked(*args, **kwargs):
        f = original(*args, **kwargs)
        files.append(f)
        return f
    monkeypatch.setattr(transport, 'tempfile', SimpleNamespace(TemporaryFile=tracked))
    a, b = socket.socketpair()
    a.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    client = Channel(a, 'nonce', lambda *_: None)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(client.call, 'run', 'x' * (MAX_FRAME * 2), None)
        try:
            # Receiving a byte proves the send passed serialization and is live.
            b.settimeout(3)
            assert b.recv(1)
            client.close()
            # Future.result's TimeoutError is an OSError too: first prove the
            # sender finished, rather than mistaking a wait timeout for cleanup.
            done, _ = wait([future], timeout=3)
            assert future in done, 'sender did not finish after channel close'
            with pytest.raises((RuntimeError, OSError, EOFError)):
                future.result()
            assert files and all(file.closed for file in files)
            assert not client._pending
        finally:
            client.close()
            b.close()
            client._thread.join(3)
            assert not client._thread.is_alive()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('direction', ['send', 'receive'])
def test_closed_transport_does_not_wait_on_reused_descriptor(monkeypatch, direction):
    files = []
    original_file = transport.tempfile.TemporaryFile
    def tracked(*args, **kwargs):
        file = original_file(*args, **kwargs)
        files.append(file)
        return file
    monkeypatch.setattr(transport, 'tempfile', SimpleNamespace(TemporaryFile=tracked))
    a, b = socket.socketpair()
    replacement, replacement_peer = socket.socketpair()
    if direction == 'send':
        replacement.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        replacement.setblocking(False)
        with pytest.raises(BlockingIOError):
            while True:
                replacement.send(b'x' * 4096)
    selected, resume = threading.Event(), threading.Event()
    original_select = select.select
    fd = a.fileno()
    reused = False

    def select_after_close(readers, writers, errors, timeout=None):
        if files and (readers == [a] or writers == [a]) and not selected.is_set():
            # select converts socket objects to integers before entering the OS.
            # Close/reuse in that gap must not strand a transfer on another socket.
            readers = [fd] if readers else []
            writers = [fd] if writers else []
            selected.set()
            assert resume.wait(3), 'descriptor-reuse setup did not finish'
        return original_select(readers, writers, errors, timeout)

    monkeypatch.setattr(select, 'select', select_after_close)
    def transfer():
        if direction == 'send':
            return transport.send_message(a,
                {'v': 1, 'generation': 'nonce', 'id': 'a' * 32, 'op': 'run',
                 'payload': 'x' * (MAX_FRAME * 2)}, time.monotonic() + 30)
        return transport.receive_message(a, 'nonce')

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(transfer)
        try:
            if direction == 'receive':
                send(b, {'v': 1, 'generation': 'nonce', 'chunk': 'a' * 32,
                    'reply_chunk': False, 'sequence': 0, 'size': MAX_FRAME + 123,
                    'data': base64.b64encode(b'x' * transport.CHUNK_BYTES).decode()})
            assert selected.wait(3), 'transfer did not reach the readiness wait'
            a.shutdown(socket.SHUT_RDWR)
            a.close()
            os.dup2(replacement.fileno(), fd)
            reused = True
            resume.set()
            done, _ = wait([future], timeout=3)
            assert future in done, 'closed transport waited on a reused descriptor'
            with pytest.raises((OSError, EOFError)):
                future.result()
            assert files and all(file.closed for file in files)
        finally:
            resume.set()
            a.close()
            b.close()
            replacement_peer.close()
            # Wake a regressed select before joining the executor on failure.
            try:
                future.result(timeout=3)
            except (OSError, EOFError):
                pass
            if reused:
                os.close(fd)
            replacement.close()


def test_handshake_receive_honors_socket_timeout():
    from agent.isolated_protocol import receive
    a, b = socket.socketpair()
    a.settimeout(0.1)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(receive, a)
        try:
            # Future timeout is deliberately longer than the socket deadline.
            with pytest.raises(TimeoutError):
                future.result(timeout=2)
            assert future.done(), 'receive ignored the bootstrap socket timeout'
        finally:
            a.shutdown(socket.SHUT_RDWR)
            a.close()
            b.close()


def test_construction_accepts_large_json_context_but_rejects_total_overflow():
    from agent.isolated_protocol import AgentConstruction
    context = [{'role': 'user', 'content': 'x' * (MAX_FRAME + 1)}]
    assert AgentConstruction('delegation', {'prefill_messages': context}).kwargs['prefill_messages'] == context
    with pytest.raises(ValueError, match='limit'):
        AgentConstruction('delegation', {'prefill_messages': ['x' * (transport.MAX_MESSAGE + 1)]})


def test_reassembled_reply_must_match_authenticated_correlation():
    a, b = socket.socketpair()
    called = []
    channel = Channel(a, 'nonce', lambda *args: called.append(args))
    message = json.dumps({'v': 1, 'generation': 'nonce', 'id': 'b'*32,
        'op': 'run', 'payload': 'x' * MAX_FRAME}).encode()
    try:
        for seq, start in enumerate(range(0, len(message), transport.CHUNK_BYTES)):
            send(b, {'v': 1, 'generation': 'nonce', 'chunk': 'a'*32, 'reply_chunk': False,
                'sequence': seq, 'size': len(message),
                'data': base64.b64encode(message[start:start + transport.CHUNK_BYTES]).decode()})
        assert channel.closed.wait(3)
        assert not called
    finally:
        channel.close()
        b.close()
