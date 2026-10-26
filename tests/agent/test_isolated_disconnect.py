"""Disconnect diagnostics use real sockets without logging private RPC data."""
from concurrent.futures import ThreadPoolExecutor
import socket
import struct
import threading

import pytest

from agent.isolated_channel import Channel
from agent.isolated_protocol import receive, send


@pytest.mark.parametrize('failure, detail', [
    ('eof', 'EOFError'),
    ('frame', 'invalid agent IPC frame size'),
    ('generation', 'stale agent generation'),
    ('json', 'JSONDecodeError'),
])
def test_receive_failure_settles_all_waiters_with_first_safe_reason(failure, detail):
    local, peer = socket.socketpair()
    peer.settimeout(3)
    channel = Channel(local, 'private-auth-token', lambda *_: None)
    with ThreadPoolExecutor(2) as pool:
        calls = [pool.submit(channel.call, 'run', {'secret': 'private-payload'}, None)
                 for _ in range(2)]
        try:
            for _ in calls:
                receive(peer)
            if failure == 'eof':
                peer.close()
            elif failure == 'frame':
                peer.sendall(struct.pack('!I', 0))
            elif failure == 'generation':
                send(peer, {'v': 1, 'generation': 'private-wrong-token',
                            'id': 'a' * 32, 'op': 'private-operation'})
            else:
                data = b'private-invalid-json'
                peer.sendall(struct.pack('!I', len(data)) + data)
            errors = []
            for future in calls:
                with pytest.raises(RuntimeError) as caught:
                    future.result(timeout=3)
                errors.append(str(caught.value))
            assert errors[0] == errors[1]
            assert 'receive' in errors[0]
            assert detail in errors[0]
            assert 'private-' not in errors[0]
            channel.close()
            with pytest.raises(RuntimeError) as caught:
                channel.call('later')
            assert str(caught.value) == errors[0]
        finally:
            channel.close()
            peer.close()
            channel._thread.join(3)


@pytest.mark.parametrize('failure, detail', [
    ('broken-pipe', 'BrokenPipeError errno='),
    ('send-queue', 'send queue timed out'),
])
def test_send_failure_survives_reader_shutdown(failure, detail, monkeypatch):
    from agent import isolated_channel
    local, peer = socket.socketpair()
    channel = Channel(local, 'private-token', lambda *_: None)
    if failure == 'broken-pipe':
        local.shutdown(socket.SHUT_WR)
    else:
        monkeypatch.setattr(isolated_channel, 'TRANSFER_TIMEOUT', 0.05)
        channel._send_lock.acquire()
    try:
        with pytest.raises(RuntimeError) as caught:
            channel.call('private-op', 'private-payload')
        assert detail in str(caught.value)
        assert 'private-' not in str(caught.value)
        channel._thread.join(3)
        assert not channel._thread.is_alive()
        assert channel.disconnect_reason == str(caught.value)
        channel.close()
        assert channel.disconnect_reason == str(caught.value)
    finally:
        if failure == 'send-queue':
            channel._send_lock.release()
        channel.close()
        peer.close()


def test_explicit_close_is_quiet_and_does_not_create_failure(caplog):
    local, peer = socket.socketpair()
    disconnected = threading.Event()
    channel = Channel(local, 'private-token', lambda *_: None, disconnected.set)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(channel.call, 'run', timeout=None)
        try:
            receive(peer)
            channel.close()
            with pytest.raises(RuntimeError, match='local close'):
                pending.result(timeout=3)
            assert disconnected.wait(3)
            assert channel.disconnect_reason is None
            assert not caplog.records
        finally:
            channel.close()
            peer.close()


def test_failure_reason_never_formats_arbitrary_exception_data():
    from agent.isolated_diagnostics import failure_reason
    for exc in (ValueError('private-secret'), KeyError('private-secret'),
                OSError(5, 'private-secret', '/private-path'),
                ValueError(['private-unhashable'])):
        text = failure_reason(exc)
        assert 'private-' not in text
        assert len(text) < 100


def test_inflight_reply_cannot_overwrite_first_disconnect(monkeypatch):
    from agent import isolated_channel
    original_send, original_receive = isolated_channel.send_message, isolated_channel.receive_message
    received, release_reader, release_caller = (threading.Event() for _ in range(3))
    def gated_receive(*args):
        message = original_receive(*args)
        received.set()
        assert release_reader.wait(3)
        return message
    def gated_send(*args):
        original_send(*args)
        assert release_caller.wait(3)
    monkeypatch.setattr(isolated_channel, 'receive_message', gated_receive)
    monkeypatch.setattr(isolated_channel, 'send_message', gated_send)
    local, peer = socket.socketpair()
    peer.settimeout(3)
    channel = Channel(local, 'token', lambda *_: None)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(channel.call, 'run', timeout=None)
        try:
            request = receive(peer)
            send(peer, {'v': 1, 'generation': 'token', 'reply': request['id'],
                        'error': 'late private error'})
            assert received.wait(3)
            channel.close(reason='first transport failure')
            release_reader.set()
            channel._thread.join(3)
            assert not channel._thread.is_alive()
            release_caller.set()
            with pytest.raises(RuntimeError, match='^first transport failure$'):
                future.result(timeout=3)
        finally:
            release_reader.set()
            release_caller.set()
            channel.close()
            peer.close()
