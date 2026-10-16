"""Private RPC failures must settle callers, including unbounded run waits."""
import socket
import threading

import pytest

from agent.isolated_channel import Channel
from agent.isolated_protocol import MAX_FRAME, send
from agent.isolated_transport import MAX_MESSAGE


def test_large_messages_roundtrip_without_lifting_frame_bound():
    from agent.isolated_protocol import encode
    a, b = socket.socketpair()
    payload = {'history': [{'role': 'user', 'content': 'é🙂' * (MAX_FRAME // 2)}]}
    with pytest.raises(ValueError, match='frame'):
        encode(dict(v=1, payload=payload))
    client = Channel(a, 'nonce', lambda *_: None)
    server = Channel(b, 'nonce', lambda op, value: value)
    try:
        assert client.call('roundtrip', payload, timeout=8) == payload
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize('response', ['oversize', 'not-json', 'huge-error'])
def test_unencodable_response_settles_unbounded_call(response):
    a, b = socket.socketpair()
    def handle(*_):
        if response == 'huge-error':
            raise ValueError('x' * (MAX_MESSAGE + 1))
        return {'messages': ['x' * (MAX_MESSAGE + 1)]} if response == 'oversize' else object()
    client = Channel(a, 'nonce', lambda *_: None)
    server = Channel(b, 'nonce', handle)
    settled = threading.Event()
    errors = []
    def call():
        try:
            client.call('run', timeout=None)
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            settled.set()
    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    try:
        assert settled.wait(3), 'unbounded RPC wait hung after encoding failure'
        assert errors and 'response' in errors[0]
    finally:
        client.close()
        server.close()
        thread.join(3)


@pytest.mark.parametrize('frame', [
    {'v': 1, 'generation': 'stale', 'id': 'a'*32, 'op': 'run'},
    {'v': 2, 'generation': 'nonce', 'id': 'a'*32, 'op': 'run'},
    {'v': 1, 'generation': 'nonce', 'op': 'run'},
    {'v': 1, 'generation': 'nonce', 'id': [], 'op': 'run'},
    {'v': 1, 'generation': 'nonce', 'reply': [], 'value': None},
])
def test_malformed_or_stale_frames_close_without_dispatch(frame):
    a, b = socket.socketpair()
    called = []
    channel = Channel(a, 'nonce', lambda *args: called.append(args))
    try:
        send(b, frame)
        assert channel.closed.wait(3)
        assert not called
    finally:
        channel.close()
        b.close()
