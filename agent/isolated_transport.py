"""Bounded logical messages over bounded frames, with anonymous disk spooling.

A channel serializes one transfer per direction. Each transfer is <=16 MiB;
Channel additionally caps outstanding calls and handlers at eight each. No file
paths or file descriptors are accepted from the peer. Partial spools are closed
on every error, deadline, disconnect and cancellation.
"""
import base64
import json
import tempfile
import time

from agent.isolated_protocol import MAX_FRAME, receive, send, validate_message

MAX_MESSAGE = 16 * 1024 * 1024
CHUNK_BYTES = 192 * 1024
TRANSFER_TIMEOUT = 30


def _check_shape(value, budget, depth=0):
    # Bound JSON expansion before the encoder allocates a single large string.
    if depth > 64:
        raise ValueError('agent IPC nesting exceeds limit')
    budget[0] -= 1
    if isinstance(value, str):
        budget[0] -= len(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError('agent IPC keys must be strings')
            _check_shape(key, budget, depth + 1)
            _check_shape(item, budget, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_shape(item, budget, depth + 1)
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise TypeError('agent IPC requires JSON values')
    if budget[0] < 0:
        raise ValueError('agent IPC message exceeds limit')


def encoded_parts(message, deadline):
    _check_shape(message, [MAX_MESSAGE])
    size = 0
    encoder = json.JSONEncoder(allow_nan=False, ensure_ascii=False, separators=(',', ':'))
    for text in encoder.iterencode(message):
        data = text.encode('utf-8')
        size += len(data)
        if size > MAX_MESSAGE:
            raise ValueError('agent IPC message exceeds limit')
        if time.monotonic() >= deadline:
            raise TimeoutError('agent IPC transfer timed out')
        yield data


def validate_payload(message):
    for _ in encoded_parts(message, time.monotonic() + TRANSFER_TIMEOUT):
        pass


def send_message(sock, message, deadline):
    # Anonymous TemporaryFile is unlinked by the OS; never persists credentials
    # or leaves a named artifact for a later generation to reuse.
    with tempfile.TemporaryFile(mode='w+b') as spool:
        size = 0
        for data in encoded_parts(message, deadline):
            size += len(data)
            spool.write(data)
        spool.seek(0)
        if size <= MAX_FRAME:
            # Already validated and encoded; avoid encoding twice.
            from agent.isolated_protocol import send_payload
            send_payload(sock, spool.read(), deadline=deadline)
            return
        identity = message.get('reply', message.get('id'))
        sequence = 0
        while data := spool.read(CHUNK_BYTES):
            send(sock, {'v': 1, 'generation': message['generation'],
                'chunk': identity, 'reply_chunk': 'reply' in message,
                'sequence': sequence, 'size': size,
                'data': base64.b64encode(data).decode('ascii')}, deadline=deadline)
            sequence += 1


def receive_message(sock, generation):
    first = receive(sock)
    if 'chunk' not in first:
        validate_message(first, generation)
        return first
    deadline = time.monotonic() + TRANSFER_TIMEOUT
    total = first.get('size')
    identity, reply = first.get('chunk'), first.get('reply_chunk')
    if type(total) is not int or not MAX_FRAME < total <= MAX_MESSAGE:
        raise ValueError('invalid agent chunk total')
    if type(reply) is not bool:
        raise ValueError('invalid agent chunk direction')
    validate_message({'generation': first.get('generation'),
        **({'reply': identity, 'value': None} if reply else {'id': identity, 'op': 'chunk'})}, generation)
    with tempfile.TemporaryFile(mode='w+b') as spool:
        frame, size, sequence = first, 0, 0
        while size < total:
            if (set(frame) != {'v', 'generation', 'chunk', 'reply_chunk', 'sequence', 'size', 'data'}
                    or frame['generation'] != generation or frame['chunk'] != identity
                    or type(frame['reply_chunk']) is not bool or frame['reply_chunk'] != reply
                    or type(frame['sequence']) is not int or frame['sequence'] != sequence
                    or type(frame['size']) is not int or frame['size'] != total
                    or not isinstance(frame['data'], str)):
                raise ValueError('invalid agent chunk sequence')
            data = base64.b64decode(frame['data'], validate=True)
            if len(data) != min(CHUNK_BYTES, total - size):
                raise ValueError('invalid agent chunk size')
            spool.write(data)
            size += len(data)
            sequence += 1
            if size < total:
                frame = receive(sock, deadline=deadline)
        spool.seek(0)
        message = json.load(spool)
    if not isinstance(message, dict) or message.get('v') != 1:
        raise ValueError('invalid agent chunk message')
    validate_message(message, generation)
    if message.get('reply' if reply else 'id') != identity or ('reply' in message) != reply:
        raise ValueError('invalid agent chunk correlation')
    _check_shape(message, [MAX_MESSAGE])
    return message
