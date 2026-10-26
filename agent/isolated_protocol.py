"""Bounded, JSON-only private agent construction and control frames."""
from dataclasses import dataclass
import json
import os
import socket
import struct

VERSION = 1
RESULT_FIELDS = frozenset({'model', 'session_prompt_tokens', 'session_completion_tokens',
    'session_estimated_cost_usd', 'session_cost_status'})
DELEGATION_FIELDS = frozenset({'_delegate_depth', '_delegate_role', '_subagent_id',
    '_parent_subagent_id', '_delegation_id'})


def validate_metadata(metadata):
    if not isinstance(metadata, dict) or set(metadata) - DELEGATION_FIELDS:
        raise ValueError('invalid agent metadata')
    for key, value in metadata.items():
        if key == '_delegate_depth':
            if type(value) is not int or not 0 <= value <= 100:
                raise ValueError('invalid delegation depth metadata')
        elif key == '_delegate_role':
            if value not in {'leaf', 'orchestrator'}:
                raise ValueError('invalid delegation role metadata')
        elif value is not None and (not isinstance(value, str) or len(value) > 256):
            raise ValueError('invalid delegation identity metadata')
    return metadata


RUN_FIELDS = frozenset('''system_message conversation_history persist_user_message
persist_user_timestamp persist_user_display_kind persist_user_display_metadata
persist_user_platform_id moa_config turn_author relay_metadata'''.split())


def validate_run_options(options):
    if not isinstance(options, dict) or set(options) - RUN_FIELDS:
        raise ValueError('unsupported isolated run options')
    return dict(options)


MAX_FRAME = 1024 * 1024
# Deliberately not introspected from AIAgent: adding an argument is not permission
# to transfer a callback, DB handle, credential, or new authority across IPC.
CONSTRUCTION_FIELDS = frozenset('''model provider base_url api_mode max_iterations
max_tokens enabled_toolsets disabled_toolsets quiet_mode ephemeral_system_prompt
log_prefix platform skip_context_files skip_memory parent_session_id session_id
request_overrides reasoning_config capabilities prefill_messages fallback_model
providers_allowed providers_ignored providers_order provider_sort
provider_require_parameters provider_data_collection openrouter_min_coding_score
acp_command acp_args session_cwd save_trajectories requested_provider
load_soul_identity skip_background_review side_agent'''.split())


def encode(value):
    payload = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    if len(payload) > MAX_FRAME:
        raise ValueError('agent IPC frame exceeds limit')
    return payload


def _ready(sock, *, writing=False, deadline=None):
    import select
    import time
    # A close in another thread need not wake select (and its captured fd may
    # already have been reused). Recheck the socket between bounded waits, even
    # for idle reads without a transfer deadline.
    while True:
        if sock.fileno() < 0:
            raise EOFError('agent control connection closed')
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError('agent IPC transfer timed out')
        interval = 0.1 if remaining is None else min(remaining, 0.1)
        try:
            readable, writable, _ = select.select(
                [] if writing else [sock], [sock] if writing else [], [], interval)
        except ValueError:
            # close can race the conversion from socket object to descriptor.
            if sock.fileno() < 0:
                raise EOFError('agent control connection closed') from None
            raise
        if sock.fileno() < 0:
            raise EOFError('agent control connection closed')
        if readable or writable:
            return


def send_payload(sock, payload, *, deadline=None):
    if not 0 < len(payload) <= MAX_FRAME:
        raise ValueError('invalid agent IPC frame size')
    data = memoryview(struct.pack('!I', len(payload)) + payload)
    while data:
        _ready(sock, writing=True, deadline=deadline)
        try:
            sent = sock.send(data, socket.MSG_DONTWAIT)
        except BlockingIOError:
            continue
        if not sent:
            raise EOFError('agent control connection closed')
        data = data[sent:]


def send(sock, value, *, deadline=None):
    send_payload(sock, encode(value), deadline=deadline)


def _read(sock, size, *, deadline=None):
    chunks = bytearray()
    while len(chunks) < size:
        _ready(sock, deadline=deadline)
        try:
            part = sock.recv(size - len(chunks))
        except BlockingIOError:
            continue
        if not part:
            raise EOFError('agent control connection closed')
        chunks.extend(part)
    return bytes(chunks)


def receive(sock, *, deadline=None):
    import time
    if deadline is None and sock.gettimeout() is not None:
        deadline = time.monotonic() + sock.gettimeout()
    # Idle model turns may be long. Once even one byte arrives, a partial
    # header/body cannot hold the reader (and its pending callers) forever.
    first = _read(sock, 1, deadline=deadline)
    deadline = deadline if deadline is not None else time.monotonic() + 30
    size = struct.unpack('!I', first + _read(sock, 3, deadline=deadline))[0]
    if not 0 < size <= MAX_FRAME:
        raise ValueError('invalid agent IPC frame size')
    value = json.loads(_read(sock, size, deadline=deadline))
    if not isinstance(value, dict) or value.get('v') != VERSION:
        raise ValueError('invalid agent IPC version')
    return value


def validate_message(message, generation):
    if message.get('generation') != generation:
        raise ValueError('stale agent generation')
    reply = 'reply' in message
    key = message.get('reply' if reply else 'id')
    if not isinstance(key, str) or len(key) != 32 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('invalid agent request identity')
    if reply:
        if 'id' in message or ('value' in message) == ('error' in message):
            raise ValueError('invalid agent reply')
        if 'error' in message and not isinstance(message['error'], str):
            raise ValueError('invalid agent error')
    elif not isinstance(message.get('op'), str) or not 0 < len(message['op']) <= 64:
        raise ValueError('invalid agent operation')


def check_peer(sock):
    if hasattr(socket, 'SO_PEERCRED') and hasattr(os, 'getuid'):
        pid, uid, _ = struct.unpack('3i', sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if uid != os.getuid():  # windows-footgun: ok -- gated above by hasattr(os, 'getuid')
            raise PermissionError('foreign agent control peer')
        return pid
    raise RuntimeError('isolated agent control requires SO_PEERCRED')


@dataclass(frozen=True)
class AgentConstruction:
    kind: str
    kwargs: dict

    def __post_init__(self):
        if self.kind not in {'delegation', 'cron-agent'}:
            raise ValueError('unsupported isolated agent kind')
        if set(self.kwargs) - CONSTRUCTION_FIELDS:
            raise ValueError('unsupported agent construction fields')
        from agent.isolated_transport import validate_payload
        validate_payload(self.kwargs)  # Same logical bound as the authenticated transport.

    def to_dict(self):
        return {'v': VERSION, 'kind': self.kind, 'kwargs': self.kwargs}
