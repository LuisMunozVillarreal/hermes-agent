"""Control-plane proxy for an AIAgent constructed only inside its owned process."""
import os
import contextvars
import threading
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import uuid

from agent.isolated_protocol import AgentConstruction, DELEGATION_FIELDS, check_peer, receive


class IsolatedAgent:
    def __init__(self, *, kind, launcher=None, workload_estimate=None, **kwargs):
        self.kind = kind
        from agent.workload_estimates import current_estimate
        self._workload_estimate = dict(current_estimate(kind) if workload_estimate is None else workload_estimate)
        self._launcher = launcher
        self._approval_callback = kwargs.pop('approval_callback', None)
        self._channel = None
        self._context = contextvars.copy_context()
        from agent.isolated_routing import capture_profile
        self._profile_context = capture_profile()
        # copy_context alone does not freeze the default home's env fallback.
        # Pin provider/config selection and approval callbacks to the same home
        # as the authenticated worker, including after ambient profile switches.
        from hermes_constants import set_hermes_home_override
        self._context.run(set_hermes_home_override, self._profile_context['home'])
        from agent.workload_context import capture_commissioning
        self._commissioning_context = capture_commissioning()
        from agent.isolated_approval import capture
        self._approval_policy = capture(kind)
        from agent.isolated_routing import capture as capture_routing
        self._routing_context = capture_routing()
        self._approval_threads = set()
        self._state_lock = threading.RLock()
        self._finished = False
        self._turn_lock = threading.Lock()
        self._generation = None
        self._event_generation = contextvars.ContextVar('isolated-event-generation', default=None)
        self._history = None
        self.tool_progress_callback = kwargs.pop('tool_progress_callback', None)
        self.thinking_callback = kwargs.pop('thinking_callback', None)
        # Resolved runtime key travels only over the authenticated private channel,
        # never argv, queue metadata, or the construction descriptor.
        self._runtime_key = kwargs.pop('api_key', None)
        self._activity = {}
        for local in ('session_db', 'iteration_budget', 'clarify_callback', 'credential_pool'):
            kwargs.pop(local, None)
        self.session_id = kwargs.setdefault('session_id', 'isolated-' + uuid.uuid4().hex)
        self.request = AgentConstruction(kind, kwargs)
        self.__dict__.update(kwargs)
        self._parent_session_id = kwargs.get('parent_session_id')
        self._session_init_model_config = {}
        self._interrupt_requested = False
        self._cancel_event = threading.Event()
        self.process = None
        self._socket = None
        self._directory = None

    def run_conversation(self, user_message, task_id=None, stream_callback=None, **kwargs):
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError('isolated agent turn already running')
        try:
            with self._state_lock:
                if self._interrupt_requested:
                    return self._cancelled_result()
                self._finished = False
                self._generation = secrets.token_hex(32)
                self._channel = None
                self.process = None
                self._activity = {}
            from agent.isolated_protocol import validate_run_options
            options = validate_run_options(kwargs)
            if options.get('conversation_history') is None:
                options['conversation_history'] = self._history
            self._stream_callback = stream_callback
            result = self._context.copy().run(
                self._run_turn, user_message, task_id, stream_callback, **options)
            with self._state_lock:
                if self._interrupt_requested:
                    return self._cancelled_result()
                self._history = result.get('messages')
            return result
        finally:
            with self._state_lock:
                self._finished = True
                self._generation = None
            try:
                self.close()
            finally:
                self._turn_lock.release()

    @staticmethod
    def _cancelled_result():
        return {'interrupted': True, 'completed': False, 'final_response': '', 'messages': []}

    def _turn_event(self, generation, op, payload):
        with self._state_lock:
            if generation != self._generation or self._interrupt_requested:
                return 'deny' if op == 'approval' else None
        token = self._event_generation.set(generation)
        try:
            return self._handle_event(op, payload)
        finally:
            self._event_generation.reset(token)

    def _run_turn(self, user_message, task_id, stream_callback, **kwargs):
        # Delegation seeds its per-task/worktree record after proxy construction.
        from tools.terminal_tool import get_session_cwd
        from agent.runtime_cwd import resolve_agent_cwd
        cwd = get_session_cwd(task_id) or self.request.kwargs.get('session_cwd')
        cwd = str(Path(cwd or self._context.copy().run(resolve_agent_cwd)).expanduser().resolve())
        self._directory = tempfile.TemporaryDirectory(prefix='hai-')
        endpoint = str(Path(self._directory.name) / 'control')
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(endpoint)
        os.chmod(endpoint, 0o600)
        listener.listen(1)
        listener.settimeout(30)
        token = self._generation
        args = [sys.executable, '-m', 'agent.isolated_worker', endpoint]
        from agent.isolated_routing import profile_environment
        options = dict(stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       cwd=str(Path(__file__).resolve().parents[1]), env=profile_environment(self._profile_context))
        try:
            if self._launcher is None:
                from agent.workload_boundary import popen
                from agent.workload_estimates import estimate_scope
                from agent.workload_context import commissioning_scope
                with estimate_scope({self.kind}, self._workload_estimate), commissioning_scope(self._commissioning_context):
                    process = popen(self.kind, args, cancel_event=self._cancel_event, **options)
            else:
                process = self._launcher(args, **options)
            with self._state_lock:
                self.process = process
                if self._interrupt_requested:
                    process.terminate()
                    raise RuntimeError("agent launch cancelled")
            lease = getattr(self.process, 'workload_lease', None)
            if self._launcher is None or lease is not None:
                from agent.workload_provider import WorkloadLease
                lease = WorkloadLease(**(lease or {})).to_dict()
            with self._state_lock:
                if self._interrupt_requested:
                    raise RuntimeError('agent bootstrap cancelled')
                self.process.stdin.write((token + '\n').encode())
                self.process.stdin.close()
            # Poll admission/handshake cancellation rather than waiting the full
            # bootstrap timeout after the child has been terminated.
            import time
            deadline = time.monotonic() + 30
            listener.settimeout(0.2)
            while True:
                if self._cancel_event.is_set():
                    raise RuntimeError('agent bootstrap cancelled')
                try:
                    self._socket, _ = listener.accept()
                    break
                except TimeoutError:
                    if time.monotonic() >= deadline or self.process.poll() is not None:
                        raise RuntimeError('agent worker did not connect')
            self._socket.settimeout(30)
            self.worker_pid = check_peer(self._socket)
            hello = receive(self._socket)
            if not secrets.compare_digest(hello.get('token', ''), token):
                raise PermissionError('invalid agent worker authentication')
            self._socket.settimeout(None)
            from agent.isolated_channel import Channel
            with self._state_lock:
                if self._interrupt_requested:
                    raise RuntimeError('agent bootstrap cancelled')
                self._channel = Channel(self._socket, token,
                    lambda op, payload: self._turn_event(token, op, payload))
            return self._channel.call('run', {'request': self.request.to_dict(),
                'profile_context': self._profile_context,
                'commissioning_context': self._commissioning_context,
                'credential': self._runtime_key,
                'approval_policy': self._approval_policy,
                'routing_context': self._routing_context,
                'task_cwd': cwd,
                'lease': lease,
                'metadata': {key: getattr(self, key) for key in DELEGATION_FIELDS if hasattr(self, key)},
                'run_options': kwargs, 'stream': stream_callback is not None,
                'user_message': user_message, 'task_id': task_id}, timeout=None)
        except (RuntimeError, OSError, EOFError):
            if not self._interrupt_requested:
                raise
            return {'interrupted': True, 'completed': False, 'final_response': '', 'messages': []}
        finally:
            listener.close()

    def _handle_event(self, op, payload):
        generation = self._event_generation.get()
        with self._state_lock:
            if generation != self._generation or self._interrupt_requested:
                return 'deny' if op == 'approval' else None
            if op == 'activity':
                self._activity = payload
            if op == 'metrics':
                from agent.isolated_protocol import RESULT_FIELDS
                if not isinstance(payload, dict) or set(payload) - RESULT_FIELDS:
                    raise ValueError('invalid worker metrics')
                self.__dict__.update(payload)
            callback = {'stream': self._stream_callback, 'progress': self.tool_progress_callback,
                        'thinking': self.thinking_callback}.get(op)
        # Never hold the lifecycle lock while running user code. Capture the
        # callback from THIS generation so a late return cannot target a retry.
        if callback is not None:
            if op == 'thinking':
                callback(payload)
            else:
                callback(*payload['args'], **payload['kwargs'])
        if op == 'approval':
            return self._approve(generation, payload)
        return None

    def _approve(self, generation, payload):
        from agent.isolated_approval import decide
        from tools.interrupt import clear_current_thread_interrupt
        ident = threading.get_ident()
        with self._state_lock:
            if generation != self._generation or self._interrupt_requested:
                return 'deny'
            self._approval_threads.add(ident)
        try:
            choice = self._context.copy().run(decide, payload, self._approval_callback, policy=self._approval_policy)
            with self._state_lock:
                valid = generation == self._generation and not self._interrupt_requested
                return choice if valid and choice in {'once', 'deny', 'session', 'always'} else 'deny'
        finally:
            with self._state_lock:
                self._approval_threads.discard(ident)
            clear_current_thread_interrupt()

    def get_activity_summary(self):
        with self._state_lock:
            channel, generation = self._channel, self._generation
        if channel is not None and generation is not None:
            try:
                activity = channel.call('activity', timeout=2)
                with self._state_lock:
                    if generation == self._generation and not self._interrupt_requested:
                        self._activity = activity or self._activity
            except (RuntimeError, OSError, EOFError):
                pass
        with self._state_lock:
            return dict(self._activity)

    def steer(self, text):
        with self._state_lock:
            channel = self._channel
            if self._finished or self._interrupt_requested or channel is None:
                return False
        return bool(channel.call('steer', text))

    def interrupt(self, message=None, *, hard=True):
        from tools.interrupt import set_interrupt
        with self._state_lock:
            if self._interrupt_requested:
                return False
            self._interrupt_requested = True
            self._cancel_event.set()
            for ident in self._approval_threads:
                set_interrupt(True, ident)
            channel, process = self._channel, self.process
        if channel is not None:
            channel.close()
        if process is not None and process.poll() is None:
            process.terminate()  # Provider terminates the whole verified workload.
        return True

    def _drain_pending_steer(self):
        return None

    def close(self):
        if self._channel is not None:
            self._channel.close()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self.process is not None:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
            if self.process.stderr:
                self.process.stderr.close()
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
