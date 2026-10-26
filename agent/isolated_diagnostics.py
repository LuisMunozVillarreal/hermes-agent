"""Payload-free diagnostics for the private worker control channel.

Never include arbitrary exception text: JSON decoder errors, OS filenames and
handler failures can contain credentials or conversation content.

The parent emits one warning for an unexpected launch/bootstrap failure or
control disconnect. diagnostic_id is fresh per turn, unrelated to the private
authentication nonce or session IDs. Match it to the propagated disconnect
error; use worker_pid and the log timestamp for external process correlation.
exit_code is observed before cleanup, with at most 0.2 seconds of exit waiting.
Negative codes represent signals on POSIX; state=running/exit_code=unknown does
not attribute the failure to the worker or to its later cleanup termination.
EOF alone does not establish OOM, a scheduler decision or any other root cause.

Stderr remains DEVNULL. Arbitrary worker output is not captured or persisted:
bounded truncation alone cannot safely redact credentials and conversation text.
Normal completion and explicit cancellation emit no failure warning. Channel
shutdown retains its first reason; later reply/shutdown races cannot replace it.
"""
import json
import subprocess


_SAFE_REASONS = frozenset({
    'invalid agent IPC frame size', 'invalid agent IPC version',
    'stale agent generation', 'invalid agent request identity',
    'invalid agent reply', 'invalid agent error', 'invalid agent operation',
    'invalid agent chunk total', 'invalid agent chunk direction',
    'invalid agent chunk sequence', 'invalid agent chunk size',
    'invalid agent chunk message', 'invalid agent chunk correlation',
    'agent control concurrency exceeded',
    'agent worker did not connect',
})


def failure_reason(exc):
    # Use built-in class names, not user-defined exception names or __str__.
    classes = (json.JSONDecodeError, UnicodeDecodeError, TimeoutError,
               ConnectionResetError, BrokenPipeError, OSError, EOFError,
               ValueError, KeyError, TypeError, RecursionError, RuntimeError)
    name = next((cls.__name__ for cls in classes if isinstance(exc, cls)), 'Exception')
    if (type(exc) in (ValueError, RuntimeError) and exc.args and type(exc.args[0]) is str
            and exc.args[0] in _SAFE_REASONS):
        return f'{name}: {exc.args[0]}'
    if isinstance(exc, OSError) and type(exc.errno) is int:
        return f'{name} errno={exc.errno}'
    return name


class ChannelDisconnected(RuntimeError):
    """A local transport failure, distinct from a remote handler error."""


def worker_status(process):
    """Bound exit observation; never confuse cleanup termination with failure."""
    if process is None:
        return 'worker_pid=unknown exit_code=unknown state=not-started'
    pid = process.pid
    pid = pid if type(pid) is int else 'unknown'
    try:
        code = process.poll()
        if code is None:
            # EOF can precede waitpid visibility by a few scheduler ticks.
            code = process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        return f'worker_pid={pid} exit_code=unknown state=running'
    except OSError:
        return f'worker_pid={pid} exit_code=unknown state=unavailable'
    if type(code) is not int:
        return f'worker_pid={pid} exit_code=unknown state=unavailable'
    return f'worker_pid={pid} exit_code={code} state=exited'
