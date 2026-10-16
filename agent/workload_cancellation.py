"""Originating tool cancellation, shared across admission and deadline workers."""
from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time

_current = ContextVar("workload_admission_cancellation", default=None)


class AdmissionCancellation:
    def __init__(self, timeout=None):
        from tools.interrupt import acting_for_tid
        self.thread_ids = (threading.get_ident(), acting_for_tid.get())
        try:
            seconds = None if timeout is None else float(timeout)
        except (TypeError, ValueError):
            seconds = 120.0
        self.deadline = None if seconds is None else time.monotonic() + seconds
        self._cancelled = threading.Event()

    def expired(self):
        return self.deadline is not None and time.monotonic() >= self.deadline

    def is_set(self):
        from tools.interrupt import is_thread_interrupted
        if self.expired() or any(is_thread_interrupted(tid) for tid in self.thread_ids):
            self.set()
        return self._cancelled.is_set()

    def set(self):
        self._cancelled.set()


def terminal_cancellation(timeout=None):
    return _current.get() or AdmissionCancellation(timeout)


@contextmanager
def admission_scope(cancellation):
    token = _current.set(cancellation)
    try:
        yield
    finally:
        _current.reset(token)
