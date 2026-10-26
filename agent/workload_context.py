"""Authenticated worker bootstrap context, never model/config supplied identity.

One isolated process owns one root agent. Its tool threads share this immutable
lease; ordinary gateway processes never install it. The provider independently
verifies the submitting PID's actual scope membership.
"""
from contextlib import contextmanager
from contextvars import ContextVar

from agent.workload_provider import WorkloadLease

_worker_context = None
_worker_commissioning = None
_commissioning_scope: ContextVar[dict | None] = ContextVar('workload_commissioning', default=None)


def install_worker_context(lease, commissioning_context=None):
    global _worker_context, _worker_commissioning
    validated = WorkloadLease(**lease)
    ownership = validate_commissioning(commissioning_context)
    if _worker_context is not None:
        raise RuntimeError("worker workload context already installed")
    _worker_commissioning = ownership
    _worker_context = validated


def worker_lease():
    return _worker_context.to_dict() if _worker_context is not None else None


def validate_commissioning(context):
    """Copy the bounded reporting keys; they confer no credential/lease authority."""
    if context is None:
        return None
    if type(context) is not dict or set(context) != {'profile_key', 'session_key'}:
        raise ValueError('invalid commissioning context')
    for key, value in context.items():
        if key == 'session_key' and value is None:
            continue
        if not isinstance(value, str) or not 1 <= len(value) <= 4096 or '\0' in value:
            raise ValueError('invalid commissioning key')
    return dict(context)


def capture_commissioning():
    # Dedicated workers keep the root's commissioning identity even after their
    # AIAgent binds its own SessionDB row or a tool starts on a fresh thread.
    if _worker_context is not None:
        return validate_commissioning(_worker_commissioning)
    captured = _commissioning_scope.get()
    if captured is not None:
        return dict(captured)
    from hermes_constants import hermes_home_key
    from gateway.session_context import get_scoped_session_id
    return validate_commissioning({'profile_key': hermes_home_key(),
                                  'session_key': get_scoped_session_id()})


@contextmanager
def commissioning_scope(context):
    """Restore a trusted launch snapshot, never accepted as a tool/Popen kwarg."""
    token = _commissioning_scope.set(validate_commissioning(context))
    try:
        yield
    finally:
        _commissioning_scope.reset(token)
