"""Request-local memory metadata, separate from shell env and lease identity."""
from contextlib import contextmanager
from contextvars import ContextVar

from agent.workload_provider import positive_bytes

MIB = 1024**2
MAX_MEMORY_MIB = (2**63 - 1) // MIB
PEAK_FIELD = "estimated_peak_memory_mb"
RETAINED_FIELD = "estimated_retained_memory_mb"
_estimates: ContextVar[tuple[frozenset[str], dict[str, int]] | None] = ContextVar("workload_estimates", default=None)


def validate_memory_mb(value):
    """Validate without coercion; stored/model units are MiB, not decimal MB."""
    if value is not None and (type(value) is not int or not 0 < value <= MAX_MEMORY_MIB):
        raise ValueError(f"memory estimate must be positive integer MiB <= {MAX_MEMORY_MIB}")
    return value


def validate_estimates(peak=None, retained=None):
    validate_memory_mb(peak)
    validate_memory_mb(retained)
    if retained is not None and peak is None:
        raise ValueError("retained memory requires a peak estimate and cannot exceed it")
    if retained is not None and peak is not None and retained > peak:
        from agent.workload_provider import WorkloadValidationError
        raise WorkloadValidationError(
            "retained-exceeds-peak", peak_memory_bytes=peak * MIB,
            retained_memory_bytes=retained * MIB,
        )


def estimate_bytes(peak=None, retained=None):
    validate_estimates(peak, retained)
    return {key: value * MIB for key, value in (
        ("peak_memory_bytes", peak), ("retained_memory_bytes", retained)) if value is not None}


@contextmanager
def estimate_scope(kinds, metadata):
    """Replace, never inherit another job's estimates; input already uses bytes."""
    for key, value in metadata.items():
        if key not in {"peak_memory_bytes", "retained_memory_bytes"}:
            raise ValueError("invalid memory estimate metadata")
        positive_bytes(value)
    token = _estimates.set((frozenset(kinds), dict(metadata)))
    try:
        yield
    finally:
        _estimates.reset(token)


def current_estimate(kind):
    scoped = _estimates.get()
    return dict(scoped[1]) if scoped is not None and kind in scoped[0] else {}


def model_peak_required(kind):
    """Only model submissions require a peak; scheduled launches keep defaults."""
    from agent.workload_boundary import execution_provider
    provider = execution_provider(kind)
    return provider is not None and getattr(provider, "memory_managed", False) is True


def validate_model_estimates(kind, peak=None, retained=None):
    from agent.workload_provider import WorkloadValidationError
    validate_estimates(peak, retained)
    if peak is None and model_peak_required(kind):
        raise WorkloadValidationError("missing-peak")


def model_required_fields(kind, required):
    # Copy rather than mutate cached/static schemas. Runtime still fails closed
    # if a configured provider cannot be selected while building tool definitions.
    from agent.workload_provider import WorkloadUnavailable
    try:
        needs_peak = model_peak_required(kind)
    except WorkloadUnavailable:
        needs_peak = False
    return [*required, PEAK_FIELD] if needs_peak else list(required)


def memory_schema(*, retained=False, model_created=False):
    description = (
        "Optional own execution-unit memory cap, included in estimated_peak_memory_mb, "
        "not added to it. Must not exceed peak. For separately admitted nested commands, "
        "reserve peak >= retained + concurrent child peaks. With an explicit peak, "
        "omitting retained uses the entire peak and leaves no child slack. " if retained else
        "Estimated peak memory for the whole process subtree, including parent and descendants. "
        + ("Required for agent-created work under memory-managed admission, even with operator defaults. "
           if model_created else "Required with protection unless an operator default is configured. ")
        + "Include realistic parent and nested-command needs. Growth needs explicit scheduler approval; "
        "a queued child does not automatically grow its parent. "
    )
    return {"type": "integer", "minimum": 1, "maximum": MAX_MEMORY_MIB,
            "description": description + "Positive integer MiB (1 MiB = 1048576 bytes)."}
