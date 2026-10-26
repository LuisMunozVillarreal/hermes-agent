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
    if retained is not None and (peak is None or retained > peak):
        raise ValueError("retained memory requires a peak estimate and cannot exceed it")


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


def memory_schema(*, retained=False):
    description = (
        "Parent process memory retained while descendants run, included in (not added to) "
        "estimated_peak_memory_mb. Must not exceed peak. " if retained else
        "Estimated peak memory for this execution's whole process tree, including parent "
        "and descendants. Required with protection unless an operator default is configured. "
    )
    return {"type": "integer", "minimum": 1, "maximum": MAX_MEMORY_MIB,
            "description": description + "Positive integer MiB (1 MiB = 1048576 bytes)."}
