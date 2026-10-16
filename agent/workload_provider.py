"""Wire-level workload metadata; policy and execution belong to providers."""

from dataclasses import asdict, dataclass
from abc import ABC, abstractmethod
import re

WORKLOAD_API_VERSION = 1


class WorkloadUnavailable(RuntimeError):
    """Required protection cannot authorize this execution."""


class WorkloadProvider(ABC):
    """Versioned execution-owner contract; a ticket is never authorization.

    Providers advertise supported kinds explicitly. ``popen`` owns admission,
    spawn, termination and completion; unsupported modes must fail closed.
    This API does not authorize serializing agents, callbacks or live clients.

    A provider may explicitly accept ``commissioning_context`` on ``popen``.
    Core supplies a captured effective-home ``profile_key`` and task-scoped
    ``session_key`` (or None), not model kwargs/config labels. Isolated workers
    inherit the root snapshot via authenticated bootstrap, independent of child
    session creation. Legacy worker context remains None; **kwargs alone is not
    opt-in. These are reporting keys, not authority: consumers must pseudonymize
    them before persistence and independently verify execution/lease ownership.
    """

    api_version = WORKLOAD_API_VERSION
    name: str

    supported_kinds = frozenset()

    def popen(self, kind, args, *, config, **kwargs):
        """Own launch and lifetime; return a Popen-compatible handle, never a lease."""
        raise WorkloadUnavailable(f"unsupported workload execution: {kind}")

    def request_growth(self, lease, peak_memory_bytes, *, retained_memory_bytes=None, config):
        """Request absolute subtree targets; provider policy may explicitly deny."""
        raise WorkloadUnavailable("workload growth unsupported")

    @abstractmethod
    def submit(self, spec: "WorkloadSpec") -> dict:
        """Return a queued ticket or raise; never run a caller closure."""


def positive_bytes(value):
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError("memory must be positive integer bytes below 2**63")
    return value


@dataclass(frozen=True)
class WorkloadLease:
    ticket: str
    generation: str

    def __post_init__(self):
        for value in (self.ticket, self.generation):
            if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
                raise ValueError("invalid fenced workload lease")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class WorkloadSpec:
    request_id: str
    kind: str
    owner: str
    domain: str
    peak_memory_bytes: int
    backend: str = "local"
    retained_memory_bytes: int | None = None
    parent_lease: WorkloadLease | None = None

    def __post_init__(self):
        positive_bytes(self.peak_memory_bytes)
        if self.retained_memory_bytes is not None:
            if positive_bytes(self.retained_memory_bytes) > self.peak_memory_bytes:
                raise ValueError("retained memory exceeds subtree peak")
        if self.parent_lease is not None and not isinstance(self.parent_lease, WorkloadLease):
            raise ValueError("invalid parent lease")
        for value in (
            self.request_id,
            self.kind,
            self.owner,
            self.domain,
            self.backend,
        ):
            if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z0-9_.:-]{1,128}", value
            ):
                raise ValueError("workload identifiers must be bounded opaque tokens")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        try:
            value = dict(value)
            if value.get("parent_lease") is not None:
                value["parent_lease"] = WorkloadLease(**value["parent_lease"])
            return cls(**value)
        except TypeError as exc:
            raise ValueError("invalid workload metadata") from exc
