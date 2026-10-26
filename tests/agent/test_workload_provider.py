"""Workload contracts reject ambiguous units before talking to providers."""

import importlib
import pytest


def api():
    try:
        return importlib.import_module("agent.workload_provider")
    except ModuleNotFoundError:
        pytest.fail("versioned workload contract is missing")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan"), 2**63])
def test_memory_is_positive_integer_bytes(value):
    with pytest.raises(ValueError):
        api().WorkloadSpec("request", "terminal", "owner", "domain", value)


def test_request_roundtrip_contains_no_payload():
    m = api()
    spec = m.WorkloadSpec("request", "terminal", "owner", "domain", 4096)
    assert m.WorkloadSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError):
        m.WorkloadSpec.from_dict({**spec.to_dict(), "command": "secret"})
