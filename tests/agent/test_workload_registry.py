import importlib
import pytest
from agent.workload_provider import WorkloadSpec, WorkloadUnavailable


def registry():
    try:
        return importlib.import_module("agent.workload_registry")
    except ModuleNotFoundError:
        pytest.fail("required workload registry is missing")


class Provider:
    api_version = 1

    def submit(self, spec):
        return {"state": "queued", "ticket": spec.request_id}


def test_disabled_does_not_discover_and_required_never_falls_back():
    r = registry().WorkloadRegistry()
    spec = WorkloadSpec("r", "terminal", "owner", "domain", 123)
    assert r.submit({"mode": "disabled"}, spec) is None
    with pytest.raises(WorkloadUnavailable):
        r.submit({"mode": "required", "provider": "fixture"}, spec)
    p = Provider()
    r.register("fixture", p)
    assert r.submit({"mode": "required", "provider": "fixture"}, spec)["ticket"] == "r"
    with pytest.raises(WorkloadUnavailable):
        r.register("fixture", Provider())
    r.unregister("fixture", p)
    with pytest.raises(WorkloadUnavailable):
        r.submit({"mode": "required", "provider": "fixture"}, spec)


@pytest.mark.parametrize("mode", [None, "enabled", True])
def test_unknown_modes_fail_closed(mode):
    with pytest.raises(WorkloadUnavailable):
        registry().WorkloadRegistry().submit({"mode": mode}, None)


def test_incompatible_and_crashed_providers_fail_closed():
    r = registry().WorkloadRegistry()
    p = Provider()
    p.api_version = 2
    with pytest.raises(WorkloadUnavailable):
        r.register("p", p)
    p.api_version = 1

    def broken(spec):
        raise OSError("private details")

    p.submit = broken
    r.register("p", p)
    with pytest.raises(WorkloadUnavailable, match="provider call failed"):
        r.submit({"mode": "required", "provider": "p"}, None)
