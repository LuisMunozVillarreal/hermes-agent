"""Trusted lease provenance and absolute nested-budget growth."""
import pytest
from agent.workload_provider import WorkloadSpec


def test_retained_subtree_metadata_roundtrip():
    from agent.workload_provider import WorkloadLease
    lease = WorkloadLease('a' * 32, 'b' * 32)
    spec = WorkloadSpec('request', 'delegation', 'session', 'local', 4096,
                        retained_memory_bytes=1024, parent_lease=lease)
    assert WorkloadSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError):
        WorkloadSpec('r', 'delegation', 's', 'local', 4096, retained_memory_bytes=8192)
    with pytest.raises(ValueError):
        WorkloadLease('a' * 32, 'stale')


def test_only_bootstrap_can_supply_parent_lease(monkeypatch):
    from agent import workload_boundary as boundary
    from agent.workload_context import install_worker_context
    from hermes_cli import config_effective
    monkeypatch.setattr(config_effective, 'load_user_config_effective', lambda **kw: {
        'workloads': {'mode': 'required', 'parent_lease': {'ticket': 'forged'}}})
    assert 'parent_lease' not in boundary.workload_config()
    import agent.workload_context as context
    monkeypatch.setattr(context, '_worker_context', None)
    lease = {'ticket': 'a' * 32, 'generation': 'b' * 32}
    install_worker_context(lease)
    assert boundary.workload_config()['parent_lease'] == lease


def test_growth_uses_bootstrap_lease_and_returns_explicit_denial(monkeypatch):
    from agent.workload_boundary import request_growth
    import agent.workload_context as context
    from agent import workload_boundary as boundary
    monkeypatch.setattr(context, '_worker_context', None)
    context.install_worker_context({'ticket': 'a' * 32, 'generation': 'b' * 32})
    seen = []
    class Provider:
        def request_growth(self, lease, peak_memory_bytes, *, retained_memory_bytes, config):
            seen.append((lease, peak_memory_bytes, retained_memory_bytes))
            return {'granted': False, 'job': {'peak': 4096, **lease}}
    monkeypatch.setattr(boundary, 'select_provider', lambda cfg: Provider())
    monkeypatch.setattr(boundary, 'workload_config', lambda: {'mode': 'required'})
    result = request_growth(8192, retained_memory_bytes=2048)
    assert result['granted'] is False
    assert seen == [(context.worker_lease(), 8192, 2048)]
    with pytest.raises(ValueError):
        request_growth(True)
