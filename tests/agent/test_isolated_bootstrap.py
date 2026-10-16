"""Construction provenance, routing, and independent runtime ownership."""
from types import SimpleNamespace
import pytest
from agent.isolated_protocol import AgentConstruction


def test_worker_restores_only_typed_delegation_metadata(monkeypatch, tmp_path):
    import run_agent
    from agent.isolated_worker import construct
    from agent import workload_context
    from tools import delegate_tool_config
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(workload_context, '_worker_context', None)
    seen = {}
    pool = SimpleNamespace()
    def factory(**kwargs):
        from agent.delegation_context import is_delegated_child_context
        seen.update(kwargs)
        seen['delegated'] = is_delegated_child_context()
        return SimpleNamespace(_session_init_model_config={}, _cache_ttl='1h')
    monkeypatch.setattr(run_agent, 'AIAgent', factory)
    monkeypatch.setattr(delegate_tool_config, '_resolve_child_credential_pool', lambda *a: pool)
    lease = {'ticket': 'a'*32, 'generation': 'b'*32}
    metadata = {'_delegate_depth': 2, '_delegate_role': 'leaf', '_subagent_id': 'child',
                '_parent_subagent_id': 'parent', '_delegation_id': 'batch'}
    agent = construct(AgentConstruction('delegation', {'model': 'target', 'provider': 'custom',
        'base_url': 'http://127.0.0.1:1/v1', 'parent_session_id': 'parent-session'}),
        credential='synthetic-key', lease=lease, metadata=metadata)
    try:
        assert workload_context.worker_lease() == lease
        assert seen['delegated']
        assert 'metadata' not in seen and 'lease' not in seen
        assert seen['api_key'] == 'synthetic-key'
        assert seen['model'] == 'target'
        assert agent._delegate_depth == 2
        assert agent._subagent_id == 'child'
        assert agent._delegation_id == 'batch'
        assert agent._session_init_model_config['_delegate_from'] == 'parent-session'
        assert agent._cache_ttl == '5m'
        assert agent._credential_pool is pool
    finally:
        seen['session_db'].close()


def test_invalid_bootstrap_metadata_cannot_set_agent_authority(monkeypatch, tmp_path):
    from agent.isolated_worker import construct
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with pytest.raises(ValueError, match='metadata'):
        construct(AgentConstruction('delegation', {}), metadata={'_yolo': True})


def test_construction_rejects_live_objects_and_credentials():
    with pytest.raises(ValueError):
        AgentConstruction('delegation', {'api_key': 'not-a-descriptor-field'})
    with pytest.raises((ValueError, TypeError)):
        AgentConstruction('delegation', {'model': object()})
