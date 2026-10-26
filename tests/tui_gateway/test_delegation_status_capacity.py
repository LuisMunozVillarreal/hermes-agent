"""Read-only delegation status must survive unavailable unlimited admission."""

from types import SimpleNamespace

import pytest
import hermes_yaml as yaml

from agent.workload_registry import register_provider, restore_registration
from tools import delegate_tool_registry as live
from tui_gateway import server


@pytest.mark.parametrize("provider_state", ["valid", "disabled", "missing", "unsupported", "not-memory"])
@pytest.mark.parametrize("limit", [0, 7, None])
def test_status_reports_configured_capacity_and_live_state(tmp_path, monkeypatch, provider_state, limit):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("DELEGATION_MAX_CONCURRENT_CHILDREN", raising=False)
    monkeypatch.setattr(live, "_active_subagents", {})
    monkeypatch.setattr(live, "_spawn_paused", False)
    delegation = {"max_spawn_depth": 4}
    if limit is not None:
        delegation["max_concurrent_children"] = limit
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "delegation": delegation,
        "workloads": {
            "mode": "disabled" if provider_state == "disabled" else "required",
            "provider": "status-memory",
        },
    }))
    provider = SimpleNamespace(
        name="status-memory", api_version=1,
        supported_kinds={"terminal"} if provider_state == "unsupported" else {"delegation"},
        memory_managed=provider_state != "not-memory",
    )
    if provider_state != "missing":
        register_provider(provider)
    record = {"subagent_id": "status-child", "goal": "Existing child", "status": "running"}
    live._register_subagent({**record, "agent": object()})
    try:
        for paused in (False, True):
            server._methods["delegation.pause"](1, {"paused": paused})
            response = server._methods["delegation.status"](2, {})
            assert response == {
                "jsonrpc": "2.0", "id": 2,
                "result": {
                    "active": [record], "paused": paused, "max_spawn_depth": 4,
                    "max_concurrent_children": 10 if limit is None else limit,
                },
            }
        live._active_subagents.clear()
        assert server._methods["delegation.status"](3, {})["result"]["active"] == []
    finally:
        restore_registration(provider.name, provider, None)
