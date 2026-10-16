"""Live agent interception must retain legacy single-goal memory metadata."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("batched", [False, True])
def test_live_dispatch_preserves_estimates(monkeypatch, tmp_path, batched):
    from run_agent import AIAgent
    from tools import delegate_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", lambda *a: {})
    monkeypatch.setattr(delegate_tool, "_announce_batch", lambda *a: None)
    captured = []

    def stop_before_construction(tasks, *args, **kwargs):
        captured.extend(tasks)
        return [], "construction boundary reached"

    monkeypatch.setattr(delegate_tool, "_build_children", stop_before_construction)
    task = {"goal": "Check memory plumbing", "estimated_peak_memory_mb": 768,
            "estimated_retained_memory_mb": 400}
    args = {"tasks": [task]} if batched else task
    result = json.loads(AIAgent._dispatch_delegate_task(SimpleNamespace(_delegate_depth=0), args))
    assert result["error"] == "construction boundary reached"
    assert captured[0]["estimated_peak_memory_mb"] == 768
    assert captured[0]["estimated_retained_memory_mb"] == 400


def test_live_dispatch_rejects_invalid_estimate_before_construction(monkeypatch, tmp_path):
    from run_agent import AIAgent
    from tools import delegate_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", lambda *a: {})
    monkeypatch.setattr(delegate_tool, "_announce_batch", lambda *a: None)
    monkeypatch.setattr(delegate_tool, "_build_children", lambda *a, **kw: pytest.fail("child constructed"))
    result = json.loads(AIAgent._dispatch_delegate_task(SimpleNamespace(_delegate_depth=0), {
        "goal": "Must not start", "estimated_peak_memory_mb": True,
    }))
    assert "positive integer MiB" in result["error"]


def test_invalid_estimate_does_not_spend_oneshot_child_budget(monkeypatch, tmp_path):
    from agent import oneshot_footprint
    from run_agent import AIAgent
    from tools import delegate_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(oneshot_footprint, "is_single_query_session", lambda: True)
    monkeypatch.setattr(delegate_tool, "_get_oneshot_max_children", lambda: 1)
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", lambda *a: {})
    monkeypatch.setattr(delegate_tool, "_announce_batch", lambda *a: None)
    monkeypatch.setattr(delegate_tool, "_build_children", lambda *a, **kw: ([], "construction boundary reached"))
    parent = SimpleNamespace(_delegate_depth=0)
    invalid = json.loads(AIAgent._dispatch_delegate_task(parent, {
        "goal": "Must not start", "estimated_peak_memory_mb": True,
    }))
    assert "positive integer MiB" in invalid["error"]
    assert getattr(parent, "_oneshot_children_spawned", 0) == 0
    valid = json.loads(AIAgent._dispatch_delegate_task(parent, {
        "goal": "Corrected request", "estimated_peak_memory_mb": 768,
    }))
    assert valid["error"] == "construction boundary reached"
    assert parent._oneshot_children_spawned == 1
