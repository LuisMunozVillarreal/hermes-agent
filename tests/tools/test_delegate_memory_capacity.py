"""Unlimited delegation requires a selected memory-admission execution owner."""

from types import SimpleNamespace

import pytest

from agent.workload_provider import WorkloadUnavailable
from agent.workload_registry import register_provider
from tools import delegate_tool as delegate


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delegate, "_load_config", lambda: {"max_concurrent_children": 0})
    (tmp_path / "config.yaml").write_text(
        "workloads:\n  mode: required\n  provider: test-memory\n"
    )
    provider = SimpleNamespace(name="test-memory", api_version=1,
                               supported_kinds={"delegation"}, memory_managed=True)
    register_provider(provider)
    return tmp_path, provider


def test_explicit_unlimited_mode_preserves_large_task_list(configured):
    assert delegate._get_max_concurrent_children() == 0
    assert delegate._get_max_async_children() == 0
    tasks = [{"goal": f"Verify the independent capacity scenario number {i}"} for i in range(12)]
    normalized, error = delegate._normalize_task_list(None, None, tasks, None, "leaf", 0)
    assert error is None
    assert normalized == tasks
    description = delegate._build_tasks_param_description()
    assert "memory" in description.lower()
    assert "up to 0" not in description


@pytest.mark.parametrize("mode", ["disabled", "missing", "unsupported", "not-memory"])
def test_unlimited_never_disables_legacy_protection(configured, mode, monkeypatch):
    home, provider = configured
    if mode == "disabled":
        (home / "config.yaml").write_text("workloads:\n  mode: disabled\n")
    elif mode == "missing":
        (home / "config.yaml").write_text("workloads:\n  mode: required\n  provider: absent\n")
    elif mode == "unsupported":
        provider.supported_kinds = {"terminal"}
    else:
        provider.memory_managed = False
    with pytest.raises(WorkloadUnavailable):
        delegate._get_max_concurrent_children()
    with pytest.raises(WorkloadUnavailable):
        delegate._get_max_async_children()
    import json
    from tools import async_delegation as ad
    monkeypatch.setattr(delegate, "_resolve_delegation_credentials", lambda *args: {})
    result = json.loads(delegate.delegate_task(goal="Check protection", parent_agent=SimpleNamespace()))
    assert "error" in result
    called = []
    with pytest.raises(WorkloadUnavailable):
        ad.dispatch_async_delegation(
            goal="must not run", context=None, toolsets=None, role="leaf",
            model="fixture", session_key="", runner=lambda: called.append(True) or {},
            max_async_children=0,
        )
    assert called == []


def test_unlimited_background_dispatch_does_not_use_a_zero_sized_pool(configured):
    import threading
    from tools import async_delegation as ad
    from tests.waiting import wait_until

    ad._reset_for_tests()
    release = threading.Event()
    def run():
        assert release.wait(5)
        return {"status": "completed", "summary": "done"}
    try:
        for index in range(2):
            receipt = ad.dispatch_async_delegation(
                goal=f"capacity {index}", context=None, toolsets=None, role="leaf",
                model="fixture", session_key="", runner=run, max_async_children=0,
            )
            assert receipt["status"] == "dispatched"
        assert ad.active_count() == 2
    finally:
        release.set()
        wait_until(lambda: ad.active_count() == 0, timeout=5)
        ad._reset_for_tests()


def test_unlimited_parallel_batch_runs_all_children(configured, monkeypatch):
    from tools import delegate_tool_dispatch as dispatch
    monkeypatch.setattr(dispatch, "_report_child_done", lambda *a: None)
    tasks = [{"goal": f"task {i}"} for i in range(2)]
    batch = SimpleNamespace(
        parent_agent=SimpleNamespace(), task_list=tasks, live_deleg_id=None,
        children=[(i, task, SimpleNamespace()) for i, task in enumerate(tasks)],
        max_children=0, unit_id=None,
        run_child=lambda i, *_: {"task_index": i, "status": "completed"},
    )
    results = []
    dispatch._run_children_parallel(batch, results, honor_parent_interrupt=True)
    assert [entry["task_index"] for entry in results] == [0, 1]


def test_schema_reports_unlimited_configuration_when_admission_is_disabled(configured):
    home, _ = configured
    (home / "config.yaml").write_text("workloads:\n  mode: disabled\n")
    description = delegate._build_dynamic_schema_overrides()["parameters"]["properties"]["tasks"]["description"]
    assert "delegation.max_concurrent_children=0" in description
    assert "required memory admission" in description
    assert "up to 10" not in description
    with pytest.raises(WorkloadUnavailable):
        delegate._get_max_async_children()


def test_positive_and_default_legacy_limits_stay_available(configured, monkeypatch):
    for config, expected in [({}, 10), ({"max_concurrent_children": 7}, 7)]:
        monkeypatch.setattr(delegate, "_load_config", lambda: config)
        assert delegate._get_max_concurrent_children() == expected
