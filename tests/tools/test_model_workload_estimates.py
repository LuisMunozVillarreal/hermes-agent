"""Model-created work requires a peak only with a memory-managed owner."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from agent.workload_registry import register_provider
from run_agent import AIAgent
from tools import delegate_tool, terminal_tool
from tools.registry import registry
from tests.agent.test_delegate_turn_capacity import _agent


@pytest.fixture
def managed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "workloads:\n  mode: required\n  provider: estimate-fixture\n"
        "  peak_memory_bytes: 4294967296\n"
        "  retained_memory_bytes: 2147483648\n"
    )
    provider = SimpleNamespace(
        name="estimate-fixture",
        api_version=1,
        memory_managed=True,
        supported_kinds={
            "delegation",
            "terminal",
            "terminal-pty",
            "cron-script",
            "cron-agent",
        },
    )
    register_provider(provider)
    monkeypatch.setattr(
        delegate_tool, "_load_config", lambda: {"max_concurrent_children": 2}
    )
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda *args: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "fixture-key",
            "api_mode": "chat_completions",
            "model": "fixture-model",
        },
    )
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    return tmp_path, provider


def test_review_uses_operator_defaults_with_required_managed_provider(
    managed, monkeypatch
):
    from agent.review_engine import start_review
    from agent.workload_boundary import execution_provider

    home, provider = managed
    builds = []

    def intercept(**kwargs):
        builds.append(kwargs)
        raise ValueError("review child construction reached")

    monkeypatch.setattr(delegate_tool, "_build_child_agent", intercept)
    assert execution_provider("delegation") is provider
    with pytest.raises(ValueError, match="^review child construction reached$"):
        start_review(
            _agent(home), [{"role": "assistant", "content": "Review this diff."}]
        )
    assert len(builds) == 1
    assert builds[0].get("estimated_peak_memory_mb") is None
    assert builds[0].get("estimated_retained_memory_mb") is None


def test_review_defaults_reach_execution_owner(managed):
    from agent.review_engine import start_review
    from agent.workload_provider import WorkloadUnavailable

    home, provider = managed
    admissions = []

    def refuse(kind, args, *, config, **kwargs):
        admissions.append((kind, config, kwargs))
        raise WorkloadUnavailable("fixture admission stopped")

    provider.popen = refuse
    result = start_review(
        _agent(home), [{"role": "assistant", "content": "Review this diff."}]
    )
    assert result["results"][0]["status"] == "error"
    assert len(admissions) == 1
    kind, config, options = admissions[0]
    assert kind == "delegation"
    assert config["peak_memory_bytes"] == 4096 * 1024**2
    assert config["retained_memory_bytes"] == 2048 * 1024**2
    assert options.get("peak_memory_bytes") is None
    assert options.get("retained_memory_bytes") is None


@pytest.mark.parametrize("entry", ["registry", "live"])
@pytest.mark.parametrize("action", ["list", "steer", "stop"])
def test_model_controls_do_not_require_estimates(managed, entry, action):
    agent = _agent(managed[0])
    args = {"action": action}
    result = json.loads(
        agent._dispatch_delegate_task(args)
        if entry == "live"
        else registry.dispatch("delegate_task", args, parent_agent=agent)
    )
    if action == "list":
        assert "error" not in result
    else:
        assert "subagent_id" in result["error"]
    assert "estimated_peak_memory_mb" not in result.get("error", "")


@pytest.mark.parametrize(
    "args",
    [
        {"goal": "fixture"},
        {"tasks": [{"goal": "fixture"}]},
        {"tasks": json.dumps([{"goal": "fixture"}])},
        {"goal": "fixture", "tasks": []},
        {
            "tasks": [
                {"goal": "valid sibling", "estimated_peak_memory_mb": 256},
                {"goal": "missing peak"},
            ]
        },
        {
            "estimated_peak_memory_mb": 256,
            "tasks": [{"goal": "null override", "estimated_peak_memory_mb": None}],
        },
        {
            "goal": "spoofed internal caller",
            "credentials_cfg": {},
            "_model_submission": False,
            "background": False,
        },
    ],
    ids=[
        "single",
        "batch",
        "json-batch",
        "empty-batch",
        "mixed-batch",
        "null-override",
        "spoof",
    ],
)
@pytest.mark.parametrize("entry", ["registry", "live"])
def test_delegation_missing_peak_refuses_before_child_construction(
    managed, monkeypatch, args, entry
):
    home, _ = managed
    agent = _agent(home)
    builds = []

    def reject(*args, **kwargs):
        builds.append(kwargs)
        raise AssertionError("child construction reached")

    monkeypatch.setattr(delegate_tool, "_build_child_agent", reject)
    result = json.loads(
        agent._dispatch_delegate_task(args)
        if entry == "live"
        else registry.dispatch("delegate_task", args, parent_agent=agent)
    )
    assert "estimated_peak_memory_mb is required" in result.get("error", ""), result
    assert builds == []


@pytest.mark.parametrize(
    "options", [{}, {"background": True}, {"background": True, "pty": True}]
)
def test_terminal_model_missing_peak_refuses_before_execution(
    managed, monkeypatch, options
):
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return json.dumps({"output": "should not run"})

    monkeypatch.setattr(terminal_tool, "terminal_tool", record)
    result = json.loads(
        registry.dispatch("terminal", {"command": "fixture", **options})
    )
    assert "estimated_peak_memory_mb is required" in result.get("error", ""), result
    assert calls == []


@pytest.mark.parametrize("memory_managed", [True, False])
def test_model_schemas_require_only_peak_conditionally(
    managed, monkeypatch, memory_managed
):
    _, provider = managed
    provider.memory_managed = memory_managed
    before = deepcopy(
        (terminal_tool.TERMINAL_SCHEMA, delegate_tool.DELEGATE_TASK_SCHEMA)
    )
    for name in ("terminal", "delegate_task"):
        monkeypatch.setattr(registry.get_entry(name), "check_fn", lambda: True)
    schemas = {
        d["function"]["name"]: d["function"]["parameters"]
        for d in registry.get_definitions({"terminal", "delegate_task"})
    }
    params = [
        schemas["terminal"],
        schemas["delegate_task"]["properties"]["tasks"]["items"],
    ]
    for schema in params:
        assert ("estimated_peak_memory_mb" in schema["required"]) is memory_managed
        assert "estimated_retained_memory_mb" not in schema["required"]
    assert (terminal_tool.TERMINAL_SCHEMA, delegate_tool.DELEGATE_TASK_SCHEMA) == before


@pytest.mark.parametrize("mode", ["disabled", "legacy-provider"])
def test_unmanaged_terminal_omission_still_dispatches(managed, monkeypatch, mode):
    home, provider = managed
    if mode == "disabled":
        (home / "config.yaml").write_text("workloads:\n  mode: disabled\n")
    else:
        provider.memory_managed = False
    calls = []
    monkeypatch.setattr(
        terminal_tool,
        "terminal_tool",
        lambda **kwargs: calls.append(kwargs) or '{"output":"legacy"}',
    )
    assert (
        json.loads(registry.dispatch("terminal", {"command": "fixture"}))["output"]
        == "legacy"
    )
    assert calls[0]["estimated_peak_memory_mb"] is None


@pytest.mark.parametrize("retained", [None, 128])
def test_terminal_optional_retained_reaches_provider(managed, monkeypatch, retained):
    from agent import workload_boundary
    from agent.workload_provider import WorkloadUnavailable

    _, provider = managed
    admissions = []

    def refuse(kind, args, *, config, **kwargs):
        admissions.append(kwargs)
        raise WorkloadUnavailable("fixture admission stopped")

    provider.popen = refuse

    def plan(*args, **kwargs):
        workload_boundary.popen("terminal", ["fixture"])
        raise AssertionError("fixture must not launch")

    monkeypatch.setattr(terminal_tool, "_plan_execution", plan)
    args = {"command": "fixture", "estimated_peak_memory_mb": 256}
    if retained is not None:
        args["estimated_retained_memory_mb"] = retained
    result = json.loads(registry.dispatch("terminal", args))
    assert result.get("error")
    assert len(admissions) == 1
    assert admissions[0]["peak_memory_bytes"] == 256 * 1024**2
    assert admissions[0].get("retained_memory_bytes") == (
        retained * 1024**2 if retained else None
    )


@pytest.mark.parametrize("entry", ["terminal", "delegation"])
def test_model_invalid_pair_reports_safe_numbers(managed, monkeypatch, entry):
    monkeypatch.setattr(
        terminal_tool,
        "terminal_tool",
        lambda **kwargs: pytest.fail("invalid pair executed"),
    )
    fields = {"estimated_peak_memory_mb": 256, "estimated_retained_memory_mb": 257}
    args = (
        {"command": "private-payload", **fields}
        if entry == "terminal"
        else {
            "tasks": [{"goal": "private-payload", **fields}],
        }
    )
    result = json.loads(
        registry.dispatch(
            "terminal" if entry == "terminal" else "delegate_task",
            args,
            parent_agent=_agent(managed[0]),
        )
    )
    assert "269484032" in result["error"] and "268435456" in result["error"]
    assert "must not exceed" in result["error"]
    assert "private-payload" not in result["error"]


@pytest.mark.parametrize("shape", ["single", "batch"])
@pytest.mark.parametrize(
    "memory",
    [
        {"estimated_peak_memory_mb": 256},
        {"estimated_peak_memory_mb": 512, "estimated_retained_memory_mb": 256},
        {},
    ],
)
def test_live_delegation_estimates_reach_execution_owner(managed, shape, memory):
    from agent.workload_provider import WorkloadUnavailable

    home, provider = managed
    provider.memory_managed = bool(
        memory
    )  # Omission remains valid for legacy providers.
    admissions = []

    def refuse(kind, args, *, config, **kwargs):
        admissions.append(kwargs)
        raise WorkloadUnavailable("fixture admission stopped")

    provider.popen = refuse
    task = {"goal": "fixture admission", **memory}
    args = task if shape == "single" else {"tasks": [task]}
    result = json.loads(_agent(home)._dispatch_delegate_task(args))
    assert result["results"][0]["status"] == "error", result
    assert len(admissions) == 1
    assert admissions[0].get("peak_memory_bytes") == (
        memory["estimated_peak_memory_mb"] * 1024**2 if memory else None
    )
    retained = memory.get("estimated_retained_memory_mb")
    assert admissions[0].get("retained_memory_bytes") == (
        retained * 1024**2 if retained else None
    )


def test_automatic_cron_scope_keeps_operator_defaults(managed):
    from agent.workload_boundary import popen
    from agent.workload_estimates import estimate_scope
    from cron.workload_estimates import with_job_estimates

    _, provider = managed
    admissions = []

    def record(kind, args, *, config, **kwargs):
        admissions.append((kind, config, kwargs))
        return "fixture-owned"

    provider.popen = record

    @with_job_estimates
    def scheduled(job):
        assert popen("cron-script", ["fixture"]) == "fixture-owned"
        assert popen("cron-agent", ["fixture"]) == "fixture-owned"

    with estimate_scope({"cron-script", "cron-agent"}, {"peak_memory_bytes": 1}):
        scheduled({})
    assert all(options == {} for _, _, options in admissions)
    assert all(
        config["peak_memory_bytes"] == 4096 * 1024**2 for _, config, _ in admissions
    )
