"""Delegation turn guard must preserve memory-managed work and control calls."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.workload_registry import register_provider
from run_agent import AIAgent
from tools import delegate_tool as delegate


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delegate, "_load_config", lambda: {"max_concurrent_children": 0})
    (tmp_path / "config.yaml").write_text(
        "workloads:\n  mode: required\n  provider: turn-memory\n", encoding="utf-8",
    )
    provider = SimpleNamespace(name="turn-memory", api_version=1,
                               supported_kinds={"delegation"}, memory_managed=True)
    register_provider(provider)
    return tmp_path, provider


def _call(call_id, name="delegate_task", **args):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(args),
    ))


def test_unlimited_guard_preserves_mixed_batch(configured):
    calls = [
        _call("spawn", goal="Check the memory-managed execution path"),
        _call("read", "read_file", path="example.txt"),
        *[_call(action, action=action) for action in ("list", "steer", "stop")],
        *[_call(f"spawn-{i}", goal=f"Check independent task {i}") for i in range(12)],
    ]
    assert AIAgent._cap_delegate_task_calls(calls) is calls


def _unavailable(configured, mode):
    home, provider = configured
    if mode == "disabled":
        (home / "config.yaml").write_text("workloads:\n  mode: disabled\n", encoding="utf-8")
    elif mode == "missing":
        (home / "config.yaml").write_text(
            "workloads:\n  mode: required\n  provider: absent\n", encoding="utf-8",
        )
    elif mode == "unsupported":
        provider.supported_kinds = {"terminal"}
    else:
        provider.memory_managed = False


@pytest.mark.parametrize("mode", ["disabled", "missing", "unsupported", "not-memory"])
@pytest.mark.parametrize("mixed", [False, True])
def test_unavailable_provider_does_not_abort_guard(configured, mode, mixed):
    _unavailable(configured, mode)
    calls = [_call("read", "read_file", path="example.txt")]
    if mixed:
        calls += [_call("spawn", goal="Must fail as a tool result")]
        calls += [_call(action, action=action) for action in ("list", "steer", "stop")]
    assert AIAgent._cap_delegate_task_calls(calls) is calls


@pytest.mark.parametrize("limit", [1, 3, 10])
def test_positive_guard_caps_spawns_not_controls(configured, monkeypatch, limit):
    monkeypatch.setattr(delegate, "_load_config", lambda: {"max_concurrent_children": limit})
    spawns = [_call(f"spawn-{i}", goal=f"Check task {i}") for i in range(limit + 1)]
    controls = [_call(action, action=f" {action.upper()} ") for action in ("list", "steer", "stop")]
    read = _call("read", "read_file", path="example.txt")
    calls = [controls[0], *spawns, read, *controls[1:]]
    assert AIAgent._cap_delegate_task_calls(calls) == [controls[0], *spawns[:limit], read, *controls[1:]]


def _agent(home):
    definitions = [{"type": "function", "function": {
        "name": name, "description": "test tool",
        "parameters": {"type": "object", "properties": {}},
    }} for name in ("delegate_task", "read_file")]
    with (
        patch("model_tools.get_tool_definitions", return_value=definitions),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    return agent


def _round(agent, calls):
    from agent.turn_tool_round import run_tool_round

    messages = []
    verdict = run_tool_round(
        agent, assistant_message=SimpleNamespace(content="", tool_calls=calls), finish_reason="tool_calls",
        messages=messages, conversation_history=[], api_call_count=1, effective_task_id="turn-capacity",
        user_message="Check delegation", system_message="", active_system_prompt="", compression_attempts=0,
        max_compression_attempts=1, final_response=None, failed=False, _turn_exit_reason=None,
        truncated_tool_call_retries=0, current_turn_user_idx=0,
    )
    assert verdict.action == "continue"
    assert not verdict.failed
    staged = next(row for row in messages if row["role"] == "assistant")
    results = {row["tool_call_id"]: json.loads(row["content"]) for row in messages if row["role"] == "tool"}
    assert [row["id"] for row in staged["tool_calls"]] == [call.id for call in calls]
    assert set(results) == {call.id for call in calls}
    return results


@pytest.mark.parametrize("mode", ["disabled", "missing", "unsupported", "not-memory"])
@pytest.mark.parametrize("mixed", [False, True])
def test_tool_round_keeps_tools_and_controls_when_provider_unavailable(configured, monkeypatch, mode, mixed):
    home, _ = configured
    agent = _agent(home)
    _unavailable(configured, mode)
    monkeypatch.setattr(delegate, "_resolve_delegation_credentials", lambda *args: {})
    target = home / "readable.txt"
    target.write_text("unrelated file remains readable", encoding="utf-8")
    calls = [_call("read", "read_file", path=str(target))]
    child = SimpleNamespace(steer=MagicMock(return_value=True), interrupt=MagicMock())
    delegate._register_subagent({"subagent_id": "owned-child", "agent": child,
                                "owner_agent_session_id": agent.session_id, "status": "running"})
    if mixed:
        calls += [_call("spawn", goal="Must not construct or launch a child")]
        calls += [_call(action, action=action, subagent_id="owned-child", message="Keep working")
                  for action in ("list", "steer", "stop")]
    try:
        with patch.object(delegate, "_build_child_agent", side_effect=AssertionError("unsafe spawn")) as build:
            results = _round(agent, calls)
        build.assert_not_called()
        assert "unrelated file remains readable" in results["read"]["content"]
        if mixed:
            assert "error" in results["spawn"]
            assert results["list"]["count"] == 1
            assert results["steer"]["status"] == "queued"
            assert results["stop"]["status"] == "interrupt_requested"
            child.steer.assert_called_once_with("Keep working")
            child.interrupt.assert_called_once()
    finally:
        delegate._unregister_subagent("owned-child")


def test_unlimited_round_still_reaches_execution_owner(configured, monkeypatch):
    from agent.workload_provider import WorkloadUnavailable

    home, provider = configured
    # A finite session joins results instead of detaching background workers.
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    agent = _agent(home)
    monkeypatch.setattr(delegate, "_resolve_delegation_credentials", lambda *args: {
        "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key", "api_mode": "chat_completions", "model": "test-model",
    })
    admissions = []

    def refuse(kind, args, *, config, **kwargs):
        admissions.append((kind, kwargs))
        raise WorkloadUnavailable("test owner refuses admission")

    provider.popen = refuse
    calls = [_call(f"spawn-{i}", goal=f"Verify admission for task {i}",
                   estimated_peak_memory_mb=64) for i in range(2)]
    calls.append(_call("list", action="list"))
    results = _round(agent, calls)
    assert len(admissions) == 2
    assert all(kind == "delegation" and options["peak_memory_bytes"] == 64 * 1024 * 1024
               for kind, options in admissions)
    for i in range(2):
        assert results[f"spawn-{i}"]["results"][0]["status"] == "error"
    assert results["list"]["action"] == "list"


@pytest.mark.parametrize("arguments", ["{", "[]", "null", '{"action": 3}', '{"action": "spawn"}'])
def test_invalid_or_spawn_actions_still_count_toward_positive_cap(configured, monkeypatch, arguments):
    monkeypatch.setattr(delegate, "_load_config", lambda: {"max_concurrent_children": 1})
    first, second = _call("first"), _call("second")
    first.function.arguments = arguments
    assert AIAgent._cap_delegate_task_calls([first, second]) == [first]
