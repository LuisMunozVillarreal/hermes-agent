"""Typed admission validation must not relaunch an unchanged terminal request."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from agent import workload_boundary
from agent.workload_provider import WorkloadUnavailable, WorkloadValidationError
from tools import terminal_tool
from tools.registry import registry


@pytest.fixture
def terminal_backend(monkeypatch, tmp_path):
    execute = Mock()
    env = SimpleNamespace(env={}, execute=execute)
    monkeypatch.setattr(terminal_tool, "_acquire_env", lambda *_a: env)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {
        "env_type": "local", "cwd": str(tmp_path), "timeout": 30,
        "host_cwd": None, "modal_mode": "auto", "docker_image": "",
        "singularity_image": "", "modal_image": "", "daytona_image": "",
    })
    monkeypatch.setattr(terminal_tool, "_pre_exec_block", lambda *_a, **_kw: None)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *_a, **_kw: {"approved": True})
    sleep = Mock()
    monkeypatch.setattr(terminal_tool.time, "sleep", sleep)
    return execute, sleep


@pytest.mark.parametrize("reason, peak, retained", [
    ("growth-required", 3072, 1024),
    ("retained-exceeds-peak", 256, 257),
])
def test_handler_returns_typed_validation_without_sleep_or_relaunch(
    monkeypatch, terminal_backend, reason, peak, retained,
):
    execute, sleep = terminal_backend
    error = WorkloadValidationError(
        reason, peak_memory_bytes=peak * 1024**2, retained_memory_bytes=retained * 1024**2,
    )
    launch = Mock(side_effect=error)
    provider = SimpleNamespace(supported_kinds={"terminal"}, popen=launch)
    monkeypatch.setattr(workload_boundary, "workload_config", lambda: {"mode": "required"})
    monkeypatch.setattr(workload_boundary, "select_provider", lambda _config: provider)
    execute.side_effect = lambda command, **_kw: workload_boundary.popen("terminal", [command])

    result = json.loads(registry.dispatch("terminal", {
        "command": "echo ok", "estimated_peak_memory_mb": 256,
    }))

    assert result["exit_code"] == -1
    assert str(error) in result["error"]
    execute.assert_called_once()
    launch.assert_called_once()
    sleep.assert_not_called()


def test_handler_still_retries_transient_backend_failures(terminal_backend):
    execute, sleep = terminal_backend
    execute.side_effect = [
        ConnectionError("connection reset"),
        {"output": "ok\n", "returncode": 0},
    ]

    result = json.loads(registry.dispatch("terminal", {"command": "echo ok"}))

    assert result["exit_code"] == 0
    assert result["output"] == "ok"
    assert execute.call_count == 2
    sleep.assert_called_once_with(2)


@pytest.mark.parametrize("exception", [ValueError, RuntimeError, WorkloadUnavailable])
def test_handler_preserves_legacy_retry_exhaustion(terminal_backend, exception):
    execute, sleep = terminal_backend
    # Matching validation wording is not enough: only the typed error opts out.
    message = str(WorkloadValidationError("growth-required"))
    execute.side_effect = exception(message)

    result = json.loads(registry.dispatch("terminal", {"command": "echo ok"}))

    assert result["exit_code"] == -1
    assert result["error"] == f"Command execution failed: {exception.__name__}: {message}"
    assert execute.call_count == 4
    assert sleep.call_args_list == [call(2), call(4), call(8)]


def test_handler_preserves_timeout_without_retry(terminal_backend):
    execute, sleep = terminal_backend
    execute.side_effect = RuntimeError("backend timeout")

    result = json.loads(registry.dispatch("terminal", {"command": "echo ok"}))

    assert result["exit_code"] == 124
    assert result["error"] == "Command timed out after 30 seconds"
    execute.assert_called_once()
    sleep.assert_not_called()
