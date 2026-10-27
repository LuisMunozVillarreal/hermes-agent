"""Provider completion provenance survives foreground and background results."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.environments.base import BaseEnvironment
from tools.terminal_tool_result import finalize_foreground_result


COMPLETION = {
    "returncode_known": False,
    "transport": "eof",
    "coordinator": {"ticket": "a" * 32, "disposition": "terminal", "state": "failed",
                    "reason": "automatic growth denied"},
}


class Environment(BaseEnvironment):
    def cleanup(self):
        pass


@pytest.mark.platforms("posix")
def test_foreground_wait_preserves_unknown_completion():
    env = Environment.__new__(Environment)
    proc = subprocess.Popen([sys.executable, "-c", "print('partial'); raise SystemExit(255)"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    proc.process_completion = COMPLETION
    try:
        result = env._wait_for_process(proc, timeout=5)
    finally:
        proc.wait(timeout=5)
    assert result["returncode"] == 255  # internal failure compatibility
    assert result["process_completion"] == COMPLETION
    rendered = json.loads(finalize_foreground_result(
        command="fixture", result=result, env=env, env_type="local", effective_task_id="test",
        task_id="test", session_id=None, session_key="test", workdir=None,
        command_cwd=None, approval_note=None,
    ))
    assert rendered["exit_code"] is None
    assert rendered["process_completion"] == COMPLETION
    assert "unknown" in rendered["error"].lower()
    assert "signal" not in rendered.get("exit_code_meaning", "").lower()
    assert "partial" in rendered["output"]


@pytest.mark.parametrize("code", [0, -15, -9, 255])
def test_known_worker_codes_keep_existing_meaning(code):
    result = {"output": "", "returncode": code,
              "process_completion": {**COMPLETION, "returncode_known": True, "transport": "returncode"}}
    rendered = json.loads(finalize_foreground_result(
        command="fixture", result=result, env=SimpleNamespace(), env_type="local", effective_task_id="test",
        task_id="test", session_id=None, session_key="test", workdir=None,
        command_cwd=None, approval_note=None,
    ))
    assert rendered["exit_code"] == code
    assert rendered["error"] is None
    assert rendered["process_completion"] == result["process_completion"]


@pytest.mark.parametrize("pty", [False, True])
def test_background_receipt_retains_unknown_status_after_handle_release(monkeypatch, pty):
    from tools.process_registry import ProcessRegistry, ProcessSession
    from tools.process_registry_results import load_completed_results

    monkeypatch.setenv("HERMES_SESSION_ID", "completion-owner")
    session = ProcessSession(id="proc_completion", command="fixture", parent_session_id="completion-owner",
                             process=SimpleNamespace(process_completion=COMPLETION),
                             notify_on_complete=True, owner_task_id="completion-worker")
    if pty:
        session._pty, session.process = session.process, None
    session.mark_exited(255)
    assert session.exit_code is None
    assert session.completion_reason == "unknown"
    registry = ProcessRegistry()
    registry._running[session.id] = session
    session.process = None
    session._pty = None
    registry._move_to_finished(session)
    assert registry.unread_completions_owned_by("completion-worker") == [session]
    loaded = load_completed_results()[session.id]
    assert registry._exit_fields(loaded)["process_completion"] == COMPLETION
    assert registry._exit_fields(loaded)["exit_code"] is None
    from tools.process_registry_notifications import format_process_notification, process_completion_display_text
    event = {"session_id": loaded.id, **registry._exit_fields(loaded)}
    text = format_process_notification(event)
    assert "exit status unknown" in text
    assert "automatic growth denied" in text
    assert "exit None" not in process_completion_display_text([event])


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("interrupt", [False, True])
def test_foreground_stop_keeps_receipt_and_executor_status(monkeypatch, interrupt):
    from tools.environments import base

    monkeypatch.setattr(base, "is_interrupted", lambda: interrupt)
    env = Environment.__new__(Environment)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    proc.process_completion = COMPLETION
    try:
        result = env._wait_for_process(proc, timeout=0)
    finally:
        proc.kill()
        proc.wait(timeout=5)
        proc.stdout.close()
    assert result["process_completion"] == COMPLETION
    rendered = json.loads(finalize_foreground_result(
        command="fixture", result=result, env=env, env_type="local", effective_task_id="test",
        task_id="test", session_id=None, session_key="test", workdir=None,
        command_cwd=None, approval_note=None,
    ))
    assert rendered["exit_code"] == (130 if interrupt else 124)
    assert rendered["process_completion"] == COMPLETION


@pytest.mark.parametrize("known,code", [(False, 255), (True, -9), (True, 0)])
def test_background_kill_does_not_invent_sigterm(monkeypatch, known, code):
    from tools.process_registry import ProcessRegistry, ProcessSession

    handle = SimpleNamespace(returncode=code, stdout=None, stderr=None, stdin=None,
                             process_completion={**COMPLETION, "returncode_known": known})
    session = ProcessSession(id="proc_killed_completion", command="fixture", process=handle)
    registry = ProcessRegistry()
    registry._running[session.id] = session
    monkeypatch.setattr(registry, "get", lambda _: session)
    monkeypatch.setattr(registry, "_signal_kill", lambda *args: None)
    monkeypatch.setattr(registry, "_stop_scope", lambda *args: True)
    monkeypatch.setattr(registry, "_post_kill_survivors", lambda *args: [])
    result = registry.kill_process(session.id)
    assert result["status"] == "killed"
    assert session.exit_code == (code if known else None)
    assert result["process_completion"] == handle.process_completion
