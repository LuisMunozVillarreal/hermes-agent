"""Unread child results must retain provider completion evidence (N01-R1)."""

import json
from types import SimpleNamespace

import pytest

from tools.delegate_tool_child_run import _ChildRun
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_notifications import format_process_notification


@pytest.fixture
def account(monkeypatch):
    from tools import process_registry as module

    registry = ProcessRegistry()
    monkeypatch.setattr(module, "process_registry", registry)
    child = _ChildRun.__new__(_ChildRun)
    child.child = SimpleNamespace()
    child.child_task_id = "sa-completion"

    def collect(code, completion):
        handle = SimpleNamespace(returncode=code, process_completion=completion)
        session = ProcessSession(
            id="proc_unread", command="fixture", process=handle,
            owner_task_id=child.child_task_id, notify_on_complete=True,
        )
        session.append_output("partial result\n")
        session.mark_exited(code)
        # Completion evidence must survive release of the provider handle.
        session.process = None
        registry._finished[session.id] = session
        entry = {"task_index": 0, "status": "completed", "summary": "Child finished"}
        child.account_background_processes(entry)
        return json.loads(json.dumps(entry))

    return collect


def notice(entry):
    return format_process_notification({
        "type": "async_delegation", "delegation_id": "delegation-test",
        "results": [entry], "goals": ["fixture"],
    })


@pytest.mark.parametrize("reason", ["automatic growth denied", "unrecognized"])
def test_unread_unknown_completion_reaches_parent(account, reason):
    completion = {
        "returncode_known": False, "transport": "eof",
        "coordinator": {"ticket": "a" * 32, "disposition": "terminal",
                        "state": "failed", "reason": reason},
    }
    entry = account(255, completion)
    unread, = entry["unread_completions"]
    assert unread["exit_code"] is None
    assert unread.get("process_completion") == completion
    text = notice(entry)
    assert text is not None
    assert "exit status unknown" in text
    assert reason in text
    assert "partial result" in text
    assert "exit code None" not in text
    assert "exit code 255" not in text


@pytest.mark.parametrize("code", [0, 255, -15, -9])
@pytest.mark.parametrize("with_receipt", [False, True])
def test_unread_known_exit_keeps_numeric_status(account, code, with_receipt):
    completion = {"returncode_known": True, "transport": "returncode"} if with_receipt else None
    entry = account(code, completion)
    unread, = entry["unread_completions"]
    assert unread["exit_code"] == code
    if with_receipt:
        assert unread["process_completion"] == completion
    else:
        assert "process_completion" not in unread
    text = notice(entry)
    assert text is not None
    assert f"finished (exit code {code})" in text
    assert "exit status unknown" not in text
    assert "partial result" in text
