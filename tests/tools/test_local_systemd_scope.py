import os
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from tools.environments.local import LocalEnvironment


pytestmark = pytest.mark.linux_only


def _environment(tmp_path):
    with patch.object(LocalEnvironment, "init_session", return_value=None):
        return LocalEnvironment(cwd=str(tmp_path), timeout=10)


def _fake_process(pid=4321):
    proc = MagicMock()
    proc.pid = pid
    proc.stdout = MagicMock()
    proc.stdin = MagicMock()
    proc.poll.return_value = None
    return proc


def test_foreground_gateway_command_runs_in_transient_scope(tmp_path, monkeypatch):
    env = _environment(tmp_path)
    monkeypatch.setattr("tools.environments.local._find_bash", lambda: "/bin/bash")
    proc = _fake_process()
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(
        "tools.environments.local._is_supervised_gateway_process", lambda: True
    )
    monkeypatch.setattr(
        "tools.environments.local._systemd_run_user_scope_available", lambda: True
    )
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("tools.environments.local.os.getpgid", lambda pid: pid)

    with patch("tools.environments.local.subprocess.Popen", side_effect=fake_popen):
        result = env._run_bash("printf scoped")

    assert result is proc
    argv = captured["argv"]
    assert argv[0] == "/usr/bin/systemd-run"
    assert argv[1:4] == ["--user", "--scope", "--quiet"]
    unit_index = argv.index("--unit")
    unit_name = argv[unit_index + 1]
    assert unit_name.startswith("hermes-worker-foreground-")
    assert getattr(proc, "_hermes_systemd_unit") == f"{unit_name}.scope"
    properties = [
        argv[index + 1]
        for index, value in enumerate(argv[:-1])
        if value == "--property"
    ]
    assert "MemoryAccounting=yes" in properties
    assert "OOMPolicy=kill" in properties
    assert "TimeoutStopSec=3s" in properties
    memory_max = next(value for value in properties if value.startswith("MemoryMax="))
    assert int(memory_max.partition("=")[2]) > 0
    separator = argv.index("--")
    assert argv[separator + 1 : separator + 3] == ["/bin/bash", "-c"]
    assert argv[separator + 3] == "printf scoped"
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["cwd"] == str(tmp_path)


@pytest.mark.parametrize(
    ("gateway", "scope_available"),
    [(False, True), (True, False)],
)
def test_foreground_non_gateway_or_unavailable_scope_keeps_direct_spawn(
    tmp_path, monkeypatch, gateway, scope_available
):
    env = _environment(tmp_path)
    monkeypatch.setattr("tools.environments.local._find_bash", lambda: "/bin/bash")
    proc = _fake_process()
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(
        "tools.environments.local._is_supervised_gateway_process", lambda: gateway
    )
    monkeypatch.setattr(
        "tools.environments.local._systemd_run_user_scope_available",
        lambda: scope_available,
    )
    monkeypatch.setattr("tools.environments.local.os.getpgid", lambda pid: pid)

    with patch("tools.environments.local.subprocess.Popen", side_effect=fake_popen):
        env._run_bash("printf direct")

    assert captured["argv"] == ["/bin/bash", "-c", "printf direct"]
    assert captured["kwargs"]["start_new_session"] is True
    assert "_hermes_systemd_unit" not in proc.__dict__


def test_foreground_timeout_cleanup_stops_entire_scope_before_pid_group(
    tmp_path, monkeypatch
):
    env = _environment(tmp_path)
    proc = _fake_process()
    proc._hermes_systemd_unit = "hermes-worker-foreground-test.scope"
    stopped = []

    monkeypatch.setattr(
        "tools.environments.local._stop_systemd_unit",
        lambda unit: stopped.append(unit) or True,
    )
    getpgid = MagicMock(return_value=proc.pid)
    killpg = MagicMock()
    monkeypatch.setattr("tools.environments.local.os.getpgid", getpgid)
    monkeypatch.setattr("tools.environments.local.os.killpg", killpg)

    env._kill_process(proc)

    assert stopped == ["hermes-worker-foreground-test.scope"]
    proc.wait.assert_called_once_with(timeout=2.0)
    getpgid.assert_not_called()
    killpg.assert_not_called()


def test_foreground_scope_stop_failure_falls_back_to_process_group(
    tmp_path, monkeypatch
):
    env = _environment(tmp_path)
    proc = _fake_process()
    proc._hermes_systemd_unit = "hermes-worker-foreground-test.scope"
    proc.poll.return_value = 0
    calls = []

    monkeypatch.setattr("tools.environments.local._stop_systemd_unit", lambda unit: False)
    monkeypatch.setattr("tools.environments.local.os.getpgid", lambda pid: proc.pid)

    def fake_killpg(pgid, sig):
        calls.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr("tools.environments.local.os.killpg", fake_killpg)

    env._kill_process(proc)

    assert calls[0] == (proc.pid, signal.SIGTERM)


def test_cached_probe_with_missing_binary_does_not_record_false_scope(
    tmp_path, monkeypatch
):
    env = _environment(tmp_path)
    proc = _fake_process()
    captured = {}

    monkeypatch.setattr("tools.environments.local._find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(
        "tools.environments.local._is_supervised_gateway_process", lambda: True
    )
    monkeypatch.setattr(
        "tools.environments.local._systemd_run_user_scope_available", lambda: True
    )
    monkeypatch.setattr(
        "tools.systemd_scope.shutil.which",
        lambda name: None if name == "systemd-run" else f"/usr/bin/{name}",
    )
    monkeypatch.setattr("tools.environments.local.os.getpgid", lambda pid: pid)

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        return proc

    with patch("tools.environments.local.subprocess.Popen", side_effect=fake_popen):
        env._run_bash("sleep 300")

    assert captured["argv"] == ["/bin/bash", "-c", "sleep 300"]
    assert "_hermes_systemd_unit" not in proc.__dict__


def test_real_scope_timeout_reaps_setsid_sigterm_ignoring_descendant(
    tmp_path, monkeypatch
):
    import tools.systemd_scope as scope

    monkeypatch.setattr(scope, "_SYSTEMD_SCOPE_AVAILABLE", None)
    if not scope._systemd_run_user_scope_available():
        pytest.skip("systemd-run --user --scope unavailable")

    child = tmp_path / "child.py"
    pid_file = tmp_path / "child.pid"
    child.write_text(
        "import os, signal, time\n"
        "os.setsid()\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(300)\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(child)!r}])\n"
        "from pathlib import Path\n"
        "deadline = time.monotonic() + 5\n"
        f"while not Path({str(pid_file)!r}).exists():\n"
        "    assert time.monotonic() < deadline\n"
        "    time.sleep(0.01)\n"
        "time.sleep(300)\n"
    )

    env = _environment(tmp_path)
    monkeypatch.setattr(
        "tools.environments.local._is_supervised_gateway_process", lambda: True
    )
    started = time.monotonic()
    result = env.execute(f"{sys.executable} {parent}", timeout=1)
    elapsed = time.monotonic() - started

    assert result["returncode"] == 124, result
    child_pid = int(pid_file.read_text())
    for _ in range(40):
        if not os.path.exists(f"/proc/{child_pid}"):
            break
        time.sleep(0.05)
    assert not os.path.exists(f"/proc/{child_pid}")
    assert elapsed < 10
