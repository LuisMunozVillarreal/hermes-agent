"""Real loader -> provider -> coordinator -> isolated process integration."""

import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import pytest


@pytest.fixture
def managed_home(tmp_path, monkeypatch):
    spec = importlib.util.find_spec("hermes_worker_queue")
    if spec is None:
        pytest.skip("out-of-tree consumer not installed in test venv")
    from hermes_cli.plugins import PluginManager

    source = Path(spec.origin).parents[2]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "hermes-worker-queue").symlink_to(source, target_is_directory=True)
    import tempfile

    queue_dir = tempfile.TemporaryDirectory(prefix="hwq-test-", dir="/tmp")
    state = Path(queue_dir.name)
    monkeypatch.setenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    monkeypatch.setenv(
        "DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus"
    )
    (tmp_path / "config.yaml").write_text(
        "plugins:\n  enabled: [hermes-worker-queue]\n"
        "workloads:\n  mode: required\n  provider: hermes-worker-queue\n"
        f"  state: {state}\n  peak_memory_bytes: 67108864\n"
    )
    manager = PluginManager()
    manager.discover_and_load()
    coordinator = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hermes_worker_queue.cli",
            "--state",
            str(state),
            "serve",
            "--budget",
            "134217728",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([coordinator.stdout], [], [], 10)[0]
        ready = coordinator.stdout.readline().strip()
        assert ready == "ready", coordinator.communicate(timeout=5) if not ready else ready
        yield tmp_path, state
    finally:
        coordinator.terminate()
        coordinator.communicate(timeout=10)
        manager.unload()
        queue_dir.cleanup()


def test_real_local_environment_is_provider_owned(managed_home):
    from tools.environments.local import LocalEnvironment

    home, state = managed_home
    env = LocalEnvironment(cwd=str(home))
    result = env.execute("printf managed-output", timeout=5)
    assert result["returncode"] == 0, result
    assert "managed-output" in result["output"]
    from hermes_worker_queue.client import Client

    jobs = Client(state).call("status")["jobs"]
    assert jobs, "native execution bypassed the provider"
    assert all(row["unit"] for row in jobs)
    env.cleanup()


def test_managed_pty_roundtrip(managed_home):
    from tools.terminal_tool import terminal_tool
    from tools.process_registry import process_registry

    home, _state = managed_home
    result = json.loads(
        terminal_tool(
            "test -t 0 && read answer; printf 'reply:%s' \"$answer\"",
            task_id="managed-pty",
            background=True,
            pty=True,
            workdir=str(home),
        )
    )
    assert not result.get("error"), result
    session = process_registry.get(result["session_id"])
    assert session._pty._hermes_managed
    session._pty.write(b"hello\n")
    session._reader_thread.join(timeout=8)
    assert session.exited
    assert session.exit_code == 0
    assert "reply:hello" in session.output_buffer


def test_cron_script_owned_execution(managed_home):
    from cron.scheduler import run_job

    home, state = managed_home
    scripts = home / "scripts"
    scripts.mkdir()
    (scripts / "small.py").write_text("print('cron-owned')")
    result = run_job({
        "id": "managed-cron",
        "name": "test",
        "no_agent": True,
        "script": "small.py",
    })
    assert result[0] is True, result
    assert "cron-owned" in str(result)
    from hermes_worker_queue.client import Client

    assert Client(state).call("status")["jobs"]


@pytest.mark.parametrize("background", [False, True])
def test_terminal_front_door_uses_provider(managed_home, background):
    from tools.terminal_tool import terminal_tool
    from tools.process_registry import process_registry

    home, _state = managed_home
    result = json.loads(
        terminal_tool(
            "printf front-door",
            task_id="managed-test",
            background=background,
            workdir=str(home),
        )
    )
    assert not result.get("error"), result
    if background:
        registry = process_registry
        session = registry.get(result["session_id"])
        assert session.process._hermes_managed
        assert session.process.wait(timeout=5) == 0
    else:
        assert "front-door" in result["output"]


def test_managed_timeout_promotion_and_cancel(managed_home):
    from tools.terminal_tool import terminal_tool
    from tools.process_registry import process_registry
    from hermes_worker_queue.client import Client

    home, state = managed_home
    result = json.loads(
        terminal_tool(
            "sleep 30", timeout=700, task_id="managed-promoted", workdir=str(home)
        )
    )
    assert result["notify_on_complete"] is True, result
    session = process_registry.get(result["session_id"])
    assert session.process._hermes_managed is True
    assert Client(state).call("status")["reserved"] > 0
    assert process_registry.kill_process(session.id)["status"] == "killed"
    assert Client(state).call("status")["reserved"] == 0


def test_managed_yield_keeps_same_execution(managed_home):
    import contextvars
    import threading
    import time
    from tools import interrupt
    from tools.terminal_tool import terminal_tool
    from tools.process_registry import process_registry
    from hermes_worker_queue.client import Client

    home, state = managed_home
    marker = home / "started"
    box = {}

    def run():
        box["result"] = json.loads(
            terminal_tool(
                f"printf x > {marker}; sleep 30",
                timeout=60,
                task_id="managed-yield",
                workdir=str(home),
            )
        )

    thread = threading.Thread(target=contextvars.copy_context().run, args=(run,))
    thread.start()
    deadline = time.monotonic() + 8
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert marker.exists()
    interrupt.request_yield(thread.ident)
    thread.join(timeout=8)
    assert not thread.is_alive()
    result = box["result"]
    try:
        assert result["status"] == "yielded_to_background", result
        session = process_registry.get(result["session_id"])
        assert session.process._hermes_managed is True
        assert Client(state).call("status")["reserved"] > 0
        assert marker.read_text() == "x"
    finally:
        process_registry.kill_process(result["session_id"])
