"""Real provider admission must settle cancellation before capacity is released."""
from concurrent.futures import ThreadPoolExecutor
import shlex
import threading

import pytest

from tests.agent.test_isolated_queue_integration import (
    BUDGET, queue_home as _queue_home, wait_for,
)

queue_home = _queue_home
pytestmark = pytest.mark.platforms("linux")


@pytest.mark.parametrize("path,cause", [
    ("foreground", "interrupt"), ("foreground", "timeout"),
    ("background", "interrupt"), ("pty", "interrupt"),
    ("probe", "timeout"),
])
def test_terminal_admission_never_executes_after_cancellation(queue_home, path, cause):
    from tools.environments.local import LocalEnvironment
    from tools.interrupt import set_interrupt
    from tools.process_registry import ProcessRegistry

    home, _, client, _, backend = queue_home
    env = LocalEnvironment(cwd=str(home))
    registry = ProcessRegistry()
    marker = home / "must-not-execute"
    command = f"printf started > {shlex.quote(str(marker))}"
    blocker = client.call("submit", request="terminal-blocker", peak=BUDGET,
                          ttl=60, cwd=str(home), argv=["/bin/sleep", "60"])
    tids = []

    def execute():
        tids.append(threading.get_ident())
        try:
            if path == "probe":
                proc = env._run_bash(command, timeout=1)
                proc.communicate(timeout=5)
                return proc
            if path == "foreground":
                return env.execute(command, timeout=1 if cause == "timeout" else 30)
            return registry.spawn_local(command, cwd=str(home), use_pty=path == "pty")
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(execute)
        try:
            queued = wait_for(lambda: next((r for r in client.call("status")["jobs"]
                              if r["state"] == "queued"), None))
            if cause == "interrupt":
                set_interrupt(True, tids[0])
            # Keep capacity occupied while the originating call settles.
            try:
                result = future.result(timeout=5)
                settled = True
            except TimeoutError:
                settled = False
                result = None
            row = client.call("status", ticket=queued["ticket"])["jobs"][0]
            cancelled_before_release = row["state"] == "cancelled"
            assert not backend.empty(blocker["unit"])
        finally:
            client.call("cancel", ticket=blocker["ticket"])
            future.result(timeout=15)
            if tids:
                set_interrupt(False, tids[0])
    wait_for(lambda: client.call("status")["reserved"] == 0)
    assert not marker.exists(), "payload executed after interrupt/timeout"
    assert settled and cancelled_before_release
    if cause == "timeout" and path == "foreground":
        assert result["returncode"] == 124
    assert all(backend.empty(r["unit"]) for r in client.call("status")["jobs"] if r["unit"])


def test_native_defaults_reach_real_ledger_and_kernel(queue_home):
    import subprocess
    _home, state, client, provider, backend = queue_home
    cap = 64 * 1024 * 1024
    proc = provider.popen("delegation", ["/bin/sleep", "60"], config={
        "state": str(state), "estimates": {"delegation": cap * 2},
        "retained_estimates": {"delegation": cap},
    }, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        row = client.call("status", ticket=proc.workload_lease["ticket"])["jobs"][0]
        assert row["peak"] == cap * 2 and row["retained"] == cap
        assert row["deadline"] == 0 and row["runtime_seconds"] == 0
        assert int((backend.path(row["unit"]) / "memory.max").read_text()) == cap
        runtime = subprocess.check_output([
            "systemctl", "--user", "show", row["unit"], "--property=RuntimeMaxUSec", "--value",
        ], text=True).strip()
        assert runtime == "infinity"
    finally:
        proc.terminate()
        proc.wait(timeout=5)
    assert backend.empty(row["unit"])


def test_cancel_does_not_hide_unverified_provider_cleanup(tmp_path, monkeypatch):
    from tools.environments.local import LocalEnvironment
    from tools.interrupt import set_interrupt
    from agent.workload_provider import WorkloadUnavailable
    env = LocalEnvironment(cwd=str(tmp_path))
    tid = threading.get_ident()

    def failed_cleanup(*args, **kwargs):
        set_interrupt(True, tid)
        raise WorkloadUnavailable("cleanup unverified")

    monkeypatch.setattr(env, "_run_bash", failed_cleanup)
    try:
        with pytest.raises(WorkloadUnavailable, match="cleanup unverified"):
            env.execute("true", timeout=10)
    finally:
        set_interrupt(False, tid)
