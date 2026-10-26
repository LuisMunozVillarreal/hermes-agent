"""Real queue/provider cron lifecycle; no remote model or platform network.

Run serially with the paired app installed and user-systemd available.
"""
import contextvars
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.agent.test_workload_execution import managed_home as _fixture_managed_home
managed_home = _fixture_managed_home
from tests.agent.test_isolated_queue_integration import wait_for as _fixture_wait_for
wait_for = _fixture_wait_for

pytestmark = pytest.mark.platforms("linux")
MIB = 1024 ** 2


def scheduler():
    from cron.scheduler_provider import CronScheduler

    class External(CronScheduler):
        name = "lifecycle-fixture"

        def start(self, *args, **kwargs):
            raise AssertionError("fixture explicitly drives claimed fires")

    return External()


def create_script(home, body):
    from cron.jobs import create_job

    scripts = home / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "fixture.py").write_text(body)
    return create_job(prompt="", schedule="every 1h", name="fixture",
                      script="fixture.py", no_agent=True, deliver="local")


def assert_empty(client, backend):
    wait_for(lambda: client.call("status")["reserved"] == 0)
    rows = client.call("status")["jobs"]
    assert all(backend.empty(r["unit"]) for r in rows if r.get("unit")), rows
    return rows


@pytest.mark.parametrize("when", ["queued", "running"])
@pytest.mark.parametrize("reason", ["cancel", "claim-loss"])
def test_cron_transport_cancel_settles_before_capacity_release(managed_home, monkeypatch, when, reason):
    from cron.jobs import use_cron_store, get_job, load_jobs, save_jobs
    from cron.executions import get_execution
    from cron import scheduler as cron_scheduler
    monkeypatch.setattr(cron_scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.1)
    from hermes_worker_queue.client import Client
    from hermes_worker_queue.enforcement.linux import Systemd

    home, state = managed_home
    client, backend = Client(state), Systemd()
    marker = home / "effect"
    cancel = threading.Event()
    blocker = None
    with use_cron_store(home):
        job = create_script(home, "from pathlib import Path\nimport time\n"
                            f"Path({str(marker)!r}).write_text('x')\ntime.sleep(30)\n")
        assert client.call("status")["jobs"] == []
        if when == "queued":
            blocker = client.call("submit", request="block", peak=128*MIB, ttl=60,
                                  cwd=str(home), argv=["/bin/sleep", "60"])
        trigger = scheduler()
        claim = trigger.claim_fire(job["id"])
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(contextvars.copy_context().run, trigger.fire_claimed,
                                 claim, cancel_event=cancel)
            try:
                row = wait_for(lambda: next((r for r in client.call("status")["jobs"]
                    if r["kind"] == "cron-script" and r["state"] == when), None))
                if when == "running":
                    wait_for(marker.exists)
                if reason == "claim-loss":
                    jobs = load_jobs()
                    jobs[0]["fire_claim"]["by"] = "replacement-owner"
                    save_jobs(jobs)
                else:
                    cancel.set()
                assert future.result(timeout=5) is True
                terminal = next(r for r in client.call("status")["jobs"]
                                if r["ticket"] == row["ticket"])
                assert terminal["state"] == "cancelled", terminal
                assert not marker.exists() if when == "queued" else marker.read_text() == "x"
                attempt = get_execution(claim["execution_id"])
                assert attempt["status"] == "failed", attempt
                if reason == "cancel":
                    assert "Interrupted" in attempt["error"], attempt
                    assert not get_job(job["id"]).get("fire_claim")
                else:
                    assert "stale result was discarded" in attempt["error"], attempt
                    assert get_job(job["id"])["fire_claim"]["by"] == "replacement-owner"
                if blocker:
                    assert client.call("status")["reserved"] == 128*MIB
            finally:
                cancel.set()
                if blocker:
                    client.call("cancel", ticket=blocker["ticket"])
        assert_empty(client, backend)


@pytest.mark.live_system_guard_bypass
def test_restart_safe_cron_real_provider_adopts_and_delivers_once(managed_home, monkeypatch):
    from tests.cron.test_restart_safe_worker import (
        test_managed_gateway_restart_preserves_active_worker_and_single_side_effect,
    )
    from hermes_worker_queue.client import Client
    from hermes_worker_queue.enforcement.linux import Systemd

    home, state = managed_home
    # Reuse the real restart-safe gateway/worker fixture, but require the actual
    # discovered queue provider in its scoped home rather than unmanaged Popen.
    profile = home / "profile"
    profile.mkdir()
    (profile / "plugins").symlink_to(home / "plugins", target_is_directory=True)
    (profile / "config.yaml").write_text((home / "config.yaml").read_text())
    test_managed_gateway_restart_preserves_active_worker_and_single_side_effect(home, monkeypatch)
    rows = assert_empty(Client(state), Systemd())
    assert len(rows) == 1 and rows[0]["kind"] == "cron-script", rows
    assert rows[0]["state"] == "finished"


@pytest.mark.parametrize("when", ["queued", "running"])
def test_crashed_cron_owner_is_unknown_not_replayed(managed_home, when):
    import subprocess
    import sys
    from pathlib import Path
    from cron.jobs import use_cron_store
    from cron.executions import get_execution, recover_interrupted_executions
    from hermes_worker_queue.client import Client
    from hermes_worker_queue.enforcement.linux import Systemd

    home, state = managed_home
    client, backend = Client(state), Systemd()
    marker, claim_file = home / "effect", home / "claimed.json"
    blocker = None
    with use_cron_store(home):
        job = create_script(home, "from pathlib import Path\nimport time\n"
            f"Path({str(marker)!r}).write_text('x')\ntime.sleep(60)\n")
        if when == "queued":
            blocker = client.call("submit", request="block", peak=128*MIB, ttl=90,
                                  cwd=str(home), argv=["/bin/sleep", "90"])
        code = (
            "import json\nfrom pathlib import Path\n"
            "from hermes_cli.plugins import get_plugin_manager\n"
            "get_plugin_manager().discover_and_load()\n"
            "from tests.cron.test_queue_lifecycle_integration import scheduler\n"
            f"s=scheduler(); claim=s.claim_fire({job['id']!r})\n"
            f"Path({str(claim_file)!r}).write_text(json.dumps(claim))\n"
            "s.fire_claimed(claim)\n")
        owner = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        try:
            wait_for(claim_file.exists)
            claim = json.loads(claim_file.read_text())
            row = wait_for(lambda: next((r for r in client.call("status")["jobs"]
                if r["kind"] == "cron-script" and r["state"] == when), None))
            if when == "running":
                wait_for(marker.exists)
            owner.kill()
            owner.communicate(timeout=5)
            wait_for(lambda: next(r for r in client.call("status")["jobs"]
                if r["ticket"] == row["ticket"])["state"] in {"cancelled", "failed"})
            assert backend.empty(row["unit"]) if row["unit"] else True
            assert recover_interrupted_executions() == 1
            recovered = get_execution(claim["execution_id"])
            assert recovered["status"] == "unknown", recovered
            assert recover_interrupted_executions() == 0
            # Re-delivery of the exact dispatch snapshot must not replay an
            # uncertain occurrence, even after its original owner disappeared.
            assert scheduler().fire_claimed(claim)
            assert len(client.call("status")["jobs"]) == (2 if blocker else 1)
            assert not marker.exists() if when == "queued" else marker.read_text() == "x"
            assert get_execution(claim["execution_id"])["status"] == "unknown"
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.communicate(timeout=5)
            if blocker:
                client.call("cancel", ticket=blocker["ticket"])
        assert_empty(client, backend)


def test_queued_recurring_overlap_misfire_and_ttl(managed_home):
    from datetime import timedelta
    from cron.jobs import use_cron_store, get_job, load_jobs, save_jobs, _hermes_now
    from cron.executions import get_execution
    from cron.scheduler_provider import fire_overdue_jobs
    from hermes_worker_queue.client import Client
    from hermes_worker_queue.enforcement.linux import Systemd

    home, state = managed_home
    client, backend = Client(state), Systemd()
    marker = home / "must-not-run"
    # Expiry is opt-in: native defaults preserve the scheduler's own lifecycle.
    from ruamel.yaml import YAML
    yaml = YAML()
    config_path = home / "config.yaml"
    config = yaml.load(config_path)
    config["workloads"]["queue_ttl_seconds"] = 60
    yaml.dump(config, config_path)
    with use_cron_store(home):
        job = create_script(home, f"from pathlib import Path\nPath({str(marker)!r}).touch()\nprint('ok')")
        assert client.call("status")["jobs"] == []
        blocker = client.call("submit", request="block", peak=128*MIB, ttl=90,
                              cwd=str(home), argv=["/bin/sleep", "90"])
        trigger = scheduler()
        claim = trigger.claim_fire(job["id"])
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(contextvars.copy_context().run, trigger.fire_claimed, claim)
            try:
                row = wait_for(lambda: next((r for r in client.call("status")["jobs"]
                    if r["kind"] == "cron-script" and r["state"] == "queued"), None))
                assert trigger.claim_fire(job["id"]) is None
                jobs = load_jobs()
                jobs[0]["next_run_at"] = (_hermes_now() - timedelta(minutes=30)).isoformat()
                save_jobs(jobs)
                assert fire_overdue_jobs(trigger) == 0
                # Exercise the configured native ManagedPopen TTL (60s), not a
                # substituted provider, edited ledger deadline, or mocked clock.
                assert future.result(timeout=70)
                expired = next(r for r in client.call("status")["jobs"]
                               if r["ticket"] == row["ticket"])
                assert expired["state"] == "expired", expired
                assert client.call("status")["reserved"] == 128*MIB
                assert not marker.exists()
                assert get_execution(claim["execution_id"])["status"] == "failed"
                assert get_job(job["id"])["last_status"] != "ok"
                assert not get_job(job["id"]).get("fire_claim")
            finally:
                client.call("cancel", ticket=blocker["ticket"])
        assert_empty(client, backend)
