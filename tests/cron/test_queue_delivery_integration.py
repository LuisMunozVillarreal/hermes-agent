"""Real cron execution -> durable notification -> crash/retention fencing."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tests.agent.test_workload_execution import managed_home as _fixture_managed_home
managed_home = _fixture_managed_home
from tests.agent.test_isolated_queue_integration import wait_for as _fixture_wait_for
wait_for = _fixture_wait_for
from tests.cron.test_queue_lifecycle_integration import scheduler as _fixture_scheduler, assert_empty as _fixture_assert_empty
scheduler = _fixture_scheduler
assert_empty = _fixture_assert_empty

pytestmark = pytest.mark.platforms("linux")


def pending_completion(home):
    from cron.jobs import create_job, use_cron_store
    from cron.executions import mark_execution_handoff_pending, get_execution
    from cron.delivery_queue import get_status

    scripts = home / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "deliver.py").write_text(
        "from pathlib import Path\n"
        f"p=Path({str(home / 'side-effects')!r})\n"
        "with p.open('a') as f: f.write('x')\nprint('notification-content')\n")
    with use_cron_store(home):
        job = create_job(prompt="", schedule="every 1h", script="deliver.py",
                         no_agent=True, deliver="telegram:fixture-chat")
        claim = scheduler().claim_fire(job["id"])
        assert mark_execution_handoff_pending(claim["execution_id"])
        payload, ack = home / "dispatch.json", home / "adopted.json"
        payload.write_text(json.dumps({"job": claim, "profile_home": str(home),
                                       "multiplex_active": False}))
        # Bound only the offline-gateway wait. Real ownership adoption, plugin
        # discovery, run_job, kernel script and scheduler delivery remain intact.
        code = ("from pathlib import Path\nfrom cron import delivery_queue as q\n"
                "q.DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS = 0\n"
                "from cron.scheduler import _run_external_worker_payload\n"
                f"assert _run_external_worker_payload(Path({str(payload)!r}), Path({str(ack)!r}))\n")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=20, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(ack.read_text())["execution_id"] == claim["execution_id"]
        attempt = get_execution(claim["execution_id"])
        assert attempt["status"] == "completed", attempt
        delivery = get_status(claim["execution_id"])
        assert delivery["status"] == "pending", delivery
        assert "notification-content" in delivery["content"]
        assert json.loads(delivery["job_json"])["execution_id"] == claim["execution_id"]
        return claim, delivery


@pytest.mark.parametrize("outcome", ["delivered", "failed", "unknown"])
def test_delivery_restart_retention_never_replays_execution(managed_home, monkeypatch, outcome):
    from cron import delivery_queue as q
    from hermes_worker_queue.client import Client
    from hermes_worker_queue.enforcement.linux import Systemd

    home, state = managed_home
    claim, _delivery = pending_completion(home)
    rows = assert_empty(Client(state), Systemd())
    assert len(rows) == 1 and rows[0]["state"] == "finished"
    assert home.joinpath("side-effects").read_text() == "x"
    sent = home / "notification-sends"
    code = (
        "from pathlib import Path\nimport time\nfrom cron import delivery_queue as q\n"
        "def send(job, content, failure):\n"
        f" assert job['execution_id'] == {claim['execution_id']!r}\n"
        " assert 'notification-content' in content and failure is False\n"
        f" with Path({str(sent)!r}).open('a') as f: f.write(job['execution_id']+'\\n')\n"
        + (" time.sleep(60)\n" if outcome == "unknown" else "")
        + (" return 'fixture transport refused'\n" if outcome == "failed" else " return None\n")
        + "assert q.drain(send) == 1\n")
    transport = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
    try:
        wait_for(sent.exists)
        if outcome == "unknown":
            transport.kill()
            transport.communicate(timeout=5)
            assert q.recover_abandoned() == 1
        else:
            out, err = transport.communicate(timeout=5)
            assert transport.returncode == 0, out + err
        assert q.get_status(claim["execution_id"])["status"] == outcome
        assert q.recover_abandoned() == 0
        # Prune verbose metadata; tombstones must still suppress the exact
        # historical notification key. This must not depend on queue row retention.
        monkeypatch.setattr(q, "MAX_TERMINAL_DELIVERIES", 0)
        q.recover_abandoned()
        with sqlite3.connect(f"file:{home / 'cron/deliveries.db'}?mode=ro", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            assert db.execute("SELECT count(*) FROM deliveries").fetchone()[0] == 0
            assert db.execute("SELECT execution_id FROM delivery_tombstones").fetchall() == [(claim["execution_id"],)]
        assert q.enqueue(claim["execution_id"], claim, "duplicate")["status"] == outcome
        assert q.drain(lambda *args: pytest.fail("duplicate notification")) == 0
        assert sent.read_text().splitlines() == [claim["execution_id"]]
        assert home.joinpath("side-effects").read_text() == "x"
        assert len(assert_empty(Client(state), Systemd())) == 1
    finally:
        if transport.poll() is None:
            transport.kill()
            transport.communicate(timeout=5)


def test_drain_completion_store_then_replace_required_provider(tmp_path, monkeypatch):
    from agent import workload_registry as registry
    from agent.workload_provider import WorkloadSpec, WorkloadUnavailable
    from hermes_cli.plugins import PluginManager
    from hermes_worker_queue.client import Client
    from hermes_worker_queue.enforcement.linux import Systemd
    from cron import delivery_queue as q

    fixture = managed_home.__wrapped__(tmp_path, monkeypatch)
    home, state = next(fixture)
    client = Client(state)
    try:
        claim, _delivery = pending_completion(home)
        active = client.call("submit", request="active", peak=128*1024**2, ttl=60,
                             cwd=str(home), argv=["/bin/sleep", "60"])
        queued = client.call("submit", request="queued", peak=64*1024**2, ttl=60,
                             cwd=str(home), argv=["/bin/true"])
        drained = client.call("drain")
        assert drained["draining"] and drained["reserved"] == 128*1024**2
        assert next(r for r in drained["jobs"] if r["ticket"] == queued["ticket"])["state"] == "cancelled"
        with pytest.raises(RuntimeError):
            client.call("submit", request="denied", peak=64*1024**2, ttl=60,
                        cwd=str(home), argv=["/bin/true"])
        client.call("cancel", ticket=active["ticket"])
        assert_empty(client, Systemd())
        # Empty resource ledger is NOT a fully drained Hermes lifecycle.
        assert q.get_status(claim["execution_id"])["status"] == "pending"
        sent = []
        assert q.drain(lambda job, text, failure: sent.append(job["execution_id"])) == 1
        assert sent == [claim["execution_id"]]
        assert q.claim_next() is None
        assert q.get_status(claim["execution_id"])["status"] == "delivered"
    finally:
        fixture.close()
    with pytest.raises(WorkloadUnavailable):
        registry.select_provider({"mode": "required", "provider": "hermes-worker-queue"})
    replacement = home / "plugins/replacement-fixture"
    replacement.mkdir()
    (replacement / "plugin.yaml").write_text("name: replacement-fixture\nversion: 1.0.0\ndescription: fixture\n")
    (replacement / "__init__.py").write_text(
        "from agent.workload_provider import WorkloadProvider\n"
        "class Replacement(WorkloadProvider):\n name='replacement-fixture'\n"
        " def submit(self,spec): return {'state':'queued','ticket':spec.request_id}\n"
        "def register(ctx): ctx.register_workload_provider(Replacement())\n")
    (home / "config.yaml").write_text("plugins:\n  enabled: [replacement-fixture]\n")
    manager = PluginManager()
    manager.discover_and_load()
    try:
        reply = registry.submit({"mode": "required", "provider": "replacement-fixture"},
            WorkloadSpec("replacement", "terminal", "owner", "domain", 1024))
        assert reply["ticket"] == "replacement"
        assert q.claim_next() is None
    finally:
        manager.unload()
