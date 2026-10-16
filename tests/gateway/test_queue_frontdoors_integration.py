"""Three native frontdoors contend concurrently through one real coordinator."""
import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

from tests.agent.test_workload_execution import managed_home as _fixture_managed_home
managed_home = _fixture_managed_home
from tests.agent.test_isolated_queue_integration import gated_model as _fixture_gated_model, wait_for as _fixture_wait_for
gated_model = _fixture_gated_model
wait_for = _fixture_wait_for

pytestmark = pytest.mark.platforms("linux")


def test_gateway_cli_cron_share_capacity(managed_home, gated_model):
    from hermes_worker_queue.client import Client
    from hermes_worker_queue.enforcement.linux import Systemd

    home, state = managed_home
    url, settings, _entered, model_release, _requests = gated_model
    client, backend = Client(state), Systemd()
    settings["command"] = (
        f"printf x >> {shlex.quote(str(home / 'terminals-ran'))}; "
        f"while [ ! -f {shlex.quote(str(home / 'release'))} ]; do sleep .05; done; "
        "printf terminal-result")
    model_release.set()
    config = home / "config.yaml"
    config.write_text(config.read_text() +
        f"model:\n  default: fixture\n  provider: custom\n  base_url: {url}\n  api_key: local-fixture\n"
        "compression:\n  enabled: false\nmemory:\n  memory_enabled: false\n"
        "  user_profile_enabled: false\nsecurity:\n  tirith_enabled: false\n"
        "platform_toolsets:\n  telegram: [terminal]\n")
    blocker = client.call("submit", request="block", peak=128*1024**2, ttl=90,
                          cwd=str(home), argv=["/bin/sleep", "90"])
    clients = {}
    samples = []
    try:
        for door in ("gateway", "cli", "cron"):
            log = (home / (door + ".log")).open("w")
            clients[door] = (subprocess.Popen([sys.executable, "-m",
                "tests.gateway.queue_frontdoor_fixture", door, url],
                stdout=log, stderr=subprocess.STDOUT), log)
        def queued():
            for door, (proc, log) in clients.items():
                if proc.poll() is not None:
                    pytest.fail(f"{door} exited {proc.returncode}: " + (home / (door+'.log')).read_text())
            rows = client.call("status")["jobs"]
            owners = {int(r["owner"].split(":")[1]) for r in rows if r["state"] == "queued"}
            return rows if owners == {p.pid for p, _ in clients.values()} else None
        rows = wait_for(queued, timeout=30)
        assert not (home / "terminals-ran").exists()
        assert not (home / "cron-ran").exists()
        client.call("cancel", ticket=blocker["ticket"])
        blocker = None
        # Every frontdoor's first shell may be its initialization probe. Keep
        # sampling through those starts until the actual held commands contend.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            status = client.call("status")
            samples.append(status["reserved"])
            assert status["reserved"] <= 128*1024**2
            held = (len((home / "terminals-ran").read_text()) if (home / "terminals-ran").exists() else 0)
            held += int((home / "cron-ran").exists())
            if held == 2 and any(r["state"] == "queued" for r in status["jobs"]):
                break
            time.sleep(.05)
        else:
            pytest.fail("two held frontdoors never forced the third to queue: " + json.dumps(status))
        (home / "release").touch()
        for door, (proc, log) in clients.items():
            assert proc.wait(timeout=30) == 0, (home / (door+'.log')).read_text()
        assert (home / "terminals-ran").read_text() == "xx"
        assert (home / "cron-ran").exists()
        assert "queue-contained-result" in (home / "gateway-reply").read_text()
        wait_for(lambda: client.call("status")["reserved"] == 0)
        rows = client.call("status")["jobs"]
        assert {r["kind"] for r in rows} >= {"terminal", "cron-script"}
        assert all(backend.empty(r["unit"]) for r in rows if r["unit"])
    finally:
        (home / "release").touch()
        if blocker:
            client.call("cancel", ticket=blocker["ticket"])
        for proc, log in clients.values():
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
            log.close()
