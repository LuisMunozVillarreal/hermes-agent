"""Kernel-bounded workload vs real gateway lifecycle liveness witnesses."""
import asyncio
import contextlib
import json
from pathlib import Path
import socket
import sys
import time

import pytest

from tests.agent.test_workload_execution import managed_home as _fixture_managed_home
managed_home = _fixture_managed_home
from tests.agent.test_isolated_queue_integration import wait_for as _fixture_wait_for
wait_for = _fixture_wait_for

pytestmark = pytest.mark.platforms("linux")


def test_gateway_heartbeat_and_tick_socket_under_denied_growth(managed_home, monkeypatch):
    from gateway import run_startup
    from gateway.shutdown_watchdog import get_loop_tick_socket_path, get_loop_heartbeat_path
    from hermes_worker_queue.client import Client
    from hermes_worker_queue.enforcement.linux import Systemd

    home, state = managed_home
    client, backend = Client(state), Systemd()
    heartbeat_home = state / "profile"
    heartbeat_home.symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(heartbeat_home))
    # Use the real lifecycle component, not a substitute sleep/tick coroutine.
    gateway = run_startup.GatewayStartupMixin()
    monkeypatch.setattr(run_startup, "DEFAULT_HEARTBEAT_INTERVAL_S", 1.0)
    gate = home / "allocate"
    latencies, heartbeats = [], []

    def exercise():
        wait_for(get_loop_tick_socket_path(heartbeat_home).exists)
        blocker = client.call("submit", request="held", peak=96*1024**2, ttl=60,
                              cwd=str(home), argv=["/bin/sleep", "60"])
        pressure = None
        try:
            pressure = client.call("submit", request="underestimated", peak=32*1024**2,
                ttl=60, cwd=str(home), argv=[sys.executable, "-c",
                    "from pathlib import Path; import time\n"
                    f"while not Path({str(gate)!r}).exists(): time.sleep(.01)\n"
                    "data=bytearray(26*1024*1024); time.sleep(60)"])
            cap = backend.path(pressure["unit"]) / "memory.max"
            assert int(cap.read_text()) == 32*1024**2
            gate.touch()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                before = time.monotonic()
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.settimeout(2.0)
                    probe.connect(str(get_loop_tick_socket_path(heartbeat_home)))
                    assert probe.recv(1) == b"1"
                latencies.append(time.monotonic() - before)
                heartbeat = json.loads(get_loop_heartbeat_path(heartbeat_home).read_text())
                assert heartbeat["loop_tick_socket"] is True
                heartbeats.append(heartbeat["monotonic"])
                assert client.call("status")["reserved"] <= 128*1024**2
                time.sleep(.05)  # health: allow HX013 -- sample real heartbeat cadence over the bounded three-second pressure window
            row = wait_for(lambda: next((r for r in client.call("status")["jobs"]
                if r["ticket"] == pressure["ticket"] and r["state"] == "failed"), None))
            assert row["reason"] == "automatic growth denied", row
            assert backend.empty(row["unit"])
            assert len(latencies) >= 10 and max(latencies) < 2.0
            assert len(set(heartbeats)) >= 2
            assert max(b-a for a,b in zip(sorted(set(heartbeats)), sorted(set(heartbeats))[1:])) < 2.0
            print("GATEWAY_RECEIPT " + json.dumps({"probes": len(latencies),
                "max_tick_latency_seconds": max(latencies), "limit_seconds": 2.0,
                "heartbeat_writes": len(set(heartbeats)), "pressure_peak_bytes": 32*1024**2,
                "pressure_outcome": row["reason"]}))
        finally:
            for row in (pressure, blocker):
                if row:
                    client.call("cancel", ticket=row["ticket"])
            wait_for(lambda: client.call("status")["reserved"] == 0)
            assert all(backend.empty(r["unit"]) for r in client.call("status")["jobs"] if r["unit"])

    async def run():
        gateway._start_loop_heartbeat_task()
        assert gateway._loop_heartbeat_task is not None
        try:
            await asyncio.to_thread(exercise)
        finally:
            gateway._loop_heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await gateway._loop_heartbeat_task
        assert not get_loop_tick_socket_path(heartbeat_home).exists()

    asyncio.run(run())
