"""Native cron/delegation using a discovered, same-process gateway queue owner."""
import asyncio
import importlib.util
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading

import pytest
import hermes_yaml as yaml

from tests.agent.test_isolated_agent import local_model as _fixture_local_model
local_model = _fixture_local_model
from tests.agent import test_native_agent_estimates as _fixture_native
native = _fixture_native
from tests.agent import test_isolated_queue_integration as _fixture_nested
nested = _fixture_nested
from tests.agent.test_isolated_queue_integration import gated_model as _fixture_gated_model
gated_model = _fixture_gated_model
from tests.cron import test_queue_lifecycle_integration as _fixture_cron_lifecycle
cron_lifecycle = _fixture_cron_lifecycle

pytestmark = pytest.mark.platforms("linux")


@pytest.fixture
def owned_home(tmp_path, monkeypatch):
    spec = importlib.util.find_spec("hermes_worker_queue")
    if spec is None:
        pytest.skip("requires paired out-of-tree app")
    from hermes_cli.plugins import get_plugin_manager
    from agent.workload_registry import snapshot_registration
    from gateway.run_startup import GatewayStartupMixin
    from gateway.plugin_lifespan import stop_plugin_lifespans
    from hermes_worker_queue.client import Client
    from hermes_worker_queue.enforcement.linux import Systemd

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    directory = tmp_path / "plugins"
    directory.mkdir()
    (directory / "hermes-worker-queue").symlink_to(Path(spec.origin).parents[2])

    def exercise(callback, *, budget=nested.BUDGET, peak=nested.AGENT_CAP):
        # A short state path keeps AF_UNIX below sun_path even in a long pytest base.
        with tempfile.TemporaryDirectory(prefix="q-", dir=os.environ["TMPDIR"]) as state_dir:
            state = Path(state_dir)
            (tmp_path / "config.yaml").write_text(yaml.safe_dump({
                "model": {"default": "fixture"},
                "compression": {"enabled": False},
                "plugins": {
                    "enabled": ["hermes-worker-queue"],
                    "entries": {"hermes-worker-queue": {"settings": {"gateway": {
                        "enabled": True, "state": str(state), "budget": budget,
                    }}}},
                },
                "workloads": {"mode": "required", "provider": "hermes-worker-queue",
                              "state": str(state), "peak_memory_bytes": peak,
                              "estimates": {"terminal": nested.TOOL_CAP}},
            }))
            manager = get_plugin_manager()
            client, backend = Client(state), Systemd()
            before = set(threading.enumerate())

            class Runner(GatewayStartupMixin):
                _running = False

                async def _start_impl(self):
                    assert (await asyncio.to_thread(client.call, "status"))["reserved"] == 0
                    self._running = True
                    return True

                @staticmethod
                async def _start_flush_runtime_status():
                    pass

            async def run():
                runner = Runner()
                rows = []
                try:
                    await runner.start()
                    with socket.socket(socket.AF_UNIX) as peer:
                        peer.connect(str(state / "queue.sock"))
                        pid, _, _ = struct.unpack("3i", peer.getsockopt(
                            socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                        assert pid == os.getpid()
                    provider = snapshot_registration("hermes-worker-queue")
                    assert provider is not None
                    await asyncio.to_thread(callback, (tmp_path, state, client, provider, backend))
                    rows = (await asyncio.to_thread(client.call, "status"))["jobs"]
                finally:
                    await stop_plugin_lifespans(runner)
                    manager.unload()
                assert not (state / "queue.sock").exists()
                assert all(backend.empty(row["unit"]) for row in rows if row["unit"])
            asyncio.run(run())
            assert not [t for t in threading.enumerate()
                        if t not in before and t.name == "hermes-worker-queue"]
    return exercise


@pytest.mark.parametrize("kind", ["delegation", "cron-agent"])
def test_gateway_owner_serves_native_agent_frontdoors(owned_home, local_model, caplog, monkeypatch, kind):
    def work(home):
        if kind == "delegation":
            native.test_live_single_goal_dispatch_estimate_reaches_queue(home, local_model, caplog, monkeypatch)
        else:
            native.test_cron_agent_estimate_survives_watchdog_thread(home, local_model)
    owned_home(work)


@pytest.mark.parametrize("when", ["queued", "running"])
def test_gateway_owner_serves_cron_claim_cancellation(owned_home, monkeypatch, when):
    def work(home):
        cron_lifecycle.test_cron_transport_cancel_settles_before_capacity_release(
            home[:2], monkeypatch, when, "cancel")
    owned_home(work, budget=128 * nested.MIB, peak=nested.TOOL_CAP)


@pytest.mark.parametrize("outcome", ["complete", "cancel"])
def test_gateway_owner_contains_nested_agent_tools(owned_home, gated_model, outcome):
    owned_home(lambda home: nested.test_real_agent_nested_terminal_growth_and_cleanup(
        home, gated_model, "delegation", outcome))
