"""Gateway resources must be ready before startup and unwind on failed boot."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from gateway.run_startup import GatewayStartupMixin


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["adapter", "plugin", None])
async def test_discovered_lifespan_precedes_work_and_unwinds_failed_start(
    tmp_path, monkeypatch, failure
):
    from hermes_cli import plugins

    directory = tmp_path / "plugins" / "lifetest"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text("name: lifetest\nversion: 1.0.0\n")
    (directory / "__init__.py").write_text("""
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifetime(*, gateway):
    if gateway.failure == "plugin":
        raise RuntimeError("plugin startup failed")
    gateway.events.append("ready")
    try:
        yield
    finally:
        gateway.events.append("closed")
def register(ctx):
    ctx.register_hook("gateway_lifespan", lifetime)
""")
    (tmp_path / "config.yaml").write_text("plugins:\n  enabled: [lifetest]\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    class Runner(GatewayStartupMixin):
        _running = False
        events = []

        async def _start_impl(self):
            assert self.events == ["ready"]
            self.events.append("work")
            if self.failure == "adapter":
                raise RuntimeError("adapter startup failed")
            self._running = True
            return True

        @staticmethod
        async def _start_flush_runtime_status():
            pass

    runner = Runner()
    runner.failure = failure
    try:
        if failure:
            with pytest.raises(RuntimeError, match=f"{failure} startup failed"):
                await runner.start()
            assert runner.events == (
                [] if failure == "plugin" else ["ready", "work", "closed"]
            )
        else:
            from gateway.run import GatewayRunner
            from gateway.run_shutdown import GatewayShutdownMixin

            await runner.start()

            async def phase(*args):
                assert runner.events == ["ready", "work"]

            for name in (
                "_stop_begin_teardown",
                "_stop_drain_active_work",
                "_stop_finalize_agents_and_adapters",
                "_stop_release_runtime_state",
            ):
                monkeypatch.setattr(GatewayRunner, name, phase)

            async def persist(*args):
                assert runner.events == ["ready", "work", "closed"]

            monkeypatch.setattr(GatewayRunner, "_stop_persist_exit_state", persist)
            monkeypatch.setattr(
                GatewayRunner, "_stop_quiesce_and_close_session_dbs", lambda *a: None
            )
            runner._restart_drain_timeout = 5
            await GatewayShutdownMixin._stop_impl(runner)
            assert runner.events == ["ready", "work", "closed"]
    finally:
        manager.unload()


@pytest.mark.asyncio
async def test_stop_during_resource_entry_waits_for_cleanup(monkeypatch):
    from gateway.plugin_lifespan import start_plugin_lifespans, stop_plugin_lifespans
    from hermes_cli import plugins

    entered, cleaned = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def resource(*, gateway):
        try:
            entered.set()
            await asyncio.Event().wait()
            yield
        finally:
            cleaned.set()

    manager = plugins.PluginManager()
    manager._discovered = True
    manager._hooks["gateway_lifespan"] = [resource]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    runner = SimpleNamespace()
    start = asyncio.create_task(start_plugin_lifespans(runner))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await stop_plugin_lifespans(runner)
        assert cleaned.is_set(), "stop returned while resource entry still owned work"
        with pytest.raises(asyncio.CancelledError):
            await start
    finally:
        start.cancel()
        await asyncio.gather(start, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_concurrent_stop_retains_ownership_until_reverse_cleanup(monkeypatch, cancel):
    from gateway.plugin_lifespan import start_plugin_lifespans, stop_plugin_lifespans
    from hermes_cli import plugins

    closing, release = asyncio.Event(), asyncio.Event()
    events = []

    @asynccontextmanager
    async def first(*, gateway):
        try:
            yield
        finally:
            events.append("first")

    @asynccontextmanager
    async def second(*, gateway):
        try:
            yield
        finally:
            closing.set()
            await release.wait()
            events.append("second")

    manager = plugins.PluginManager()
    manager._discovered = True
    manager._hooks["gateway_lifespan"] = [first, second]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    runner = SimpleNamespace()
    await start_plugin_lifespans(runner)
    stop = asyncio.create_task(stop_plugin_lifespans(runner))
    sibling = None
    try:
        await asyncio.wait_for(closing.wait(), 5)
        if cancel:
            stop.cancel()
            await asyncio.sleep(0)
            stop.cancel()
        sibling = asyncio.create_task(stop_plugin_lifespans(runner))
        done, _ = await asyncio.wait({sibling}, timeout=0.05)
        assert not done, "second stop returned while required cleanup was still running"
        with pytest.raises(RuntimeError, match="already started"):
            await start_plugin_lifespans(runner)
    finally:
        release.set()
        results = await asyncio.gather(stop, *([sibling] if sibling else []), return_exceptions=True)
    assert events == ["second", "first"]
    if cancel:
        assert isinstance(results[0], asyncio.CancelledError)
    await start_plugin_lifespans(runner)
    await stop_plugin_lifespans(runner)


@pytest.mark.asyncio
async def test_failed_start_unwind_survives_repeated_cancellation(monkeypatch):
    from gateway.plugin_lifespan import start_plugin_lifespans
    from hermes_cli import plugins

    closing, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def acquired(*, gateway):
        try:
            yield
        finally:
            closing.set()
            await release.wait()
            closed.set()

    @asynccontextmanager
    async def failing(*, gateway):
        raise ValueError("entry failed")
        yield

    manager = plugins.PluginManager()
    manager._discovered = True
    manager._hooks["gateway_lifespan"] = [acquired, failing]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    runner = SimpleNamespace()
    start = asyncio.create_task(start_plugin_lifespans(runner))
    try:
        await asyncio.wait_for(closing.wait(), 5)
        start.cancel()
        await asyncio.sleep(0)
        start.cancel()
        done, _ = await asyncio.wait({start}, timeout=0.05)
        assert not done, "startup cancellation abandoned resource unwind"
    finally:
        release.set()
        await asyncio.gather(start, return_exceptions=True)
    assert closed.is_set()


def test_shutdown_watchdog_bounds_stuck_plugin_exit(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent('''
        import asyncio
        from contextlib import asynccontextmanager
        from pathlib import Path
        from types import SimpleNamespace
        from gateway import run_shutdown
        from gateway.run import GatewayRunner
        from gateway.plugin_lifespan import start_plugin_lifespans
        from hermes_cli import plugins

        @asynccontextmanager
        async def resource(*, gateway):
            try:
                yield
            finally:
                await asyncio.Event().wait()

        manager = plugins.PluginManager()
        manager._discovered = True
        manager._hooks["gateway_lifespan"] = [resource]
        plugins.get_plugin_manager = lambda: manager
        async def phase(*args):
            pass
        for name in ("_stop_begin_teardown", "_stop_drain_active_work",
                     "_stop_finalize_agents_and_adapters", "_stop_release_runtime_state"):
            setattr(GatewayRunner, name, phase)
        GatewayRunner._stop_quiesce_and_close_session_dbs = lambda *a: None
        GatewayRunner._shutdown_watchdog_snapshot = lambda *a: {"phase": "plugin-exit"}
        async def persist(*args):
            Path("clean-exit").touch()
        GatewayRunner._stop_persist_exit_state = persist
        run_shutdown._effective_watchdog_leash = lambda *a: 0.2
        async def main():
            runner = SimpleNamespace(_restart_drain_timeout=1)
            await start_plugin_lifespans(runner)
            await run_shutdown.GatewayShutdownMixin._stop_impl(runner)
        asyncio.run(main())
    ''')
    env = dict(os.environ, HERMES_HOME=str(tmp_path))
    env.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 1, result.stderr
    assert not (tmp_path / "clean-exit").exists()
    dump = tmp_path / "logs" / "gateway-shutdown-watchdog.log"
    assert "shutdown_watchdog_fired" in dump.read_text()

@pytest.mark.asyncio
async def test_stop_during_entry_refuses_callback_restart_until_stop_returns(monkeypatch):
    from gateway.plugin_lifespan import start_plugin_lifespans, stop_plugin_lifespans
    from hermes_cli import plugins

    entered = asyncio.Event()
    calls, closed = [], []

    @asynccontextmanager
    async def resource(*, gateway):
        calls.append(len(calls) + 1)
        number = calls[-1]
        if number == 1:
            entered.set()
            await asyncio.Event().wait()
        try:
            yield
        finally:
            closed.append(number)

    manager = plugins.PluginManager()
    manager._discovered = True
    manager._hooks["gateway_lifespan"] = [resource]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    runner = SimpleNamespace()
    replacement = []
    starting = asyncio.create_task(start_plugin_lifespans(runner))
    starting.add_done_callback(lambda _: replacement.append(
        asyncio.create_task(start_plugin_lifespans(runner))))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await stop_plugin_lifespans(runner)
        results = await asyncio.gather(*replacement, return_exceptions=True)
        assert len(results) == 1 and isinstance(results[0], RuntimeError), (
            "replacement reported success while the old stop silently closed it", results, closed)
        await start_plugin_lifespans(runner)
        assert closed == []
    finally:
        await stop_plugin_lifespans(runner)
    assert closed == [2]
