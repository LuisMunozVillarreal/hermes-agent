"""Gateway teardown with required resources under the existing watchdog."""

async def stop_with_resources(self) -> None:
    """Run every ``_stop_*`` phase under the thread-based shutdown watchdog."""
    import os
    import threading
    from gateway.run_shutdown import (GatewayShutdownMixin, arm_shutdown_watchdog,
                                      _effective_watchdog_leash, effective_stop_drain_timeout, logger)
    from gateway.run import GatewayRunner
    # Thread-based watchdog (asyncio timeouts cannot recover a frozen loop): dumps stacks and
    # os._exit past drain+grace so the service manager revives us. Skipped under pytest.
    # Arm a plain OS thread at the start of stop(); if teardown never finishes within drain+grace it
    # dumps faulthandler stacks and os._exit so KeepAlive/systemd can revive. Skip under pytest so
    # stop()-driving unit tests don't get a delayed hard-exit in the worker. See #66892.
    _watchdog_done = threading.Event()
    self._shutdown_watchdog_done = _watchdog_done
    # Shutdown-path doubles may lack the deferred-worker counter.
    ctx = GatewayShutdownMixin._StopContext(
        deferred_count=getattr(self, "_active_deferred_agent_worker_count", lambda: 0)
    )
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        arm_shutdown_watchdog(
            _effective_watchdog_leash(self), done_event=_watchdog_done,
            snapshot_fn=lambda: GatewayRunner._shutdown_watchdog_snapshot(self, ctx), exit_code=1,
        )
    try:
        await GatewayRunner._stop_begin_teardown(self, ctx)
        timeout = effective_stop_drain_timeout(self)
        if timeout < self._restart_drain_timeout:
            logger.warning(
                "Shutdown drain capped to %.0fs (configured %.0fs) to fit the live launchd exit "
                "timeout of %.0fs — launchd SIGKILLs past it",
                timeout, self._restart_drain_timeout, self._launchd_exit_timeout_s,
            )
        await GatewayRunner._stop_drain_active_work(self, timeout, ctx)
        if ctx.timed_out:
            await GatewayRunner._stop_interrupt_remaining_work(self, ctx)
        await GatewayRunner._stop_finalize_agents_and_adapters(self, ctx)
        await GatewayRunner._stop_release_runtime_state(self, ctx)
        GatewayRunner._stop_quiesce_and_close_session_dbs(self, timeout, ctx)
        from gateway.plugin_lifespan import stop_plugin_lifespans
        await stop_plugin_lifespans(self)
        await GatewayRunner._stop_persist_exit_state(self, ctx)
    finally:
        from gateway.plugin_lifespan import stop_plugin_lifespans
        try:
            await stop_plugin_lifespans(self)
        finally:
            _watchdog_done.set()
