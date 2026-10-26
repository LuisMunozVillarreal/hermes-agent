"""Required plugin resources owned by one gateway runner, not observer hooks."""

import asyncio
from contextlib import AsyncExitStack


async def start_plugin_lifespans(gateway):
    from hermes_cli.plugins import _ensure_plugins_discovered

    if (
        getattr(gateway, "_plugin_lifespan_stack", None) is not None
        or getattr(gateway, "_plugin_lifespan_start_task", None) is not None
        or getattr(gateway, "_plugin_lifespan_stops", 0)
    ):
        raise RuntimeError("gateway plugin lifespans already started")
    manager = _ensure_plugins_discovered()
    stack = AsyncExitStack()
    gateway._plugin_lifespan_start_task = asyncio.current_task()
    try:
        for factory in manager.iter_hook_callbacks("gateway_lifespan"):
            await stack.enter_async_context(factory(gateway=gateway))
    except BaseException:
        gateway._plugin_lifespan_stack = stack
        await stop_plugin_lifespans(gateway)
        raise
    finally:
        gateway._plugin_lifespan_start_task = None
    gateway._plugin_lifespan_stack = stack


async def stop_plugin_lifespans(gateway):
    # Keep admission closed until every stop waiter has finished, including the
    # scheduling gap after a cancelled startup task completes.
    gateway._plugin_lifespan_stops = getattr(gateway, "_plugin_lifespan_stops", 0) + 1
    try:
        await _stop_plugin_lifespans(gateway)
    finally:
        gateway._plugin_lifespan_stops -= 1


async def _stop_plugin_lifespans(gateway):
    starting = getattr(gateway, "_plugin_lifespan_start_task", None)
    if starting is not None and starting is not asyncio.current_task():
        starting.cancel()
        await asyncio.gather(starting, return_exceptions=True)
    stack = getattr(gateway, "_plugin_lifespan_stack", None)
    if stack is not None:
        closing = getattr(gateway, "_plugin_lifespan_close_task", None)
        if closing is None:
            closing = asyncio.create_task(stack.aclose())
            gateway._plugin_lifespan_close_task = closing
        cancelled = False
        try:
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError:
                    cancelled = True
            closing.result()
        finally:
            if gateway._plugin_lifespan_stack is stack:
                gateway._plugin_lifespan_stack = None
                gateway._plugin_lifespan_close_task = None
        if cancelled:
            raise asyncio.CancelledError
