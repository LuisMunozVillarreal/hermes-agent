"""Composition keeps idle-stop cancellation for workers and detached audio."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.session import build_session_key
from tests.gateway.test_background_auto_tts import adapter_and_event


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted_workers", [0, 1])
async def test_idle_stop_cancels_audio_after_worker_stop(monkeypatch, interrupted_workers):
    from tools import async_delegation

    adapter, event = adapter_and_event()
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._background_tasks = set()
    runner.adapters = {Platform.DISCORD: adapter}
    key = build_session_key(event.source)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_key=key, session_id="test"))
    adapter.gateway_runner = runner
    adapter.set_message_handler(AsyncMock(return_value="Complete reply."))
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text-1"))
    adapter.send_voice = AsyncMock()
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def blocked_synthesis(text, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    adapter._synthesize_auto_tts = blocked_synthesis
    interrupt_workers = Mock(return_value=interrupted_workers)
    monkeypatch.setattr(async_delegation, "interrupt_for_session", interrupt_workers)
    try:
        await adapter.handle_message(event)
        await asyncio.wait_for(started.wait(), 5)
        owner = adapter._session_tasks.get(key)
        if owner:
            await asyncio.wait_for(asyncio.shield(owner), 2)
        assert not adapter._active_sessions
        assert await runner._handle_stop_command(event) == t("gateway.stop.stopped")
        interrupt_workers.assert_called_once_with(
            session_key=key, reason="stop_command", parent_session_id="test")
        assert cancelled.is_set()
        assert not adapter._background_tasks
        adapter.send_voice.assert_not_called()
    finally:
        await adapter.cancel_background_tasks()
