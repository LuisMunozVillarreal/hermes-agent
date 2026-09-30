"""Native Discord typing must cover the whole conversation's live work."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
async def test_native_typing_refreshes_before_discord_expiry(monkeypatch):
    import plugins.platforms.discord.adapter as module

    delays = asyncio.Queue()
    release = asyncio.Event()

    async def sleep(delay):
        await delays.put(delay)
        await release.wait()

    monkeypatch.setattr(module, "asyncio", SimpleNamespace(
        **{name: getattr(asyncio, name) for name in dir(asyncio) if not name.startswith("_")},
    ))
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=AsyncMock()))
    try:
        await adapter.send_typing("100")
        delay = await asyncio.wait_for(delays.get(), 5)
        assert 0 < delay < 10, "Discord expires typing after about ten seconds"
        await adapter.send_typing("100")
        assert adapter._client.http.request.await_count == 1
    finally:
        await adapter.stop_typing("100")
