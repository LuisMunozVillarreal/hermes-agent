"""Native Discord typing transport and lifecycle."""

import asyncio
import logging

import discord

logger = logging.getLogger(__name__)


class DiscordTypingMixin:
    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Start a persistent typing loop (POST typing every 5s; indicator lasts ~10s).
        TYPING_START is unreliable for bots in DMs; 429 sleeps ``retry_after``; CancelledError ends it."""
        if not self._client:
            return
        if chat_id in self._typing_tasks:
            return

        async def _typing_loop() -> None:
            try:
                while True:
                    try:
                        route = discord.http.Route(
                            "POST", "/channels/{channel_id}/typing", channel_id=chat_id,
                        )
                        await self._client.http.request(route)
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        retry_after = self._extract_discord_retry_after(e)
                        if retry_after is not None:
                            logger.warning(
                                "Typing indicator rate-limited for %s; retrying in %.1fs",
                                chat_id, retry_after,
                            )
                        else:
                            logger.debug("Discord typing indicator failed for %s: %s", chat_id, e)
                            return
                        await asyncio.sleep(retry_after)
                        continue
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass
            finally:
                self._typing_tasks.pop(chat_id, None)
        self._typing_tasks[chat_id] = asyncio.create_task(_typing_loop())

    async def stop_typing(self, chat_id: str) -> None:
        """Stop the persistent typing indicator for a channel."""
        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

