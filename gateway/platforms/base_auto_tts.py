"""Post-handler auto-TTS synthesis for ``BasePlatformAdapter`` (voice reply to a voice note).

Runs after the message handler returned, so the routed profile's secret scope that wrapped the
turn is gone. Under a multiplexed gateway an unscoped provider read (``GEMINI_BASE_URL``, a plugin
provider's key in ``is_available``) raises ``UnscopedSecretError`` and the voice reply is lost, so
synthesis re-enters the scope of the profile that serves ``source`` (#145151).
"""

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _profile_scope(adapter, source) -> contextlib.AbstractAsyncContextManager:
    """The runner's secret-hydrating scope for ``source``; a no-op without a runner."""
    resolve = getattr(getattr(adapter, "gateway_runner", None), "_async_profile_scope_for_source", None)
    if not callable(resolve) or source is None:
        return contextlib.nullcontext()
    return resolve(source)


def _synthesize(text: str, output_path: str) -> list[str]:
    """Worker-thread half: run the TTS tool and keep the files it reports that exist."""
    from tools.tts_tool import text_to_speech_tool
    tts_data = json.loads(text_to_speech_tool(text=text, output_path=output_path))
    if not tts_data.get("success", False):
        raise RuntimeError("TTS tool returned success=false")
    raw_tts_paths = tts_data.get("file_paths") or [tts_data.get("file_path")]
    declared_paths = [str(path) for path in raw_tts_paths if path]
    if (len(declared_paths) != len(raw_tts_paths) or not declared_paths
            or any(not Path(path).exists() for path in declared_paths)):
        for path in declared_paths:
            with contextlib.suppress(OSError):
                os.remove(path)
        raise RuntimeError("TTS generation reported success without a complete set of usable audio output files")
    return declared_paths


async def synthesize_auto_tts(adapter, text_content: str, source, output_path_for) -> tuple[list[str], Optional[str], Optional[str]]:
    """``(existing_paths, requested_path)``; empty/None on failure (logged, never raised).
    ``output_path_for(platform)`` builds the platform-aware temp path (``.ogg`` for Opus platforms)."""
    paths: list[str] = []
    requested_path = None
    error_notice = None
    try:
        from tools.tts_tool import check_tts_requirements
        async with _profile_scope(adapter, source):
            if check_tts_requirements():
                speech_text = adapter.prepare_tts_text(text_content)
                if not speech_text:
                    raise ValueError("Empty text after markdown cleanup")
                requested_path = output_path_for(adapter.platform)
                paths = await asyncio.to_thread(_synthesize, speech_text, requested_path)
            else:
                raise RuntimeError("TTS provider requirements are unavailable")
    except Exception as tts_err:  # an auto voice reply never fails the text reply it accompanies
        logger.warning("[%s] Auto-TTS failed: %s", adapter.name, tts_err, exc_info=True)
        error_notice = "Audio could not be fully delivered because TTS failed with the configured provider."
    return paths, requested_path, error_notice
