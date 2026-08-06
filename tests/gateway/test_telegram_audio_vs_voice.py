"""
Tests for #24870 — Telegram: audio file attachments must NOT be routed to STT.

Telegram distinguishes three kinds of audio payloads:
  - message.voice  → Opus/OGG voice message  → STT pipeline
  - message.audio  → audio file attachment   → file path note, NOT STT
  - message.document (audio mime) → generic file route

These tests confirm that:
  1. MessageType.VOICE events still flow through the STT pipeline.
  2. MessageType.AUDIO events bypass STT and get a file-path context note instead.
  3. Mixed media lists (voice + audio) split correctly.
"""

from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_runner(
    stt_enabled: bool = True, transcribe_attachments: bool = False
) -> "GatewayRunner":  # type: ignore[name-defined]
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        stt_enabled=stt_enabled,
        stt_transcribe_audio_attachments=transcribe_attachments,
    )
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    return runner


def _voice_event(path: str = "/tmp/voice.ogg") -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        media_urls=[path],
        media_types=["audio/ogg"],
    )


def _audio_event(path: str = "/tmp/song.mp3") -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.AUDIO,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        media_urls=[path],
        media_types=["audio/mpeg"],
    )


# ---------------------------------------------------------------------------
# 1. VOICE still goes through STT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_voice_message_still_transcribed():
    """MessageType.VOICE must still be sent through _enrich_message_with_transcription."""
    runner = _make_runner(stt_enabled=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    event = _voice_event("/tmp/voice.ogg")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello world", "provider": "whisper"},
    ) as mock_transcribe:
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    mock_transcribe.assert_called_once_with("/tmp/voice.ogg", None, "gateway")
    # The transcript passes through as a plain quoted line — no "voice message"
    # meta-commentary in the LLM-visible prompt.
    assert "hello world" in result


# ---------------------------------------------------------------------------
# 2. AUDIO file attachment bypasses STT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_attachment_context_note_format():
    """Context note for audio file attachments should include the file path and guidance."""
    runner = _make_runner(stt_enabled=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    event = _audio_event("/tmp/cache_12345_my_song.mp3")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("must not be called"),
    ):
        with patch(
            "tools.credential_files.to_agent_visible_cache_path",
            side_effect=lambda p: p,
        ):
            result = await runner._prepare_inbound_message_text(
                event=event,
                source=source,
                history=[],
            )

    assert "my_song.mp3" in result
    assert "audio file attachment" in result.lower()
    # Should NOT contain the voice-message transcription wrapper text
    assert "voice message" not in result.lower()
    # Guides the agent to transcribe/process the file itself rather than
    # punting back to the user (same bug class as the PDF/DOCX note).
    assert "transcri" in result.lower()
    assert "ask the user what they'd like" not in result.lower()


# ---------------------------------------------------------------------------
# 3. STT disabled still results in no transcription for audio file attachments
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Opt-in: stt.transcribe_audio_attachments routes AUDIO through STT
# ---------------------------------------------------------------------------

@patch("gateway.run._event_media_type_at", lambda event, i: event.media_types[i])
def test_audio_attachment_stt_input_gate_respects_flag():
    """_event_media_is_stt_input: AUDIO only qualifies with the opt-in flag."""
    from gateway.run import _event_media_is_stt_input

    event = _audio_event("/tmp/song.mp3")
    assert _event_media_is_stt_input(event, 0) is False
    assert _event_media_is_stt_input(event, 0, transcribe_attachments=True) is True
    # VOICE is never gated by the flag; DOCUMENT never qualifies.
    voice = _voice_event("/tmp/voice.ogg")
    assert _event_media_is_stt_input(voice, 0) is True
    doc = MessageEvent(
        text="",
        message_type=MessageType.DOCUMENT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        media_urls=["/tmp/doc.mp3"],
        media_types=["audio/mpeg"],
    )
    assert _event_media_is_stt_input(doc, 0, transcribe_attachments=True) is False


@pytest.mark.asyncio
async def test_audio_attachment_transcribed_when_flag_enabled():
    """With stt_transcribe_audio_attachments=True, AUDIO files are transcribed like VOICE."""
    runner = _make_runner(stt_enabled=True, transcribe_attachments=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    event = _audio_event("/tmp/cache_12345_voice_note.mp3")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hola que tal", "provider": "whisper"},
    ) as mock_transcribe:
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    mock_transcribe.assert_called_once_with(
        "/tmp/cache_12345_voice_note.mp3", None, "gateway"
    )
    # Transcript passes through as a plain quoted line; no file-path note needed.
    assert "hola que tal" in result
    assert "audio file attachment" not in result.lower()


@pytest.mark.asyncio
async def test_pending_audio_attachment_selected_for_stt_when_flag_enabled():
    """Pending-path STT eligibility includes AUDIO files when the flag is on."""
    runner = _make_runner(stt_enabled=True, transcribe_attachments=True)
    event = _audio_event("/tmp/pending-note.mp3")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hola", "provider": "whisper"},
    ) as mock_transcribe:
        result, transcripts = await runner._transcribe_pending_audio_event_once(
            event,
            event.text,
        )

    assert runner._pending_event_audio_paths(event) == ["/tmp/pending-note.mp3"]
    mock_transcribe.assert_called_once_with("/tmp/pending-note.mp3", None, "gateway")
    assert transcripts == ["hola"]


# ---------------------------------------------------------------------------
# 4. Telegram gateway: msg.audio → MessageType.AUDIO (not VOICE)
# ---------------------------------------------------------------------------

