"""Factory for transcription clients — returns cloud or local based on config."""
from __future__ import annotations

from src.config import Config


def get_transcription_client():
    """Return a transcription client based on the current config.

    Returns WhisperClient (cloud) or LocalWhisperClient (on-device).
    Reads config fresh on every call so settings changes take effect immediately.
    """
    config = Config()
    provider = config.transcription_provider

    if provider == "local":
        from src.transcription.local_whisper_client import LocalWhisperClient
        return LocalWhisperClient()
    else:
        from src.transcription.whisper_client import WhisperClient
        return WhisperClient()
