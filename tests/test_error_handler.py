"""Tests for custom exception hierarchy."""
from __future__ import annotations

from src.services.error_handler import (
    FlowAIError,
    APIKeyError,
    RecordingError,
    TranscriptionError,
    RewriteError,
)


def test_flowai_error_is_exception():
    err = FlowAIError("base error")
    assert isinstance(err, Exception)
    assert str(err) == "base error"
    assert err.user_message == "base error"


def test_user_message_override():
    err = FlowAIError("internal msg", "User-friendly message")
    assert err.user_message == "User-friendly message"
    assert str(err) == "internal msg"


def test_subclasses_are_flowai_errors():
    for cls in (APIKeyError, RecordingError, TranscriptionError, RewriteError):
        err = cls("test")
        assert isinstance(err, FlowAIError)
        assert isinstance(err, Exception)


def test_api_key_error_user_message():
    err = APIKeyError("OPENAI_API_KEY not set", "Please add your key in Settings.")
    assert err.user_message == "Please add your key in Settings."


def test_recording_error_user_message():
    err = RecordingError("PortAudioError", "Microphone not found.")
    assert err.user_message == "Microphone not found."
