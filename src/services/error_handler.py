from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger("flowai")


class FlowAIError(Exception):
    """Base exception for FlowAI."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or message


class APIKeyError(FlowAIError):
    """Raised when API key is missing or invalid."""
    pass


class RecordingError(FlowAIError):
    """Raised when audio recording fails."""
    pass


class TranscriptionError(FlowAIError):
    """Raised when Whisper API fails."""
    pass


class RewriteError(FlowAIError):
    """Raised when GPT API fails."""
    pass


def setup_logging() -> None:
    """Configure application logging."""
    log_dir = Path(__file__).resolve().parent.parent.parent
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                log_dir / "flowai.log", encoding="utf-8"
            ),
        ],
    )
