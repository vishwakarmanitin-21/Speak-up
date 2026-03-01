from __future__ import annotations

import logging
import time
from typing import Callable

from src.audio.recorder import AudioRecorder
from src.audio.silence_detector import SilenceDetector
from src.config import Config
from src.context.context_builder import ContextBuilder
from src.context.session_memory import SessionMemory
from src.output.inserter import OutputInserter
from src.rewrite.engine import RewriteEngine
from src.rewrite.modes import RewriteMode
from src.transcription.factory import get_transcription_client

logger = logging.getLogger("flowai")


class PipelineState:
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    REWRITING = "rewriting"
    DONE = "done"
    ERROR = "error"


class Pipeline:
    """Orchestrates the voice -> text -> rewrite -> output pipeline."""

    def __init__(self) -> None:
        config = Config()
        self._recorder = AudioRecorder(sample_rate=config.sample_rate)
        self._silence_detector = SilenceDetector(
            silence_duration_ms=config.silence_timeout_ms,
        )
        self._rewriter = RewriteEngine()
        self._session_memory = SessionMemory()
        self._context_builder = ContextBuilder(self._session_memory)
        self._inserter = OutputInserter()
        self._state = PipelineState.IDLE
        self._on_state_change: Callable[[str], None] | None = None
        self._on_silence_detected: Callable[[], None] | None = None
        self._cancelled = False

        # Wire silence detection into audio callback
        self._config = config
        original_cb = self._recorder._audio_callback

        def _cb_with_silence(indata, frames, time_info, status):
            original_cb(indata, frames, time_info, status)
            if (
                self._config.auto_stop_on_silence
                and self._recorder.is_recording
                and self._silence_detector.feed(indata)
            ):
                self._recorder.stop()
                if self._on_silence_detected:
                    self._on_silence_detected()

        self._recorder._audio_callback = _cb_with_silence

    def set_state_callback(self, callback: Callable[[str], None]) -> None:
        self._on_state_change = callback

    def set_silence_callback(self, callback: Callable[[], None]) -> None:
        self._on_silence_detected = callback

    def _set_state(self, state: str) -> None:
        self._state = state
        if self._on_state_change:
            self._on_state_change(state)

    @property
    def state(self) -> str:
        return self._state

    def start_recording(self) -> None:
        self._cancelled = False
        self._silence_detector.reset()
        self._set_state(PipelineState.RECORDING)
        self._recorder.start()

    def stop_recording(self) -> None:
        self._recorder.stop()

    def cancel(self) -> None:
        """Cancel the current pipeline run and stop recording if active."""
        self._cancelled = True
        if self._recorder.is_recording:
            self._recorder.stop()
        self._set_state(PipelineState.IDLE)

    async def process(
        self,
        mode: RewriteMode,
        output_mode: str | None = None,
    ) -> tuple[str, str]:
        """Transcribe recorded audio, rewrite, and deliver output.

        Args:
            mode: The rewrite mode to apply.
            output_mode: Override output mode (uses config default if None).

        Returns:
            Tuple of (raw_transcription, rewritten_text).
        """
        self._cancelled = False
        _start = time.monotonic()
        config = Config()
        logger.info(
            "Pipeline run: mode=%s, model=%s, temp=%s, provider=%s",
            mode.value, config.gpt_model, config.temperature,
            config.transcription_provider,
        )
        try:
            # 1. Build context from clipboard, selection, session memory, VS Code
            context = self._context_builder.build()
            logger.info(
                "Context: %d chars" if context else "Context: none",
                len(context) if context else 0,
            )
            if self._cancelled:
                self._set_state(PipelineState.IDLE)
                return "", ""

            # 2. Transcribe — pick client from config (cloud or local)
            self._set_state(PipelineState.TRANSCRIBING)
            wav_bytes = self._recorder.get_wav_bytes()

            # Build a Whisper prompt from recent session history for better accuracy
            whisper_prompt = self._session_memory.get_whisper_prompt()

            whisper = get_transcription_client()
            raw_text = await whisper.transcribe(wav_bytes, prompt=whisper_prompt)
            logger.info(
                "Transcription: %d words (%d chars)",
                len(raw_text.split()), len(raw_text),
            )
            if self._cancelled:
                self._set_state(PipelineState.IDLE)
                return "", ""

            # 3. Rewrite
            self._set_state(PipelineState.REWRITING)
            rewritten = await self._rewriter.rewrite(raw_text, mode, context)
            logger.info(
                "Rewrite: %d words (%d chars)",
                len(rewritten.split()), len(rewritten),
            )
            if self._cancelled:
                self._set_state(PipelineState.IDLE)
                return "", ""

            # 4. Store in session memory
            self._session_memory.add(raw_text, rewritten, mode.value)

            # 5. Deliver output
            self._inserter.deliver(rewritten, output_mode)

            # 6. Record usage analytics (non-blocking, best-effort)
            if config.track_usage:
                try:
                    from src.services.usage_tracker import record_run
                    record_run(
                        raw_text=raw_text,
                        rewritten_text=rewritten,
                        mode=mode.value,
                        provider=config.transcription_provider,
                        duration_ms=int((time.monotonic() - _start) * 1000),
                    )
                except Exception:
                    pass

            elapsed = time.monotonic() - _start
            logger.info("Pipeline complete in %.1fs", elapsed)
            self._set_state(PipelineState.DONE)
            return raw_text, rewritten

        except Exception as exc:
            logger.error("Pipeline error: %s", exc)
            self._set_state(PipelineState.ERROR)
            raise
