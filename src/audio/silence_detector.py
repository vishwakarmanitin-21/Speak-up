from __future__ import annotations

import time

import numpy as np


class SilenceDetector:
    """Detects sustained silence in audio frames.

    Monitors the RMS energy of audio frames. When the energy stays
    below the threshold for longer than silence_duration_ms, the
    detector reports silence.
    """

    def __init__(
        self,
        threshold_rms: float = 0.01,
        silence_duration_ms: int = 2000,
    ) -> None:
        self._threshold = threshold_rms
        self._silence_duration_s = silence_duration_ms / 1000.0
        self._silence_start: float | None = None

    def feed(self, frame: np.ndarray) -> bool:
        """Feed an audio frame and check for silence.

        Args:
            frame: Audio data as a numpy array (float32, -1 to 1).

        Returns:
            True if silence has been sustained long enough to trigger auto-stop.
        """
        rms = float(np.sqrt(np.mean(frame ** 2)))

        if rms < self._threshold:
            if self._silence_start is None:
                self._silence_start = time.monotonic()
            elapsed = time.monotonic() - self._silence_start
            return elapsed >= self._silence_duration_s
        else:
            # Sound detected — reset the silence timer
            self._silence_start = None
            return False

    def reset(self) -> None:
        """Reset the silence timer."""
        self._silence_start = None
