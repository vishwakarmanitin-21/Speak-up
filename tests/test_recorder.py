"""Tests for AudioRecorder."""
from __future__ import annotations

import io
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_recorder():
    from src.audio.recorder import AudioRecorder
    return AudioRecorder(sample_rate=16000)


def test_get_wav_bytes_raises_when_no_audio():
    """get_wav_bytes raises ValueError when nothing has been recorded."""
    recorder = _make_recorder()
    with pytest.raises(ValueError, match="No audio recorded"):
        recorder.get_wav_bytes()


def test_get_wav_bytes_returns_bytesio():
    """get_wav_bytes returns a seeked BytesIO with a .name attribute."""
    recorder = _make_recorder()
    # Inject a fake audio frame
    recorder._frames = [np.zeros((1024, 1), dtype=np.float32)]
    result = recorder.get_wav_bytes()
    assert isinstance(result, io.BytesIO)
    assert result.name == "recording.wav"
    assert result.tell() == 0  # rewound to start


def test_start_raises_recording_error_on_portaudio_failure():
    """start() wraps PortAudioError as RecordingError."""
    import sounddevice as sd
    from src.services.error_handler import RecordingError

    recorder = _make_recorder()
    with patch("sounddevice.InputStream", side_effect=sd.PortAudioError("no mic")):
        with pytest.raises(RecordingError):
            recorder.start()
    assert not recorder.is_recording


def test_stop_is_safe_when_not_started():
    """stop() does not raise if called without start()."""
    recorder = _make_recorder()
    recorder.stop()  # should not raise


def test_audio_callback_accumulates_frames():
    """_audio_callback appends frames when recording."""
    recorder = _make_recorder()
    recorder._is_recording = True
    frame = np.ones((512, 1), dtype=np.float32)
    recorder._audio_callback(frame, 512, None, None)
    recorder._audio_callback(frame, 512, None, None)
    assert len(recorder._frames) == 2
