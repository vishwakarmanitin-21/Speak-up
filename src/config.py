from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _find_paths() -> tuple[Path, Path]:
    """Return (bundle_dir, data_dir) for both dev and PyInstaller frozen modes.

    bundle_dir — read-only bundled assets (config_defaults.json, .env.example).
                 In PyInstaller onefile mode this is sys._MEIPASS (temp extraction dir).
    data_dir   — user-writable files (config.json, .env, usage_stats.json).
                 In frozen mode this is the directory containing the .exe.
    """
    if getattr(sys, "frozen", False):
        bundle = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        data = Path(sys.executable).parent
        return bundle, data
    root = Path(__file__).resolve().parent.parent
    return root, root


_BUNDLE_DIR, _DATA_DIR = _find_paths()


class Config:
    """Singleton configuration manager.

    Loads settings from three layers (lowest to highest priority):
    1. config_defaults.json (committed defaults)
    2. config.json (user overrides, git-ignored)
    3. .env file (API keys only)
    """

    _instance: Config | None = None

    def __new__(cls) -> Config:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self) -> None:
        # Load .env for API keys (user-writable, lives next to exe)
        env_path = _DATA_DIR / ".env"
        load_dotenv(env_path, override=True)

        # Load defaults (read-only, bundled inside the exe)
        defaults_path = _BUNDLE_DIR / "config_defaults.json"
        with open(defaults_path, encoding="utf-8") as f:
            self._defaults: dict = json.load(f)

        # Load user overrides (user-writable, lives next to exe)
        self._overrides: dict = {}
        overrides_path = _DATA_DIR / "config.json"
        if overrides_path.exists():
            with open(overrides_path, encoding="utf-8") as f:
                self._overrides = json.load(f)

    def _get(self, key: str, fallback=None):
        """Get a config value: overrides > defaults > fallback."""
        if key in self._overrides:
            return self._overrides[key]
        if key in self._defaults:
            return self._defaults[key]
        return fallback

    def reload(self) -> None:
        """Reload config from disk (call after settings change to hot-reload)."""
        self._load()

    # --- API Keys (from environment) ---

    @property
    def openai_api_key(self) -> str:
        from src.services.error_handler import APIKeyError

        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise APIKeyError(
                "OPENAI_API_KEY not set. Copy .env.example to .env and add your key.",
                "OpenAI API key is not configured. Add it in Settings or your .env file.",
            )
        return key

    # --- Model Settings ---

    @property
    def whisper_model(self) -> str:
        return self._get("whisper_model", "whisper-1")

    @property
    def gpt_model(self) -> str:
        return self._get("gpt_model", "gpt-4o")

    @property
    def temperature(self) -> float:
        return float(self._get("temperature", 0.3))

    # --- Audio Settings ---

    @property
    def sample_rate(self) -> int:
        return int(self._get("sample_rate", 16000))

    @property
    def silence_timeout_ms(self) -> int:
        return int(self._get("silence_timeout_ms", 2000))

    @property
    def auto_stop_on_silence(self) -> bool:
        return bool(self._get("auto_stop_on_silence", False))

    # --- Hotkey ---

    @property
    def hotkey(self) -> str:
        return self._get("hotkey", "ctrl+shift+space")

    # --- Behavior ---

    @property
    def default_rewrite_mode(self) -> str:
        return self._get("default_rewrite_mode", "clean_grammar")

    @property
    def output_mode(self) -> str:
        return self._get("output_mode", "clipboard")

    # --- Context Toggles ---

    @property
    def include_clipboard(self) -> bool:
        return bool(self._get("include_clipboard", True))

    @property
    def include_selection(self) -> bool:
        return bool(self._get("include_selection", True))

    @property
    def include_session_memory(self) -> bool:
        return bool(self._get("include_session_memory", True))

    @property
    def include_vscode_file(self) -> bool:
        return bool(self._get("include_vscode_file", False))

    # --- Transcription Provider ---

    @property
    def transcription_provider(self) -> str:
        """'cloud' (OpenAI Whisper API) or 'local' (faster-whisper)."""
        return self._get("transcription_provider", "cloud")

    @property
    def whisper_local_model_size(self) -> str:
        """faster-whisper model size: tiny, base, small, medium, large."""
        return self._get("whisper_local_model_size", "base")

    # --- Widget Appearance ---

    @property
    def widget_position(self) -> str:
        """'bottom_right', 'bottom_left', or 'bottom_center'."""
        return self._get("widget_position", "bottom_right")

    @property
    def widget_scale(self) -> str:
        """'compact' (0.5x), 'normal' (1x), or 'large' (2x)."""
        return self._get("widget_scale", "normal")

    # --- Startup ---

    @property
    def auto_start(self) -> bool:
        """Whether FlowAI should start automatically with Windows."""
        return bool(self._get("auto_start", False))

    # --- Usage Analytics ---

    @property
    def track_usage(self) -> bool:
        return bool(self._get("track_usage", True))

    # --- Mutation ---

    def save_user_overrides(self, overrides: dict) -> None:
        """Write user overrides to config.json (next to the exe / project root)."""
        self._overrides.update(overrides)
        overrides_path = _DATA_DIR / "config.json"
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(self._overrides, f, indent=4)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None
