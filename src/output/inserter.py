from __future__ import annotations

import time

import pyperclip
from pynput.keyboard import Controller, Key

from src.config import Config


class OutputMode:
    AUTO_PASTE = "auto_paste"
    CLIPBOARD = "clipboard"
    PREVIEW = "preview"


class OutputInserter:
    """Delivers rewritten text to the user."""

    def __init__(self) -> None:
        self._config = Config()
        self._keyboard = Controller()

    def deliver(self, text: str, mode: str | None = None) -> str:
        """Deliver text using the specified output mode.

        Args:
            text: The rewritten text to deliver.
            mode: Override output mode (uses config default if None).

        Returns:
            The output mode that was used.
        """
        mode = mode or self._config.output_mode

        if mode == OutputMode.AUTO_PASTE:
            self._auto_paste(text)
        elif mode == OutputMode.CLIPBOARD:
            self._copy_to_clipboard(text)
        elif mode == OutputMode.PREVIEW:
            # Preview is handled by the UI layer
            self._copy_to_clipboard(text)
        else:
            self._copy_to_clipboard(text)

        return mode

    def _auto_paste(self, text: str) -> None:
        """Copy to clipboard and simulate Ctrl+V at cursor position."""
        pyperclip.copy(text)
        time.sleep(0.05)
        self._keyboard.press(Key.ctrl)
        self._keyboard.press("v")
        self._keyboard.release("v")
        self._keyboard.release(Key.ctrl)

    def _copy_to_clipboard(self, text: str) -> None:
        """Just copy to clipboard."""
        pyperclip.copy(text)
