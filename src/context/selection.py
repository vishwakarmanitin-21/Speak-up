from __future__ import annotations

import time

import pyperclip
from pynput.keyboard import Controller, Key


def get_selected_text() -> str | None:
    """Attempt to get currently selected text from the active window.

    Strategy: save clipboard, simulate Ctrl+C, read clipboard, restore original.
    Returns None if no text was selected.

    Note: This should be called *before* recording starts, not during.
    The simulated keystrokes could interfere with the user's workflow.
    """
    keyboard = Controller()
    original_clipboard = ""
    try:
        original_clipboard = pyperclip.paste()
    except Exception:
        pass

    try:
        # Clear clipboard
        pyperclip.copy("")

        # Simulate Ctrl+C
        keyboard.press(Key.ctrl)
        keyboard.press("c")
        keyboard.release("c")
        keyboard.release(Key.ctrl)

        # Brief delay for clipboard to update
        time.sleep(0.15)

        selected = pyperclip.paste()

        if selected and selected.strip():
            return selected.strip()
        return None

    except Exception:
        return None
    finally:
        # Restore original clipboard
        try:
            pyperclip.copy(original_clipboard)
        except Exception:
            pass
