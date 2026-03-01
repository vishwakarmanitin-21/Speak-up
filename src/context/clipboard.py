from __future__ import annotations

import pyperclip


def get_clipboard_text() -> str | None:
    """Return current clipboard text, or None if empty/non-text."""
    try:
        text = pyperclip.paste()
        return text if text and text.strip() else None
    except Exception:
        return None
