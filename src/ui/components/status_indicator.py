from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel

from src.ui.styles import STATUS_STYLES


class StatusIndicator(QLabel):
    """Displays current app status with color coding."""

    STATE_TEXT = {
        "idle": "Ready",
        "listening": "Listening...",
        "processing": "Processing...",
        "done": "Done",
        "error": "Error",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumWidth(80)
        self.set_state("idle")

    def set_state(self, state: str, message: str | None = None) -> None:
        text = self.STATE_TEXT.get(state, "Ready")
        self.setText(text)
        style = STATUS_STYLES.get(state, STATUS_STYLES["idle"])
        self.setStyleSheet(style)
        # Show full error message as tooltip so the user can read it
        self.setToolTip(message if message else "")
