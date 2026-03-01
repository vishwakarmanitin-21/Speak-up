from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QPushButton

from src.ui.styles import MIC_BUTTON_STYLES


class MicButton(QPushButton):
    """Circular mic button with state-based styling."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(QSize(48, 48))
        self.setCursor(Qt.PointingHandCursor)
        self.set_state("idle")

    def set_state(self, state: str) -> None:
        """Set visual state: 'idle', 'recording', or 'processing'."""
        icons = {"idle": "🎤", "recording": "⏺", "processing": "⏳"}  # noqa: RUF001
        self.setText(icons.get(state, "🎤"))
        style = MIC_BUTTON_STYLES.get(state, MIC_BUTTON_STYLES["idle"])
        self.setStyleSheet(style)
