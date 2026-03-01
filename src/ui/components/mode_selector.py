from __future__ import annotations

from PyQt5.QtWidgets import QComboBox

from src.rewrite.modes import RewriteMode
from src.ui.styles import COMBOBOX_STYLE


class ModeSelector(QComboBox):
    """Dropdown to select the active rewrite mode."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(COMBOBOX_STYLE)
        for mode in RewriteMode:
            self.addItem(mode.display_name, mode)

    def current_mode(self) -> RewriteMode:
        return self.currentData()

    def set_mode(self, mode: RewriteMode) -> None:
        for i in range(self.count()):
            if self.itemData(i) == mode:
                self.setCurrentIndex(i)
                break
