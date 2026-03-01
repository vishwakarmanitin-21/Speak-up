from __future__ import annotations

from src.config import Config
from src.context.clipboard import get_clipboard_text
from src.context.selection import get_selected_text
from src.context.session_memory import SessionMemory


class ContextBuilder:
    """Assembles context from configured sources."""

    def __init__(self, session_memory: SessionMemory) -> None:
        self._session_memory = session_memory
        self._config = Config()

    def build(self) -> str | None:
        """Build context string from enabled sources.

        Returns None if no context is available.
        """
        parts: list[str] = []

        if self._config.include_clipboard:
            clip = get_clipboard_text()
            if clip:
                parts.append(f"[Clipboard]\n{clip}")

        if self._config.include_selection:
            sel = get_selected_text()
            if sel:
                parts.append(f"[Selected Text]\n{sel}")

        if self._config.include_session_memory:
            mem = self._session_memory.get_context_summary()
            if mem:
                parts.append(f"[Session History]\n{mem}")

        if self._config.include_vscode_file:
            try:
                from src.context.vscode_context import get_vscode_file_context
                vsc = get_vscode_file_context()
                if vsc:
                    parts.append(vsc)
            except Exception:
                pass

        return "\n\n".join(parts) if parts else None
