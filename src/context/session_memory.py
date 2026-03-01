from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SessionEntry:
    raw_text: str
    rewritten_text: str
    mode: str
    timestamp: datetime = field(default_factory=datetime.now)


class SessionMemory:
    """In-memory session history for context continuity."""

    def __init__(self, max_entries: int = 10) -> None:
        self._entries: list[SessionEntry] = []
        self._max_entries = max_entries

    def add(self, raw_text: str, rewritten_text: str, mode: str) -> None:
        entry = SessionEntry(
            raw_text=raw_text,
            rewritten_text=rewritten_text,
            mode=mode,
        )
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries.pop(0)

    def get_context_summary(self, last_n: int = 3) -> str | None:
        """Return a summary of recent entries for context."""
        if not self._entries:
            return None
        recent = self._entries[-last_n:]
        parts = []
        for entry in recent:
            parts.append(f"[{entry.mode}] {entry.rewritten_text[:200]}")
        return "\n\n".join(parts)

    def clear(self) -> None:
        self._entries.clear()
