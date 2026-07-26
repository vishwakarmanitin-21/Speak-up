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
        """Return a summary of recent entries for context.

        Duplicates are collapsed: repeating the same dictation would otherwise
        fill the context with N copies of one string, which heavily biases the
        rewrite toward reproducing it verbatim in the output.
        """
        if not self._entries:
            return None
        parts: list[str] = []
        seen: set[str] = set()
        for entry in reversed(self._entries):          # newest first
            text = entry.rewritten_text[:200]
            key = text.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            parts.append(f"[{entry.mode}] {text}")
            if len(parts) >= last_n:
                break
        if not parts:
            return None
        return "\n\n".join(reversed(parts))            # restore chronological order

    def clear(self) -> None:
        self._entries.clear()
