"""Deterministic repair of personal-dictionary terms the transcriber mis-heard.

Speech-to-text reliably mangles proper nouns it has never seen — and it often
SPLITS them into ordinary words ("Vestora" -> "west or a", "WealQuest" ->
"well quest"). The rewrite prompt asks the model to fix these, but a model call
is fuzzy, costs tokens, and can decline; a phonetic match is exact, free and
instant.

How it works: every dictionary term is reduced to a "sound key" that folds the
consonant groups speech recognisers confuse (v/w/f, b/p, d/t, k/c/g/q, s/z,
m/n) and flattens vowels, since vowels are what STT mangles most. Sliding
n-gram windows over the transcript get the same treatment, so a term split
across several words still collapses to the same key and is put back together.

This is Phase 1 of the self-learning roadmap: it improves accuracy using only
what the user already told us (their dictionary) — no correction capture, no
stored speech, no privacy trade-off, no added latency.

Precision is deliberately favoured over recall: a wrong replacement corrupts the
user's words, while a miss just leaves the old behaviour. Hence the guards —
minimum term length, no replacing common English words, and exact key matching
(near-matches only for long, distinctive keys).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("speakup")

# Sounds that speech recognisers routinely swap. Each group folds to one symbol.
_DIGRAPHS = (
    ("ph", "f"), ("ck", "k"), ("sh", "j"), ("ch", "j"), ("th", "t"),
    ("gh", "k"), ("kn", "n"), ("wr", "r"), ("qu", "kw"),
)
_LETTER_FOLD = {
    "v": "f", "w": "f", "f": "f",
    "b": "p", "p": "p",
    "d": "t", "t": "t",
    "g": "k", "k": "k", "c": "k", "q": "k",
    "s": "s", "z": "s", "x": "s",
    "m": "n", "n": "n",
    "j": "j", "l": "l", "r": "r", "y": "a",
    "h": "",   # effectively silent for matching purposes
    "a": "a", "e": "a", "i": "a", "o": "a", "u": "a",   # vowels flattened
}

_MIN_TERM_LEN = 4        # shorter terms fold to keys that collide with everything
_FUZZY_MIN_KEY_LEN = 7   # only long, distinctive keys may match inexactly
_MAX_WINDOW = 4          # how many transcript words a term may have been split into
# Joining several words to reach a SHORT key is where false positives come from:
# "came all" folds to the same key as "Komal", "not in" to the same as "Nitin".
# Multi-word matches therefore have to clear a longer, more distinctive key —
# "Vestora" (7), "WealQuest" (8) and "Wealducate" (9) still qualify.
_MULTIWORD_MIN_KEY = 6

# Never rewrite these, however they sound — they are ordinary English.
_NEVER_REPLACE = {
    "the", "this", "that", "there", "their", "they", "them", "then", "than",
    "and", "but", "for", "with", "from", "have", "has", "was", "were", "will",
    "what", "when", "where", "which", "while", "would", "could", "should",
    "about", "after", "before", "because", "please", "thanks", "thank", "your",
    "you", "our", "not", "are", "any", "all", "can", "may", "one", "two",
    "some", "such", "into", "over", "under", "just", "like", "make", "made",
    "want", "need", "here", "more", "most", "much", "many", "well", "very",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


# Everyday English that sometimes ends up in a personal dictionary (accepted
# suggestions, typos). Deliberately NOT the vocab-learner stop-list: that one
# also holds brands like LinkedIn and GitHub, which are perfectly good
# dictionary entries — this set is only about ordinary words.
_ORDINARY_EXTRA = {
    "order", "world", "speak", "code", "note", "notes", "report", "review",
    "update", "draft", "email", "team", "plan", "action", "task", "time",
    "week", "month", "year", "day", "meeting", "project", "design", "sales",
    "finance", "budget", "summary", "message", "answer", "question", "story",
    "point", "issue", "value", "number", "money", "people", "person", "thing",
    "place", "right", "left", "back", "front", "start", "stop", "help", "work",
    "call", "send", "read", "write", "open", "close", "check", "change",
    # discourse adverbs that often start a sentence and get mis-learned
    "additionally", "however", "therefore", "basically", "actually", "finally",
    "currently", "generally", "obviously", "definitely", "probably", "overall",
}


def _is_ordinary_word(term: str) -> bool:
    """True if `term` is everyday English rather than a name/jargon.

    Such entries must not be correction targets (they would rewrite normal
    speech) nor boosted at the transcriber (Deepgram warns that boosting common
    words raises false positives). Real brands/names are left alone.
    """
    low = term.strip().lower()
    if "'" in low:                       # contractions: "I'm", "Let's"
        return True
    return low in _NEVER_REPLACE or low in _ORDINARY_EXTRA


def sound_key(text: str) -> str:
    """Fold text to a key that survives common speech-recognition confusions."""
    s = re.sub(r"[^a-z]", "", (text or "").lower())
    if not s:
        return ""
    for a, b in _DIGRAPHS:
        s = s.replace(a, b)
    # Soft c: before e/i/y it is an /s/ sound, not /k/ — "Vercel" is "ver-SELL",
    # so it must fold like the "s" a transcriber hears ("WERSAL"). Note 'g' is
    # NOT treated the same way: "get"/"give" keep a hard g before e/i.
    s = re.sub(r"c(?=[eiy])", "s", s)
    out = [_LETTER_FOLD.get(ch, ch) for ch in s]
    folded = "".join(out)
    # Collapse runs ("ll" -> "l"): doubled sounds are not distinguishable.
    collapsed = []
    for ch in folded:
        if not collapsed or collapsed[-1] != ch:
            collapsed.append(ch)
    return "".join(collapsed)


def _edit_distance(a: str, b: str, cap: int = 2) -> int:
    """Levenshtein distance, short-circuited once it exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


class PhoneticCorrector:
    """Restores dictionary terms that were mis-heard or split into other words."""

    def __init__(self, terms: list[str]) -> None:
        self._by_key: dict[str, str] = {}
        self._max_window = 1
        for term in terms or []:
            term = (term or "").strip()
            if len(term.replace(" ", "")) < _MIN_TERM_LEN:
                continue
            # Dictionaries pick up ordinary words over time ("Order", "Speak",
            # "Additionally"). Matching on those would rewrite normal English, so
            # they are never used as correction targets.
            if _is_ordinary_word(term):
                continue
            key = sound_key(term)
            if not key:
                continue
            # First term wins, so an earlier dictionary entry is never shadowed.
            self._by_key.setdefault(key, term)
            self._max_window = max(self._max_window, min(_MAX_WINDOW, len(term.split())+ 2))

    @property
    def active(self) -> bool:
        return bool(self._by_key)

    def _lookup(self, key: str) -> str | None:
        if not key:
            return None
        hit = self._by_key.get(key)
        if hit is not None:
            return hit
        if len(key) >= _FUZZY_MIN_KEY_LEN:
            # Allow one dropped/added sound, but only for long distinctive keys.
            best, best_d = None, 2
            for k, term in self._by_key.items():
                if abs(len(k) - len(key)) > 1 or len(k) < _FUZZY_MIN_KEY_LEN:
                    continue
                d = _edit_distance(k, key, cap=1)
                if d < best_d:
                    best, best_d = term, d
            if best is not None and best_d <= 1:
                return best
        return None

    def correct(self, text: str) -> str:
        """Return `text` with mis-heard dictionary terms restored."""
        if not text or not self._by_key:
            return text
        try:
            return self._correct(text)
        except Exception as e:            # never let a cosmetic fix break output
            logger.debug("phonetic correction skipped: %s", e)
            return text

    def _correct(self, text: str) -> str:
        tokens = text.split(" ")
        out: list[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            matched = False
            # Longest window first, so a multi-word term wins over a single word.
            for size in range(min(self._max_window, n - i), 0, -1):
                window = tokens[i:i + size]
                words = [m.group(0) for t in window if (m := _TOKEN_RE.search(t))]
                if len(words) != size:
                    continue
                joined = "".join(words)
                if len(joined) < _MIN_TERM_LEN:
                    continue
                if size == 1 and words[0].lower() in _NEVER_REPLACE:
                    continue
                key = sound_key(joined)
                if size > 1 and len(key) < _MULTIWORD_MIN_KEY:
                    continue      # too short to safely stitch words together
                term = self._lookup(key)
                if term is None:
                    continue
                if joined.lower() == term.replace(" ", "").lower():
                    continue                      # already correct — leave it alone
                # Keep whatever punctuation trailed the last token of the window.
                last = window[-1]
                m = _TOKEN_RE.search(last)
                trailing = last[m.end():] if m else ""
                leading = ""
                m0 = _TOKEN_RE.search(window[0])
                if m0:
                    leading = window[0][:m0.start()]
                out.append(f"{leading}{term}{trailing}")
                logger.info("Phonetic fix: %r -> %r", " ".join(window), term)
                i += size
                matched = True
                break
            if not matched:
                out.append(tokens[i])
                i += 1
        return " ".join(out)


def build_corrector() -> PhoneticCorrector:
    """Build a corrector from the user's dictionary (honours the config switch)."""
    from src.config import Config

    config = Config()
    if not config.phonetic_correction:
        return PhoneticCorrector([])
    return PhoneticCorrector(config.custom_vocabulary)
