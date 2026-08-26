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
user's words, while a miss just leaves the old behaviour.

WHY THE GUARDS BELOW ARE SO HEAVY
---------------------------------
Flattening every vowel to one symbol makes the key short and low-entropy, so
ordinary English words collide with dictionary terms outright: "houses",
"issues", "assess" and "excess" all fold to the same key as "Axis"; "medium"
and "Notion" fold to "Nitin"; "guides" to "Codex". Sliding windows made it
worse — they happily swallowed a neighbouring word ("of Vestora" -> "Vestora")
or a sentence boundary ("Vestora. I" -> "Vestora"), silently deleting speech.

So a sound-key hit is now treated as a *candidate*, not a verdict, and has to
survive three further checks:

  1. Vowel agreement — the sound key is re-checked with the vowels PUT BACK,
     and must still be within roughly a third of the term's length in edits.
     Vowels are dropped in the first pass because STT mangles them, but it
     mangles them a little ("Wursal"/"Vercel"), not beyond recognition; the
     collisions differ in almost every vowel ("houses"/"Axis").
  2. Ordinary-English protection — a word that is ordinary English (including
     its plural/past/-ing forms) is never overwritten.
  3. Window sanity — a multi-word window may not contain a function word, may
     not cross a clause or sentence boundary, may not already contain the term,
     and must match its key exactly rather than fuzzily.
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
# Below this many symbols, a vowel-sensitive key must match exactly: short
# acronyms ("Axis", "SEBI", "CAMS") sit one edit away from ordinary words.
_NO_SLACK_KEY_LEN = 5

# Punctuation that ends a clause or sentence. A term cannot have been split
# across one of these, so a window is never allowed to span it — that is how
# "Vestora. I" lost both the full stop and the "I".
_CLAUSE_PUNCT = frozenset(".?!,;:")

# Never rewrite these, however they sound — they are ordinary English. The short
# function words matter mainly inside multi-word windows, where the minimum
# term length cannot protect them: "of", "is", "our" and "the" were all eaten.
_NEVER_REPLACE = {
    "the", "this", "that", "there", "their", "they", "them", "then", "than",
    "and", "but", "for", "with", "from", "have", "has", "was", "were", "will",
    "what", "when", "where", "which", "while", "would", "could", "should",
    "about", "after", "before", "because", "please", "thanks", "thank", "your",
    "you", "our", "not", "are", "any", "all", "can", "may", "one", "two",
    "some", "such", "into", "over", "under", "just", "like", "make", "made",
    "want", "need", "here", "more", "most", "much", "many", "well", "very",
    # function words that only ever show up mid-window
    "a", "an", "as", "at", "be", "been", "being", "by", "do", "does", "did",
    "he", "her", "him", "his", "how", "i", "if", "in", "is", "it", "its",
    "me", "my", "no", "of", "off", "on", "or", "other", "out", "own", "per",
    "she", "so", "these", "those", "to", "up", "us", "via", "we",
    "who", "whom", "whose", "why", "yes", "yet", "am", "also", "only", "even",
    "ever", "each", "both", "same", "still", "every", "again", "already",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


# Everyday English that must never be overwritten by a dictionary term, and
# that must never BE a correction target either. Two jobs, one list:
#   - as a target: dictionaries pick up ordinary words over time ("Order",
#     "Speak", "Additionally"); matching on those rewrites normal English.
#   - as a candidate: "houses", "issues", "assess" and "medium" are real words
#     the user meant, whatever they happen to fold to.
# Deliberately NOT the vocab-learner stop-list: that one also holds brands like
# LinkedIn and GitHub, which are perfectly good dictionary entries.
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
    # observed collisions: every one of these was being silently overwritten
    # with a dictionary term in real dictations (see the module docstring)
    "house", "assess", "access", "excess", "medium", "notion", "guide", "case",
    "cause", "cost", "count", "con", "cons", "pros", "size", "sense", "since",
    "success", "process", "service", "source", "course", "choice", "chance",
    "class", "cash", "cast", "list", "least", "last", "next", "best", "test",
    "store", "restore", "state", "study", "system", "side", "step", "sets",
    "view", "video", "voice", "vault", "cloud", "clear", "credit",
    "debit", "bank", "fund", "funds", "asset", "assets", "share", "shares",
    "market", "price", "rate", "risk", "unit", "units", "date",
    "welcome", "boss", "oasis",
}


def _plain(text: str) -> str:
    """Letters only, lowercased — the form spellings are compared in."""
    return re.sub(r"[^a-z]", "", (text or "").lower())


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


def _is_english_candidate(word: str) -> bool:
    """True if the transcript word is ordinary English and must be left alone.

    Checks the plural/past/-ing forms too, so protecting "issue" also protects
    "issues" — otherwise the list would need every inflection spelled out and
    would still miss one.
    """
    low = _plain(word)
    if not low:
        return False
    if low in _NEVER_REPLACE or low in _ORDINARY_EXTRA:
        return True
    for suffix in ("ing", "ed", "es", "s"):
        if not low.endswith(suffix):
            continue
        base = low[: -len(suffix)]
        if len(base) < 3:
            continue
        if base in _NEVER_REPLACE or base in _ORDINARY_EXTRA:
            return True
        # "issues" -> "issue", "changes" -> "change": dropping the -s exposes a
        # base that still needs its silent -e put back.
        if suffix in ("es", "s") and (base + "e") in _ORDINARY_EXTRA:
            return True
    return False


def _vowel_allowance(term_key: str) -> int:
    """How far the vowel-sensitive keys may differ and still count as a match.

    Short keys get no slack at all: a four-symbol key is already so close to
    everything that one edit of leeway lets "oasis" reach "Axis" and "scipy"
    reach "SEBI". Longer names get the room a real mis-hearing needs — "Wursal"
    is two edits from "Vercel" and genuinely is one.
    """
    if len(term_key) < _NO_SLACK_KEY_LEN:
        return 0
    return max(1, len(term_key) // 3)


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


# The same folding, but with the vowels left intact. `sound_key` deliberately
# throws vowels away — that is what lets it match "Wellquest" to "WealQuest",
# and equally what makes "houses" collide with "Axis". This second key is the
# tie-breaker: it forgives the consonant swaps STT really makes (v/w, soft c)
# while still noticing that "houses" and "Axis" share no vowel at all.
_VOWEL_FOLD = dict(_LETTER_FOLD, a="a", e="e", i="i", o="o", u="u", y="i")


def vowel_key(text: str) -> str:
    """`sound_key`, but keeping each vowel's identity."""
    s = re.sub(r"[^a-z]", "", (text or "").lower())
    if not s:
        return ""
    for a, b in _DIGRAPHS:
        s = s.replace(a, b)
    s = re.sub(r"c(?=[eiy])", "s", s)
    folded = "".join(_VOWEL_FOLD.get(ch, ch) for ch in s)
    collapsed: list[str] = []
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


def _trailing_punct(token: str) -> str:
    """Whatever followed the word inside `token` ("Vestora," -> ",")."""
    m = _TOKEN_RE.search(token)
    return token[m.end():] if m else token


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

    def _lookup(self, key: str, allow_fuzzy: bool = True) -> str | None:
        if not key:
            return None
        hit = self._by_key.get(key)
        if hit is not None:
            return hit
        # Fuzzy matching is single-word only. Across a window it was matching a
        # term PLUS a neighbouring word ("of Vestora" is one edit from
        # "Vestora"), and replacing the pair deleted the neighbour.
        if allow_fuzzy and len(key) >= _FUZZY_MIN_KEY_LEN:
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
                # A word the user actually meant is never overwritten, however
                # it folds. This is what keeps "houses" from becoming "Axis".
                if size == 1 and _is_english_candidate(words[0]):
                    continue
                if size > 1 and not self._window_is_joinable(window):
                    continue
                key = sound_key(joined)
                if size > 1 and len(key) < _MULTIWORD_MIN_KEY:
                    continue      # too short to safely stitch words together
                term = self._lookup(key, allow_fuzzy=(size == 1))
                if term is None:
                    continue
                joined_plain, term_plain = _plain(joined), _plain(term)
                if joined_plain == term_plain:
                    continue                      # already correct — leave it alone
                # A window that already contains the term is not a mis-hearing;
                # merging it would swallow the neighbouring word.
                if size > 1 and any(_plain(w) == term_plain for w in words):
                    continue
                # Matching with the vowels thrown away is not enough — put them
                # back and the two must still agree. "houses" and "Axis" are
                # identical without vowels and unrecognisable with them.
                term_vk, cand_vk = vowel_key(term), vowel_key(joined)
                allowance = _vowel_allowance(term_vk)
                if _edit_distance(cand_vk, term_vk, cap=allowance) > allowance:
                    continue
                # Keep whatever punctuation trailed the last token of the window.
                trailing = _trailing_punct(window[-1])
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

    @staticmethod
    def _window_is_joinable(window: list[str]) -> bool:
        """Whether these adjacent tokens could plausibly be one split-up term.

        A term is spoken as one continuous sound, so it cannot straddle a clause
        or sentence break — that guard alone recovers "Vestora. I", "folder, I"
        and "want. Are you", all of which silently lost a word.

        Note there is deliberately NO "window contains a function word" rule:
        the whole point of windowing is that STT splits a name INTO ordinary
        words, and the flagship case ("west or a" -> "Vestora") is two function
        words out of three. The spelling-proximity check below is what keeps
        "we start the" and "what's there" from being swallowed instead.
        """
        return not any(
            ch in _CLAUSE_PUNCT
            for token in window[:-1]
            for ch in _trailing_punct(token)
        )


def build_corrector() -> PhoneticCorrector:
    """Build a corrector from the user's dictionary (honours the config switch)."""
    from src.config import Config

    config = Config()
    if not config.phonetic_correction:
        return PhoneticCorrector([])
    return PhoneticCorrector(config.custom_vocabulary)
