"""Tests for recently-added features.

Covers: cost estimation, auto-learning dictionary (VocabLearner), prompt
assembly, the streaming inserter (chunk boundaries, ordering, clipboard
restore), and the live→batch fallback notice.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --------------------------------------------------------------------------- #
# Cost estimation (usage_tracker)
# --------------------------------------------------------------------------- #

def test_estimate_cost_positive_and_model_sensitive():
    from src.services.usage_tracker import _estimate_cost

    mini = _estimate_cost(100, 100, "gpt-4o-transcribe", "gpt-4o-mini")
    big = _estimate_cost(100, 100, "gpt-4o-transcribe", "gpt-4o")
    assert mini > 0
    assert big > mini  # gpt-4o is pricier than gpt-4o-mini


def test_estimate_cost_local_stt_is_cheaper():
    from src.services.usage_tracker import _estimate_cost

    cloud = _estimate_cost(200, 50, "gpt-4o-transcribe", "gpt-4o-mini")
    local = _estimate_cost(200, 50, "local", "gpt-4o-mini")
    assert local < cloud  # local transcription has no per-minute charge


def test_cost_summary_shape():
    from src.services.usage_tracker import get_cost_summary

    c = get_cost_summary()
    for key in ("month_label", "month_runs", "month_cost", "total_runs",
                "total_cost", "avg_cost"):
        assert key in c


# --------------------------------------------------------------------------- #
# Auto-learning dictionary (VocabLearner)
# --------------------------------------------------------------------------- #

def _fresh_learner(tmp_path, monkeypatch):
    from src.config import Config
    from src.services.vocab_learner import VocabLearner

    # Pretend the dictionary is empty so nothing is pre-filtered as "known".
    monkeypatch.setattr(Config, "custom_vocabulary", property(lambda self: []))
    vl = VocabLearner()
    vl._path = tmp_path / "vocab_suggestions.json"
    vl._data = {"counts": {}, "ignored": []}
    return vl


def test_learner_suggests_recurring_proper_noun(tmp_path, monkeypatch):
    vl = _fresh_learner(tmp_path, monkeypatch)
    # Must appear capitalised MID-sentence (not just at a sentence start),
    # and recur past the threshold before being suggested.
    vl.observe("I spoke to Zephyrion today about the plan.")
    vl.observe("We saw Zephyrion again yesterday.")
    vl.observe("Ask Zephyrion about it.")
    assert "Zephyrion" in vl.pending_suggestions()


def test_learner_ignores_sentence_start_words(tmp_path, monkeypatch):
    vl = _fresh_learner(tmp_path, monkeypatch)
    # "Considering" / "Hope" only ever start sentences -> never suggested.
    vl.observe("Considering the plan. Hope it works.")
    vl.observe("Considering that. Hope so.")
    pending = vl.pending_suggestions()
    assert "Considering" not in pending
    assert "Hope" not in pending


def test_learner_ignores_common_and_single_words(tmp_path, monkeypatch):
    vl = _fresh_learner(tmp_path, monkeypatch)
    vl.observe("Today The Plan is good. Monday works.")
    vl.observe("The Plan continues.")
    pending = vl.pending_suggestions()
    # Common capitalised words (sentence starters / days) are never suggested.
    for word in ("Today", "The", "Monday"):
        assert word not in pending


def test_learner_ignore_removes_suggestion(tmp_path, monkeypatch):
    vl = _fresh_learner(tmp_path, monkeypatch)
    vl.observe("I called Zephyrion earlier.")   # mid-sentence -> 1
    vl.observe("Please meet Zephyrion now.")    # mid-sentence -> 2
    vl.observe("Tell Zephyrion today.")         # mid-sentence -> 3 (threshold)
    assert "Zephyrion" in vl.pending_suggestions()
    vl.ignore("Zephyrion")
    assert "Zephyrion" not in vl.pending_suggestions()


# --------------------------------------------------------------------------- #
# Phonetic repair of mis-heard dictionary terms (self-learning phase 1)
# --------------------------------------------------------------------------- #

_VOCAB = ["Nitin", "Prisha", "Komal", "Pranit", "Vestora", "WealQuest", "Wealducate"]


def _corrector():
    from src.services.phonetic import PhoneticCorrector
    return PhoneticCorrector(_VOCAB)


@pytest.mark.parametrize("heard,expected", [
    ("I spoke to west or a about it", "Vestora"),      # term split across words
    ("the well quest roadmap is ready", "WealQuest"),
    ("send it to well due cate today", "Wealducate"),
    ("Nithin will join", "Nitin"),                      # single mis-spelling
    ("ask Komall about it", "Komal"),
])
def test_phonetic_restores_misheard_terms(heard, expected):
    assert expected in _corrector().correct(heard)


@pytest.mark.parametrize("text", [
    "the west side of the building was quiet",   # 'west' alone is not Vestora
    "well, that went better than expected",
    "I want to make a request for the team",
    "please send the report before the meeting",
])
def test_phonetic_leaves_ordinary_english_alone(text):
    """A wrong replacement corrupts the user's words — precision over recall."""
    assert _corrector().correct(text) == text


@pytest.mark.parametrize("text", [
    "she came all the way from the coast",   # 'came all' folds like 'Komal'
    "it is not in the report yet",           # 'not in'   folds like 'Nitin'
    "please come all the way here",
])
def test_phonetic_does_not_stitch_ordinary_words_into_a_name(text):
    """Joining words to reach a SHORT key is the main false-positive risk.

    Regression: 'she came all the way' once became 'she Komal the way'.
    """
    from src.services.phonetic import PhoneticCorrector
    assert PhoneticCorrector(["Komal", "Nitin", "Vestora"]).correct(text) == text


def test_phonetic_leaves_already_correct_terms_untouched():
    text = "Vestora and WealQuest are both fine"
    assert _corrector().correct(text) == text


def test_phonetic_preserves_punctuation():
    out = _corrector().correct("I called west or a, then left.")
    assert "Vestora," in out and out.endswith("left.")


def test_phonetic_noop_without_vocabulary():
    from src.services.phonetic import PhoneticCorrector
    text = "west or a and well quest"
    assert PhoneticCorrector([]).correct(text) == text


def test_deepgram_url_boosts_dictionary_terms(monkeypatch):
    """The dictionary must reach the live transcriber, not just the rewrite."""
    import src.transcription.deepgram_client as dg

    class _Cfg:
        custom_vocabulary = ["Vestora", "WealQuest"]

    monkeypatch.setattr(dg, "Config", lambda: _Cfg())
    url = dg._build_url()
    assert "keywords=Vestora%3A2" in url
    assert "keywords=WealQuest%3A2" in url
    assert "model=nova-2" in url  # base params still present


# --------------------------------------------------------------------------- #
# Live-caption typewriter reveal
# --------------------------------------------------------------------------- #

def test_caption_reveal_grows_smoothly_and_reaches_target():
    from src.ui.components.caption_window import CaptionWindow as C

    # A 2-3 word chunk lands at once; the reveal walks toward it over frames
    # (never overshooting), and converges exactly on the target.
    cur, target = "", "Sending a trial message"
    steps = 0
    while cur != target:
        cur = target[: C._reveal_len(cur, target)]
        steps += 1
        assert target.startswith(cur)        # only ever a prefix of the target
        assert steps < 200                    # always converges
    assert cur == target
    assert steps > 1                          # revealed gradually, not one jump


def test_caption_reveal_handles_shrink_and_revision():
    from src.ui.components.caption_window import CaptionWindow as C

    # Interim shrank / finalized to a shorter prefix -> snap straight to it.
    assert C._reveal_len("hello world", "hello") == len("hello")
    # Interim was revised mid-word -> fall back toward the shared prefix.
    assert C._reveal_len("hello", "hello world") <= len("hello world")


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #

def test_build_user_prompt_includes_text_and_vocab():
    from src.rewrite.modes import RewriteMode
    from src.rewrite.prompts import build_user_prompt

    prompt = build_user_prompt(
        RewriteMode.CLEAN_GRAMMAR,
        "hello there world",
        context="prior note",
        vocabulary=["Vestora"],
    )
    assert "hello there world" in prompt
    assert "Vestora" in prompt


def test_vocabulary_prompt_forbids_forcing_a_match():
    """Regression: 'finish' was being rewritten to a term containing 'fin'.

    The old wording ("if a spoken word sounds like one of these") was loose
    enough that the model matched ordinary English on a few shared sounds.
    """
    from src.rewrite.modes import RewriteMode
    from src.rewrite.prompts import build_user_prompt

    prompt = build_user_prompt(
        RewriteMode.CLEAN_GRAMMAR, "let me finish this", vocabulary=["KFinTech"]
    )
    assert "Do NOT force a match" in prompt
    assert "when in doubt, keep the dictated word" in prompt


def test_guard_comes_after_the_transcript():
    """The anti-answering guard must be LAST.

    When the transcript is the final thing in the prompt, a spoken imperative
    ("give me a step-by-step guide...") reads as the live instruction and the
    model obeys it instead of transcribing. The guard has to come after it.
    """
    from src.rewrite.modes import RewriteMode
    from src.rewrite.prompts import build_user_prompt

    spoken = "give me a step-by-step guide to connect to the server"
    prompt = build_user_prompt(RewriteMode.SMART, spoken, context="prior stuff")

    assert spoken in prompt
    assert prompt.index(spoken) < prompt.index("Reminder before you answer")
    assert "NOT a task for you" in prompt
    assert "Do not copy anything from the Context section" in prompt
    # The guard must be the FINAL block — no input/context section after it.
    tail = prompt[prompt.index("Reminder before you answer"):]
    assert "--- " not in tail


def test_session_memory_dedupes_repeated_dictations():
    """Repeating a dictation must not fill the context with N copies of it."""
    from src.context.session_memory import SessionMemory

    m = SessionMemory()
    dmarc = "v=DMARC1; p=none; rua=mailto:dmarc@example.com; fo=1"
    for _ in range(3):
        m.add(dmarc, dmarc, "smart")
    summary = m.get_context_summary()
    assert summary.count("v=DMARC1") == 1

    # Distinct entries are still all kept.
    m2 = SessionMemory()
    for t in ("alpha", "beta", "gamma"):
        m2.add(t, t, "smart")
    s2 = m2.get_context_summary()
    for t in ("alpha", "beta", "gamma"):
        assert t in s2


def test_system_prompt_has_number_formatting_rule():
    """Spoken numbers must come out as digits where they carry data."""
    from src.rewrite.prompts import SYSTEM_PROMPT

    assert "Numbers:" in SYSTEM_PROMPT
    assert "DIGITS" in SYSTEM_PROMPT
    # The rule must cover the data-carrying cases and keep casual prose spelled out.
    for kind in ("money", "percentages", "times", "versions"):
        assert kind in SYSTEM_PROMPT
    assert "one-to-nine" in SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Streaming inserter: boundaries, ordering, clipboard restore
# --------------------------------------------------------------------------- #

def test_flush_boundaries():
    from src.output.inserter import OutputInserter as O

    # First chunk flushes after a few words; body waits for a clause/sentence.
    assert O._should_flush_first("Before we rebuild ") is True
    assert O._should_flush_first("Hi ") is False
    assert O._should_flush("a short run ") is False
    assert O._should_flush("a full clause, ") is True


class _FakeClip:
    def __init__(self, val=""):
        self.clip = val
        self.copies = []

    def paste(self):
        return self.clip

    def copy(self, v):
        self.clip = v
        self.copies.append(v)


def _make_inserter(monkeypatch, fake_clip):
    monkeypatch.setattr("src.output.inserter.pyperclip", fake_clip)
    from src.output.inserter import OutputInserter
    return OutputInserter()


def test_stream_preserves_order(monkeypatch):
    fake = _FakeClip("ORIGINAL")
    inst = _make_inserter(monkeypatch, fake)
    recorded: list[str] = []
    inst._paste = lambda chunk: recorded.append(chunk)  # don't touch real keyboard

    inst.begin_stream()
    for ch in "Before we rebuild, we can address this. Second sentence here now. ":
        inst.feed_stream(ch)
    inst.end_stream()  # drains + joins the worker

    assert "".join(recorded).strip() == (
        "Before we rebuild, we can address this. Second sentence here now."
    )


def _stub_paste(inst, fake):
    """Stand-in for OutputInserter._paste: writes the clipboard and records what
    we last put there (the real method does both), but touches no keyboard."""
    def _paste(chunk):
        fake.copy(chunk)
        inst._last_pasted = chunk
    return _paste


def _wait_for(predicate, timeout=2.0):
    """Wait for a background (delayed) clipboard restore to land."""
    import time as _t

    deadline = _t.monotonic() + timeout
    while _t.monotonic() < deadline:
        if predicate():
            return True
        _t.sleep(0.01)
    return False


def test_stream_restores_clipboard(monkeypatch):
    monkeypatch.setattr("src.output.inserter._RESTORE_DELAY_S", 0.02)
    fake = _FakeClip("ORIGINAL")
    inst = _make_inserter(monkeypatch, fake)
    inst._paste = _stub_paste(inst, fake)   # pastes for real, minus the keyboard

    inst.begin_stream()           # snapshots "ORIGINAL"
    inst.feed_stream("Some new dictated text here. ")
    inst.end_stream()             # schedules the guarded restore

    assert _wait_for(lambda: fake.clip == "ORIGINAL"), f"got {fake.clip!r}"


def test_restore_is_delayed_not_immediate(monkeypatch):
    """The restore must NOT land while a Ctrl+V we sent could still be pending.

    Restoring immediately is what made dictation paste the user's OLD clipboard
    instead of the transcript.
    """
    # A long delay makes this deterministic: the restore cannot plausibly fire
    # before the assertion below, however loaded the machine is.
    monkeypatch.setattr("src.output.inserter._RESTORE_DELAY_S", 30.0)
    fake = _FakeClip("ORIGINAL")
    inst = _make_inserter(monkeypatch, fake)
    inst._paste = _stub_paste(inst, fake)

    inst.begin_stream()
    inst.feed_stream("Dictated sentence lands here. ")
    inst.end_stream()

    # Right after end_stream the dictation must still own the clipboard.
    assert fake.clip != "ORIGINAL", "restored too early — a pending paste would insert the old clipboard"


def test_no_restore_when_nothing_was_pasted(monkeypatch):
    """If we never wrote the clipboard, we must not 'restore' over it.

    Otherwise a dictation that produced no output would blindly overwrite
    whatever the user copied in the meantime.
    """
    monkeypatch.setattr("src.output.inserter._RESTORE_DELAY_S", 0.02)
    fake = _FakeClip("ORIGINAL")
    inst = _make_inserter(monkeypatch, fake)

    inst.begin_stream()
    inst.end_stream()                       # no chunks fed → nothing pasted
    fake.copy("USER COPIED THIS AFTERWARDS")

    import time as _t
    _t.sleep(0.2)
    assert fake.clip == "USER COPIED THIS AFTERWARDS"


def test_restore_skipped_when_another_app_takes_clipboard(monkeypatch):
    """If something else owns the clipboard by restore time, don't clobber it."""
    # Generous delay so the competing copy below is definitely in place before
    # the restore worker wakes — otherwise this test would race the scheduler.
    monkeypatch.setattr("src.output.inserter._RESTORE_DELAY_S", 0.4)
    fake = _FakeClip("ORIGINAL")
    inst = _make_inserter(monkeypatch, fake)
    inst._paste = _stub_paste(inst, fake)

    inst.begin_stream()
    inst.feed_stream("Dictated text. ")
    inst.end_stream()
    fake.copy("USER COPIED SOMETHING ELSE")  # another app wins the clipboard

    import time as _t
    _t.sleep(1.2)                             # well past the restore delay
    assert fake.clip == "USER COPIED SOMETHING ELSE"


# --------------------------------------------------------------------------- #
# Live → batch fallback notice (pipeline)
# --------------------------------------------------------------------------- #

def _make_pipeline():
    with (
        patch("src.services.pipeline.AudioRecorder"),
        patch("src.services.pipeline.SilenceDetector"),
        patch("src.services.pipeline.RewriteEngine"),
        patch("src.services.pipeline.SessionMemory"),
        patch("src.services.pipeline.ContextBuilder"),
        patch("src.services.pipeline.OutputInserter"),
    ):
        from src.services.pipeline import Pipeline
        return Pipeline()


@pytest.mark.asyncio
async def test_realtime_fallback_emits_notice():
    from src.rewrite.modes import RewriteMode

    pipeline = _make_pipeline()
    notices: list[str] = []
    pipeline.set_notice_callback(notices.append)

    realtime = MagicMock()
    realtime.used_fallback = True
    realtime.stop_and_transcribe = AsyncMock(return_value="hello world")
    pipeline._use_realtime = True
    pipeline._realtime = realtime
    pipeline._rewriter.rewrite = AsyncMock(return_value="Hello world.")
    pipeline._context_builder.build.return_value = None

    with (
        patch("src.services.usage_tracker.record_run"),
        patch("src.services.vocab_learner.VocabLearner"),
    ):
        # clipboard output -> non-streaming path (uses rewriter.rewrite)
        await pipeline.process(RewriteMode.CLEAN_GRAMMAR, output_mode="clipboard")

    assert any("standard mode" in n for n in notices)
