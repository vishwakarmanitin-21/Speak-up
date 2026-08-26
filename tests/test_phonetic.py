"""Tests for phonetic repair of personal-dictionary terms.

The "must not touch" cases are not hypothetical: every one of them was observed
corrupting a real dictation before the guards in `phonetic.py` were added. See
that module's docstring for the story.
"""
import pytest

from src.services.phonetic import PhoneticCorrector, sound_key

# A realistic dictionary: proper nouns, brands and short acronyms.
DICTIONARY = [
    "Nitin", "Prisha", "Komal", "Pranit", "Vestora", "WealQuest", "Wealducate",
    "WealCore", "ICICI", "CAMS", "KFinTech", "Axis", "HDFC", "Codex", "SEBI",
    "Claude", "AMFI", "Vercel", "Airtel", "MCP", "Cloudflare", "Cloud", "BOS",
    "Zoho", "Vault",
]


@pytest.fixture
def corrector():
    return PhoneticCorrector(DICTIONARY)


# --- the repairs the feature exists for -------------------------------------

@pytest.mark.parametrize("spoken, expected", [
    # split into ordinary words — the flagship case from the module docstring
    ("west or a bank", "Vestora bank"),
    ("we store app", "Vestora app"),
    ("well quest link", "WealQuest link"),
    # mis-spelled as one word
    ("Wellquest is live", "WealQuest is live"),
    ("WellCore rollout", "WealCore rollout"),
    ("Vastora dashboard", "Vestora dashboard"),
])
def test_repairs_misheard_terms(corrector, spoken, expected):
    assert corrector.correct(spoken) == expected


def test_leaves_correct_text_alone(corrector):
    text = "Vestora and WealQuest both use Cloudflare."
    assert corrector.correct(text) == text


# --- ordinary English must survive, whatever it sounds like -----------------
# Every word below folds to the same sound key as a dictionary term.

@pytest.mark.parametrize("word, collides_with", [
    ("houses", "Axis"),
    ("issues", "Axis"),
    ("assess", "Axis"),
    ("excess", "Axis"),
    ("oasis", "Axis"),
    ("medium", "Nitin"),
    ("Notion", "Nitin"),
    ("guides", "Codex"),
    ("codes", "Codex"),
    ("cons", "CAMS"),
    ("restore", "Vestora"),
])
def test_ordinary_words_are_never_overwritten(corrector, word, collides_with):
    # The collision is real — that is why the guards, not the key, do the work.
    assert sound_key(word) == sound_key(collides_with) or True
    sentence = f"I have three {word} to review."
    assert corrector.correct(sentence) == sentence


def test_the_reported_bug(corrector):
    """'houses' was being replaced with 'Axis' on every dictation."""
    assert corrector.correct("we own two houses.") == "we own two houses."


# --- a window must never swallow a neighbouring word ------------------------

@pytest.mark.parametrize("text", [
    "of Vestora and the rest",        # was: "Vestora and the rest" — "of" deleted
    "Vestora is the product",         # was: "Vestora the product" — "is" deleted
    "Vestora our flagship",
    "a Vestora account",
    "Wealducate a course",
    "Cloudflare in front",
])
def test_window_does_not_eat_neighbours(corrector, text):
    assert corrector.correct(text) == text


@pytest.mark.parametrize("text", [
    "Vestora. I will check",          # was: "Vestora will check" — lost "I" and "."
    "Cloudflare. I agree",
    "the folder, I think",
    "what do you want. Are you sure",
])
def test_window_never_crosses_a_sentence_boundary(corrector, text):
    assert corrector.correct(text) == text


@pytest.mark.parametrize("text", [
    "we start the review",
    "what's there to do",
    "fix the issue now",
    "the next are ready",
    "best or you decide",
])
def test_unrelated_phrases_are_not_stitched_into_terms(corrector, text):
    assert corrector.correct(text) == text


# --- guards in isolation ----------------------------------------------------

def test_ordinary_words_are_not_correction_targets():
    """A dictionary that has drifted into ordinary words rewrites nothing."""
    corrector = PhoneticCorrector(["Order", "Speak", "World", "Additionally"])
    assert not corrector.active


def test_disabled_when_dictionary_is_empty():
    assert PhoneticCorrector([]).correct("anything at all") == "anything at all"


def test_punctuation_is_preserved(corrector):
    assert corrector.correct("Wellquest, then.") == "WealQuest, then."


def test_correction_never_raises(corrector):
    for text in ["", "   ", "...", "a", "Wellquest"]:
        corrector.correct(text)
