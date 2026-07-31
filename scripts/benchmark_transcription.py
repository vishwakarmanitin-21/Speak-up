"""Compare speech-to-text engines on YOUR voice and YOUR vocabulary.

A newer model is a hypothesis, not an improvement. OpenAI's `gpt-transcribe`
scored WORSE than the model it supersedes on this project's own terms until the
dictionary was wired in, so "is it the latest?" is the wrong question — "does it
get MY words right?" is the right one. This script makes that cheap to answer.

What it measures, per engine:
  * term recall — how many of your dictionary terms survive verbatim. This is
    what you actually notice, since a mangled name is what you have to retype.
  * word error rate (WER) — overall accuracy against the reference sentence.

Usage
-----
  # generate a reference clip with TTS (no microphone needed), then compare
  python scripts/benchmark_transcription.py --generate

  # use a real recording of your own voice (best signal — do this before
  # switching models for real)
  python scripts/benchmark_transcription.py --audio my_voice.wav

  # narrow the field
  python scripts/benchmark_transcription.py --generate --engines nova-3,nova-2

Reads OPENAI_API_KEY / DEEPGRAM_API_KEY the same way the app does.
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from src.config import Config  # noqa: E402

# A passage dense with the kind of terms that actually break: product names,
# financial acronyms, Indian names, and a spoken number.
REFERENCE = (
    "Vestora and WealQuest are the two products. "
    "Please send the KFinTech and CAMS statements to Prisha. "
    "The SEBI and AMFI guidelines apply to every ICICI and HDFC folio. "
    "Nitin deployed the build on Vercel and Cloudflare. "
    "NSDL confirmed the transfer at nine thirty."
)
EXPECTED_TERMS = [
    "Vestora", "WealQuest", "KFinTech", "CAMS", "Prisha", "SEBI", "AMFI",
    "ICICI", "HDFC", "Nitin", "Vercel", "Cloudflare", "NSDL",
]

_DG_URL = "https://api.deepgram.com/v1/listen"
_OPENAI_ENGINES = ["gpt-4o-transcribe", "gpt-transcribe", "gpt-4o-mini-transcribe"]
_DG_ENGINES = ["nova-2", "nova-3"]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _norm(text: str) -> list[str]:
    keep = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in text)
    return keep.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER: (substitutions + insertions + deletions) / reference words."""
    r, h = _norm(reference), _norm(hypothesis)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def term_hits(text: str, terms: list[str]) -> tuple[int, list[str]]:
    """How many expected terms appear verbatim (case-insensitive)."""
    low = text.lower()
    missed = [t for t in terms if t.lower() not in low]
    return len(terms) - len(missed), missed


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #

def transcribe_openai(model: str, audio: bytes, vocabulary: list[str]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=Config().openai_api_key)
    buf = io.BytesIO(audio)
    buf.name = "audio.wav"
    kwargs: dict = {"model": model, "file": buf, "response_format": "text"}
    if vocabulary:
        kwargs["prompt"] = "Terms: " + ", ".join(vocabulary)
        # gpt-transcribe adds a keywords parameter aimed at literal names.
        if model.startswith("gpt-transcribe"):
            kwargs["extra_body"] = {"keywords": vocabulary[:100]}
    resp = client.audio.transcriptions.create(**kwargs)
    return resp if isinstance(resp, str) else getattr(resp, "text", "")


def transcribe_deepgram(model: str, audio: bytes, vocabulary: list[str]) -> str:
    """Deepgram REST. nova-3 replaced weighted `keywords` with plain `keyterm`."""
    key = Config().deepgram_api_key
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY not set")
    params: list[tuple[str, str]] = [
        ("model", model), ("smart_format", "true"), ("punctuate", "true"),
        ("language", "en"),
    ]
    for term in vocabulary[:100]:
        if model.startswith("nova-3"):
            params.append(("keyterm", term))          # no weights on nova-3
        else:
            params.append(("keywords", f"{term}:2"))
    url = _DG_URL + "?" + urllib.parse.urlencode(params)
    r = httpx.post(
        url, content=audio, timeout=120.0,
        headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
    )
    r.raise_for_status()
    data = r.json()
    return data["results"]["channels"][0]["alternatives"][0]["transcript"]


def record_microphone(seconds: int, rate: int = 16000) -> bytes:
    """Record from the default mic and return WAV bytes.

    Synthesised speech is clean and accent-neutral; only a real recording tells
    you how the engines handle YOUR voice, mic and room.
    """
    import io as _io

    import numpy as np
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write

    print("\n" + "=" * 70)
    print("Read this aloud, at your normal dictation pace:\n")
    print(f"  {REFERENCE}\n")
    print(f"Recording for {seconds}s — starting in 3 seconds...")
    print("=" * 70)
    sd.sleep(3000)
    print(">>> SPEAK NOW <<<")
    frames = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    print("Recorded.\n")
    buf = _io.BytesIO()
    wav_write(buf, rate, (np.clip(frames[:, 0], -1, 1) * 32767).astype("<i2"))
    return buf.getvalue()


def generate_speech(text: str, voice: str = "alloy") -> bytes:
    from openai import OpenAI

    client = OpenAI(api_key=Config().openai_api_key)
    return client.audio.speech.create(
        model="tts-1", voice=voice, input=text, response_format="wav"
    ).content


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", help="WAV file of you reading the reference text")
    ap.add_argument("--generate", action="store_true", help="synthesise the clip with TTS")
    ap.add_argument("--record", nargs="?", const=25, type=int, metavar="SECONDS",
                    help="record YOUR voice from the mic (default 25s) and use that "
                         "— the only result that really counts")
    ap.add_argument("--engines", help="comma-separated subset to run")
    ap.add_argument("--no-vocab", action="store_true", help="run without dictionary biasing")
    ap.add_argument("--trials", type=int, default=1,
                    help="repeat N times and average — single runs are noisy "
                         "(engines vary run to run), so use 3+ before deciding")
    args = ap.parse_args()

    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if args.record:
        clips = [record_microphone(args.record)]
        source = f"your voice ({args.record}s from the microphone)"
        out = Path(__file__).resolve().parent.parent / "benchmark_voice.wav"
        out.write_bytes(clips[0])
        print(f"Saved to {out} — rerun with --audio to re-test without re-recording.\n")
        args.trials = 1
    elif args.audio:
        clips = [Path(args.audio).read_bytes()]
        source = args.audio
        if args.trials > 1:
            print("(--trials ignored: a fixed recording gives the same audio each run)")
            args.trials = 1
    elif args.generate:
        print(f"Synthesising {args.trials} reference clip(s) (tts-1)...")
        # Vary the voice per trial so the result is not tuned to one speaker.
        clips = [generate_speech(REFERENCE, voices[i % len(voices)])
                 for i in range(args.trials)]
        source = f"generated speech (tts-1, {args.trials} voice(s))"
    else:
        ap.error("pass --record, --audio <file.wav>, or --generate")

    vocabulary = [] if args.no_vocab else Config().custom_vocabulary
    engines = _DG_ENGINES + _OPENAI_ENGINES
    if args.engines:
        wanted = {e.strip() for e in args.engines.split(",")}
        engines = [e for e in engines if e in wanted]

    print(f"\nSource      : {source}")
    print(f"Dictionary  : {len(vocabulary)} term(s)" + (" (disabled)" if args.no_vocab else ""))
    print(f"Reference   : {REFERENCE}\n")

    results: dict[str, dict] = {}
    for engine in engines:
        hits_all, wer_all, missed_all, last, failed = [], [], [], "", None
        for clip in clips:
            try:
                if engine in _DG_ENGINES:
                    text = transcribe_deepgram(engine, clip, vocabulary)
                else:
                    text = transcribe_openai(engine, clip, vocabulary)
                hits, missed = term_hits(text, EXPECTED_TERMS)
                hits_all.append(hits)
                wer_all.append(word_error_rate(REFERENCE, text))
                missed_all.extend(missed)
                last = text.strip()
            except Exception as e:
                failed = str(e)[:110]
        results[engine] = {
            "hits": sum(hits_all) / len(hits_all) if hits_all else -1,
            "wer": sum(wer_all) / len(wer_all) if wer_all else 1.0,
            # Terms missed in EVERY trial are the reliable failures.
            "always_missed": sorted({m for m in missed_all
                                     if missed_all.count(m) == len(clips)}),
            "text": last or f"FAILED: {failed}",
        }

    order = sorted(results.items(), key=lambda kv: (-kv[1]["hits"], kv[1]["wer"]))
    n = len(EXPECTED_TERMS)
    print(f"{'engine':<24}{'terms':>10}{'WER':>9}   always missed")
    print("-" * 78)
    for engine, r in order:
        got = "n/a" if r["hits"] < 0 else f"{r['hits']:.1f}/{n}"
        print(f"{engine:<24}{got:>10}{r['wer']:>8.1%}   "
              f"{', '.join(r['always_missed']) if r['always_missed'] else '-'}")

    print("\nLast transcript per engine")
    print("-" * 78)
    for engine, r in order:
        print(f"  {engine:<22} {r['text']}")

    best_engine, best = order[0]
    if best["hits"] >= 0:
        print(f"\nBest on your terms: {best_engine} ({best['hits']:.1f}/{n} terms, "
              f"{best['wer']:.1%} WER, averaged over {len(clips)} trial(s))")
    print("\nNOTE: synthesised speech is only indicative — rerun with --audio using a "
          "recording of your own voice before switching models for real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
