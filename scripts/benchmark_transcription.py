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
# Realtime-only: 404s on the file endpoint, so it goes over a WebSocket session.
_REALTIME_ENGINES = ["gpt-live-transcribe"]


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


def _beep(freq: int = 880, ms: int = 220, rate: int = 44100) -> None:
    """Short tone so the cue is AUDIBLE — the terminal may not be in view."""
    try:
        import numpy as np
        import sounddevice as sd

        t = np.linspace(0, ms / 1000, int(rate * ms / 1000), endpoint=False)
        tone = (0.25 * np.sin(2 * np.pi * freq * t)).astype("float32")
        sd.play(tone, rate)
        sd.wait()
    except Exception:
        pass


def transcribe_openai_realtime(model: str, audio: bytes, vocabulary: list[str]) -> str:
    """gpt-live-transcribe only exists over a realtime session — it 404s on the
    file endpoint — so feed the clip through a WebSocket the way the app does.

    This also exercises the app's own session config (languages array vs
    language string, keyword biasing) against the live API.
    """
    import asyncio
    import base64
    import json as _json

    import numpy as np
    from scipy.io.wavfile import read as wav_read

    from src.transcription.realtime_client import build_transcription_config

    rate, samples = wav_read(io.BytesIO(audio))
    if samples.ndim > 1:
        samples = samples[:, 0]
    if samples.dtype != np.int16:                       # normalise to PCM16
        samples = (np.clip(samples.astype("float32"), -1, 1) * 32767).astype("<i2")
    # The Realtime API rejects rates below 24 kHz ('integer_below_min_value'),
    # so upsample anything recorded lower rather than failing the comparison.
    if rate < 24000:
        from scipy.signal import resample_poly
        samples = resample_poly(samples.astype("float32"), 24000, rate)
        samples = np.clip(samples, -32768, 32767).astype("<i2")
        rate = 24000
    pcm = samples.astype("<i2").tobytes()

    async def _run() -> str:
        import websockets

        url = "wss://api.openai.com/v1/realtime?intent=transcription"
        headers = {"Authorization": f"Bearer {Config().openai_api_key}"}
        try:
            ws = await websockets.connect(url, additional_headers=headers, max_size=None)
        except TypeError:
            ws = await websockets.connect(url, extra_headers=headers, max_size=None)

        finals: list[str] = []
        async with ws:
            cfg = build_transcription_config(model, vocabulary)
            await ws.send(_json.dumps({
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {"input": {
                        "format": {"type": "audio/pcm", "rate": int(rate)},
                        "transcription": cfg,
                        # No server VAD: we push a whole file and commit it.
                        "turn_detection": None,
                    }},
                },
            }))
            chunk = int(rate * 2 * 0.2)                  # ~200ms of PCM16
            for i in range(0, len(pcm), chunk):
                await ws.send(_json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[i:i + chunk]).decode(),
                }))
            await ws.send(_json.dumps({"type": "input_audio_buffer.commit"}))

            loop = asyncio.get_running_loop()
            deadline = loop.time() + 60.0
            while loop.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    if finals:
                        break
                    continue
                except Exception:
                    break
                msg = _json.loads(raw)
                mtype = msg.get("type", "")
                if mtype.endswith("input_audio_transcription.completed"):
                    t = (msg.get("transcript") or "").strip()
                    if t:
                        finals.append(t)
                elif mtype == "error":
                    raise RuntimeError(str(msg.get("error"))[:160])
        return " ".join(finals)

    return asyncio.run(_run())


def record_microphone(seconds: int, rate: int = 24000, lead_in: int = 6) -> bytes:
    """Record from the default mic and return WAV bytes.

    Synthesised speech is clean and accent-neutral; only a real recording tells
    you how the engines handle YOUR voice, mic and room. Start/stop are signalled
    with beeps so you do not have to watch the terminal to get the timing right.
    """
    import io as _io

    import numpy as np
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write

    print("\n" + "=" * 70)
    print("Read this aloud, at your normal dictation pace:\n")
    print(f"  {REFERENCE}\n")
    print(f"HIGH beep = start speaking. {seconds}s later, LOW beep = done.")
    print(f"Starting in {lead_in} seconds...")
    print("=" * 70, flush=True)
    sd.sleep(lead_in * 1000)
    _beep(880, 250)                      # high tone: go
    print(">>> SPEAK NOW <<<", flush=True)
    frames = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    _beep(440, 350)                      # low tone: stop
    print("Recorded.", flush=True)

    peak = float(np.max(np.abs(frames))) if frames.size else 0.0
    print(f"Signal peak {peak:.3f}", flush=True)
    buf = _io.BytesIO()
    wav_write(buf, rate, (np.clip(frames[:, 0], -1, 1) * 32767).astype("<i2"))
    return buf.getvalue(), peak


def generate_speech(text: str, voice: str = "alloy") -> bytes:
    from openai import OpenAI

    client = OpenAI(api_key=Config().openai_api_key)
    return client.audio.speech.create(
        model="tts-1", voice=voice, input=text, response_format="wav"
    ).content


# --------------------------------------------------------------------------- #

def main() -> int:
    # Transcripts can contain characters the Windows console codepage cannot
    # encode; without this the whole run dies at the final print.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", help="WAV file of you reading the reference text")
    ap.add_argument("--generate", action="store_true", help="synthesise the clip with TTS")
    ap.add_argument("--record", nargs="?", const=25, type=int, metavar="SECONDS",
                    help="record YOUR voice from the mic (default 25s) and use that "
                         "— the only result that really counts")
    ap.add_argument("--engines", help="comma-separated subset to run")
    ap.add_argument("--no-vocab", action="store_true", help="run without dictionary biasing")
    ap.add_argument("--lead-in", type=int, default=6, metavar="SECONDS",
                    help="pause before recording starts (default 6)")
    ap.add_argument("--trials", type=int, default=1,
                    help="repeat N times and average — single runs are noisy "
                         "(engines vary run to run), so use 3+ before deciding")
    args = ap.parse_args()

    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if args.record:
        clip, peak = record_microphone(args.record, lead_in=args.lead_in)
        out = Path(__file__).resolve().parent.parent / "benchmark_voice.wav"
        out.write_bytes(clip)
        print(f"Saved to {out}\n")
        if peak < 0.03:
            # Abort rather than burn API calls scoring an empty room — and the
            # scores would look like real engine failures, which is worse.
            print("=" * 70)
            print(f"ABORTED: that clip is essentially silent (peak {peak:.3f}).")
            print("The mic may be muted/wrong device, or the timing was missed.")
            print("Re-run and start reading at the HIGH beep:")
            print("  .venv\\Scripts\\python.exe scripts\\benchmark_transcription.py "
                  "--record 32 --lead-in 15")
            print("=" * 70)
            return 1
        clips = [clip]
        source = f"your voice ({args.record}s from the microphone)"
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
    if vocabulary:
        # Match what the app actually sends: ordinary words are stripped out.
        # Boosting them measurably HURTS — with "I'm" left in the dictionary,
        # nova-2 transcribed "CAMS" as "I'm".
        from src.services.phonetic import _is_ordinary_word
        dropped = [t for t in vocabulary if _is_ordinary_word(t)]
        vocabulary = [t for t in vocabulary if not _is_ordinary_word(t)]
        if dropped:
            print(f"(filtered {len(dropped)} ordinary word(s) from the dictionary, "
                  f"as the app does: {', '.join(dropped)})")
    engines = _DG_ENGINES + _OPENAI_ENGINES + _REALTIME_ENGINES
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
                elif engine in _REALTIME_ENGINES:
                    text = transcribe_openai_realtime(engine, clip, vocabulary)
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
