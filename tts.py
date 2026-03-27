"""
tts.py — Text-to-Speech module for HearMeAI.

Supports two backends:
  - gTTS     (Google Text-to-Speech, requires internet, produces MP3)
  - pyttsx3  (offline engine, requires espeak / sapi5 / nsss on the host OS)

The function `synthesize_speech` accepts a *backend* parameter:
  "auto"    — tries pyttsx3 first, then falls back to gTTS
  "gtts"    — forces gTTS (online)
  "pyttsx3" — forces pyttsx3 (offline)
"""

import io
import os
import tempfile
from typing import Literal


# ──────────────────────────────────────────────
# Backend: pyttsx3 (offline)
# ──────────────────────────────────────────────

def _tts_pyttsx3(text: str, rate: int = 150, volume: float = 1.0) -> bytes:
    """
    Synthesise speech using pyttsx3 (fully offline).

    Requires a system TTS engine: espeak-ng on Linux, SAPI5 on Windows,
    NSSpeechSynthesizer on macOS.

    Returns raw WAV bytes.
    """
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name

    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return audio_bytes


# ──────────────────────────────────────────────
# Backend: gTTS (online)
# ──────────────────────────────────────────────

def _tts_gtts(text: str, lang: str = "en", slow: bool = False) -> bytes:
    """
    Synthesise speech using gTTS (Google Text-to-Speech — requires internet).

    Returns MP3 bytes.
    """
    from gtts import gTTS

    tts = gTTS(text=text, lang=lang, slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def synthesize_speech(
    text: str,
    backend: Literal["auto", "gtts", "pyttsx3"] = "auto",
    lang: str = "en",
    rate: int = 150,
) -> tuple[bytes, str]:
    """
    Convert text to speech.

    Parameters
    ----------
    text    : The text to synthesise.
    backend : Which TTS engine to use.
              "auto"    — tries pyttsx3 first, falls back to gTTS.
              "gtts"    — use Google TTS (online, MP3 output).
              "pyttsx3" — use pyttsx3 (offline, WAV output).
    lang    : BCP-47 language code for gTTS (e.g. "en", "fr", "es").
    rate    : Words-per-minute rate for pyttsx3.

    Returns
    -------
    (audio_bytes, file_extension)
        e.g. (b"...", ".wav")  or  (b"...", ".mp3")
    """
    if not text.strip():
        raise ValueError("Input text is empty.")

    if backend == "pyttsx3":
        return _tts_pyttsx3(text, rate=rate), ".wav"

    if backend == "gtts":
        return _tts_gtts(text, lang=lang), ".mp3"

    # "auto": try pyttsx3, fall back to gTTS
    try:
        return _tts_pyttsx3(text, rate=rate), ".wav"
    except Exception:
        return _tts_gtts(text, lang=lang), ".mp3"
