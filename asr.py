"""
asr.py — Automatic Speech Recognition using OpenAI Whisper.

Whisper runs entirely locally; no API key is required.
The 'base' model is used by default for a good balance between speed and accuracy
on CPU.  Users can switch to 'tiny' (faster) or 'small'/'medium' (more accurate)
via the Streamlit sidebar.
"""

import warnings
import torch
import whisper

warnings.filterwarnings("ignore")

# Model instance cached in module-level variable so it is loaded only once per session
_model = None
_loaded_model_size = None


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_model(model_size: str = "base"):
    """
    Load (or reuse a cached) Whisper model.

    Parameters
    ----------
    model_size : str
        One of "tiny", "base", "small", "medium", "large".
        "base" is recommended for CPU usage.

    Returns
    -------
    model : whisper.Whisper
    device : str  ("cuda" or "cpu")
    """
    global _model, _loaded_model_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Re-use cached model when the size hasn't changed
    if _model is not None and _loaded_model_size == model_size:
        return _model, device

    _model = whisper.load_model(model_size, device=device)
    _loaded_model_size = model_size
    return _model, device


# ──────────────────────────────────────────────
# Transcription
# ──────────────────────────────────────────────

def transcribe_audio(model, audio_path: str, language: str = None) -> dict:
    """
    Transcribe an audio file and return timestamped segments.

    Parameters
    ----------
    model : whisper.Whisper
        Pre-loaded Whisper model.
    audio_path : str
        Absolute path to a WAV / MP3 audio file.
    language : str | None
        BCP-47 language code (e.g. "en", "fr").  Pass None to auto-detect.

    Returns
    -------
    dict with keys:
        "segments"  : list of dicts {start, end, text}
        "language"  : detected/specified language code
        "full_text" : complete concatenated transcript
    """
    transcribe_kwargs = {
        "word_timestamps": True,
        "verbose": False,
        "fp16": False,  # keeps CPU inference stable
    }
    if language:
        transcribe_kwargs["language"] = language

    try:
        result = model.transcribe(audio_path, **transcribe_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Transcription failed: {exc}") from exc

    segments = []
    for seg in result.get("segments", []):
        segments.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"],
            }
        )

    return {
        "segments": segments,
        "language": result.get("language", "unknown"),
        "full_text": result.get("text", ""),
    }
