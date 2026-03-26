"""
utils.py — Audio utilities, formatting helpers, and display functions for HearMeAI.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import librosa

# Colour palette for up to 6 distinct speakers
SPEAKER_COLORS = {
    "Speaker 1": "#FF6B6B",  # red
    "Speaker 2": "#4ECDC4",  # teal
    "Speaker 3": "#45B7D1",  # blue
    "Speaker 4": "#96CEB4",  # green
    "Speaker 5": "#FFEAA7",  # yellow
    "Speaker 6": "#DDA0DD",  # plum
}

# ──────────────────────────────────────────────
# File helpers
# ──────────────────────────────────────────────

def save_uploaded_file(uploaded_file) -> str:
    """Persist a Streamlit UploadedFile to a temp path and return that path."""
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def save_audio_bytes(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Save raw audio bytes (e.g. from microphone recorder) to a temp file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        return tmp.name


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio format supported by pydub/ffmpeg to a 16 kHz mono WAV.
    Returns the path of the converted file.
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        out_path = tmp.name
    audio.export(out_path, format="wav")
    return out_path


def load_audio(file_path: str, sr: int = 16000):
    """
    Load an audio file and resample to *sr* Hz (mono).
    Returns (audio_array: np.ndarray, sample_rate: int).
    """
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return audio, sample_rate


def cleanup_temp_file(path: str) -> None:
    """Remove a temporary file, ignoring errors if it no longer exists."""
    try:
        os.remove(path)
    except OSError:
        pass


# ──────────────────────────────────────────────
# Timestamp / text formatting
# ──────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to a human-readable MM:SS.s string."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def get_speaker_color(speaker_label: str) -> str:
    """Return a hex colour string for a given speaker label."""
    return SPEAKER_COLORS.get(speaker_label, "#AAAAAA")


def create_transcript_text(segments: list) -> str:
    """
    Build a plain-text transcript from a list of segment dicts.
    Each dict must have keys: start, end, speaker, text.
    """
    lines = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        speaker = seg.get("speaker", "Unknown")
        text = seg["text"].strip()
        lines.append(f"[{start} → {end}]  {speaker}: {text}")
    return "\n".join(lines)


def render_colored_transcript(segments: list) -> str:
    """
    Build an HTML string that renders each utterance in the speaker's colour.
    Safe to pass directly to st.markdown(..., unsafe_allow_html=True).
    """
    html_parts = ['<div style="font-family: monospace; line-height: 1.8; font-size: 14px;">']

    prev_speaker = None
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        color = get_speaker_color(speaker)
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()

        # Add a small vertical gap between speaker changes
        gap = '<br/>' if speaker != prev_speaker and prev_speaker is not None else ""

        html_parts.append(
            f'{gap}'
            f'<span style="color:{color}; font-weight:bold;">[{start} → {end}] {speaker}:</span> '
            f'<span style="color:#E8E8E8;">{text}</span><br/>'
        )
        prev_speaker = speaker

    html_parts.append("</div>")
    return "".join(html_parts)


def speaker_stats(segments: list) -> dict:
    """
    Compute per-speaker total speaking time (seconds) from the merged segments.
    Returns a dict like {"Speaker 1": 12.4, "Speaker 2": 8.7}.
    """
    totals: dict[str, float] = {}
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        duration = seg["end"] - seg["start"]
        totals[speaker] = totals.get(speaker, 0.0) + duration
    return totals
