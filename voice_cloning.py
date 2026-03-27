"""
voice_cloning.py — Lightweight voice cloning for HearMeAI.

Approach (CPU-friendly, no GPU required):
  1. Load a short reference audio clip from the target speaker.
  2. Estimate the speaker's mean fundamental frequency (F0) using librosa.pyin.
  3. Synthesise the requested text with gTTS.
  4. Pitch-shift the synthesised speech so its mean F0 matches the reference.

This is a *basic* approximation — it captures the pitch profile of the
speaker but does not model timbre, accent, or fine-grained prosody.
"""

import io
import numpy as np
import librosa
import soundfile as sf

from utils import estimate_mean_f0


# ──────────────────────────────────────────────
# Helper: gTTS → numpy array
# ──────────────────────────────────────────────

def _gtts_to_array(text: str, lang: str = "en") -> tuple[np.ndarray, int]:
    """
    Synthesise text with gTTS and return the audio as a float32 numpy array.

    Returns (samples, sample_rate) at 22 050 Hz mono.
    """
    from gtts import gTTS
    from pydub import AudioSegment

    tts = gTTS(text=text, lang=lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)

    # Convert MP3 bytes → PCM via pydub
    seg = AudioSegment.from_mp3(buf)
    seg = seg.set_channels(1).set_frame_rate(22050)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(np.int16).max  # normalise to [-1, 1]
    return samples, 22050


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def clone_voice(
    text: str,
    reference_audio_path: str,
    lang: str = "en",
) -> bytes:
    """
    Generate speech that approximates the pitch profile of a reference speaker.

    Parameters
    ----------
    text                 : Text to synthesise.
    reference_audio_path : Path to a short (≥3 s) audio file from the target
                           speaker (WAV, MP3, etc.).
    lang                 : BCP-47 language code passed to gTTS (e.g. "en").

    Returns
    -------
    WAV bytes of the pitch-adjusted synthesised speech.
    """
    # 1. Load reference audio and estimate its mean F0
    ref_audio, ref_sr = librosa.load(reference_audio_path, sr=None, mono=True)
    ref_f0 = estimate_mean_f0(ref_audio, ref_sr)

    # 2. Synthesise TTS speech
    tts_audio, tts_sr = _gtts_to_array(text, lang=lang)
    tts_f0 = estimate_mean_f0(tts_audio, tts_sr)

    # 3. Compute pitch shift required to match reference F0
    if tts_f0 > 0 and ref_f0 > 0:
        semitones = 12.0 * np.log2(ref_f0 / tts_f0)
        # Clamp to avoid severe artefacts
        semitones = float(np.clip(semitones, -12.0, 12.0))
    else:
        semitones = 0.0

    # 4. Pitch-shift the TTS audio towards the reference speaker's pitch
    shifted = librosa.effects.pitch_shift(tts_audio, sr=tts_sr, n_steps=semitones)

    # 5. Export to WAV bytes
    buf = io.BytesIO()
    sf.write(buf, shifted, tts_sr, format="WAV")
    buf.seek(0)
    return buf.read()
