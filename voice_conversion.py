"""
voice_conversion.py — Lightweight voice conversion for HearMeAI.

Approach (CPU-friendly, no GPU required):
  - If a *target* reference audio is provided:
      estimate the mean F0 of both source and target, then pitch-shift
      the source so its mean F0 matches the target.
  - If no target is provided:
      apply a manual pitch shift (semitones) specified by the user.

An optional speed factor (time-stretch) can be applied on top of pitch
shifting to further differentiate speakers.
"""

import io
import numpy as np
import librosa
import soundfile as sf

from utils import estimate_mean_f0


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def convert_voice(
    source_audio_path: str,
    target_audio_path: str | None = None,
    pitch_shift_semitones: float = 0.0,
    speed_factor: float = 1.0,
) -> bytes:
    """
    Transform a source voice towards a target voice (or apply manual adjustments).

    Parameters
    ----------
    source_audio_path     : Path to the audio file to be converted.
    target_audio_path     : (Optional) Path to a reference audio file whose
                            pitch profile the source should be shifted towards.
                            When provided, overrides *pitch_shift_semitones*.
    pitch_shift_semitones : Manual pitch shift in semitones (−24 … +24).
                            Used only when *target_audio_path* is None.
    speed_factor          : Playback speed multiplier (0.25 – 4.0).
                            Values > 1 speed up, < 1 slow down.

    Returns
    -------
    WAV bytes of the converted audio.
    """
    # Load source audio (22 050 Hz, mono — good balance of quality and speed)
    source, sr = librosa.load(source_audio_path, sr=22050, mono=True)

    # ── Determine pitch shift ─────────────────────────────────────────────
    if target_audio_path is not None:
        target, _ = librosa.load(target_audio_path, sr=22050, mono=True)
        src_f0 = estimate_mean_f0(source, sr)
        tgt_f0 = estimate_mean_f0(target, sr)
        if src_f0 > 0 and tgt_f0 > 0:
            semitones = 12.0 * np.log2(tgt_f0 / src_f0)
            semitones = float(np.clip(semitones, -24.0, 24.0))
        else:
            semitones = 0.0
    else:
        semitones = float(np.clip(pitch_shift_semitones, -24.0, 24.0))

    # ── Apply pitch shift ─────────────────────────────────────────────────
    if abs(semitones) > 0.01:
        converted = librosa.effects.pitch_shift(source, sr=sr, n_steps=semitones)
    else:
        converted = source.copy()

    # ── Apply time stretch (speed change) ─────────────────────────────────
    speed_factor = float(np.clip(speed_factor, 0.25, 4.0))
    if abs(speed_factor - 1.0) > 0.01:
        converted = librosa.effects.time_stretch(converted, rate=speed_factor)

    # ── Export to WAV bytes ───────────────────────────────────────────────
    buf = io.BytesIO()
    sf.write(buf, converted, sr, format="WAV")
    buf.seek(0)
    return buf.read()
