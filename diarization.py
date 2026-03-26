"""
diarization.py — Lightweight speaker diarization using MFCC features + K-Means clustering.

This module does NOT require any external API tokens.  The algorithm is:

1. Slide a short window across the audio and extract MFCC + delta features.
2. Normalise features (StandardScaler).
3. Cluster with K-Means → each cluster represents one speaker.
4. (Optional) Auto-estimate the number of speakers using the silhouette score.
5. Post-process to merge consecutive windows with the same speaker label.
6. Align the resulting speaker segments with the Whisper transcript.
"""

import warnings
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────

def extract_features(
    audio: np.ndarray,
    sr: int,
    window_duration: float = 1.5,
    hop_duration: float = 0.5,
    n_mfcc: int = 20,
) -> tuple[np.ndarray, list[float]]:
    """
    Slide a window over the audio and extract a feature vector per window.

    Feature vector = [ mean(MFCC), std(MFCC), mean(ΔMFCC) ] → length 3×n_mfcc.

    Returns
    -------
    features   : np.ndarray of shape (n_windows, 3*n_mfcc)
    timestamps : list[float] — start time (s) of each window
    """
    window_len = int(window_duration * sr)
    hop_len = int(hop_duration * sr)

    features: list[np.ndarray] = []
    timestamps: list[float] = []

    for start in range(0, len(audio) - window_len + 1, hop_len):
        segment = audio[start : start + window_len]

        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)

        feat = np.concatenate(
            [np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(mfcc_delta, axis=1)]
        )
        features.append(feat)
        timestamps.append(start / sr)

    return np.array(features), timestamps


# ──────────────────────────────────────────────
# Number-of-speakers estimation
# ──────────────────────────────────────────────

def estimate_n_speakers(features: np.ndarray, max_speakers: int = 6) -> int:
    """
    Choose the number of clusters (2 … max_speakers) that maximises the
    average silhouette score.  Falls back to 2 if there are too few windows.
    """
    if len(features) < max_speakers * 2:
        return 2

    best_score = -1.0
    best_n = 2

    for n in range(2, min(max_speakers + 1, len(features))):
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = km.fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_n = n

    return best_n


# ──────────────────────────────────────────────
# Diarization pipeline
# ──────────────────────────────────────────────

def perform_diarization(
    audio: np.ndarray,
    sr: int,
    n_speakers: int | None = None,
) -> list[dict]:
    """
    Run the full diarization pipeline on a loaded audio array.

    Parameters
    ----------
    audio      : 1-D float32 numpy array (mono, already resampled)
    sr         : sample rate (typically 16 000 Hz)
    n_speakers : number of expected speakers.  Pass None to auto-detect.

    Returns
    -------
    List of dicts: [{"start": float, "end": float, "speaker": str}, …]
    sorted by start time.
    """
    if len(audio) < sr * 2:
        # Audio shorter than 2 s — assign a single speaker
        return [{"start": 0.0, "end": len(audio) / sr, "speaker": "Speaker 1"}]

    features, timestamps = extract_features(audio, sr)

    if len(features) < 2:
        return [{"start": 0.0, "end": len(audio) / sr, "speaker": "Speaker 1"}]

    # Normalise
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Determine number of speakers
    if n_speakers is None:
        n_speakers = estimate_n_speakers(features_scaled)

    n_speakers = max(1, min(n_speakers, len(features)))

    # Cluster
    km = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
    raw_labels = km.fit_predict(features_scaled)

    # ── Post-processing: smooth with a majority-vote over a 3-window median ──
    labels = _smooth_labels(raw_labels, window=3)

    # ── Build speaker segments by collapsing consecutive identical labels ──
    segments: list[dict] = []
    current_label = labels[0]
    current_start = timestamps[0]
    hop = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.5

    for ts, label in zip(timestamps[1:], labels[1:]):
        if label != current_label:
            segments.append(
                {
                    "start": current_start,
                    "end": ts,
                    "speaker": f"Speaker {int(current_label) + 1}",
                }
            )
            current_label = label
            current_start = ts

    # Append final segment
    segments.append(
        {
            "start": current_start,
            "end": timestamps[-1] + hop,
            "speaker": f"Speaker {int(current_label) + 1}",
        }
    )

    return segments


def _smooth_labels(labels: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply a simple majority-vote smoothing over a sliding window."""
    from scipy.stats import mode as scipy_mode

    smoothed = labels.copy()
    half = window // 2

    for i in range(half, len(labels) - half):
        neighborhood = labels[i - half : i + half + 1]
        result = scipy_mode(neighborhood, keepdims=True)
        smoothed[i] = int(result.mode[0])

    return smoothed


# ──────────────────────────────────────────────
# Merge transcript with diarization
# ──────────────────────────────────────────────

def merge_transcript_diarization(
    transcript_segments: list[dict],
    diarization_segments: list[dict],
) -> list[dict]:
    """
    Assign a speaker label to each Whisper transcript segment by finding
    which diarization segment covers the midpoint of the transcript segment.

    Parameters
    ----------
    transcript_segments  : list of {start, end, text}
    diarization_segments : list of {start, end, speaker}

    Returns
    -------
    list of {start, end, text, speaker}
    """
    merged: list[dict] = []

    for seg in transcript_segments:
        midpoint = (seg["start"] + seg["end"]) / 2.0
        speaker = _find_speaker(midpoint, diarization_segments)
        merged.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": speaker,
            }
        )

    return merged


def _find_speaker(midpoint: float, diarization_segments: list[dict]) -> str:
    """Return the speaker label whose segment contains *midpoint*."""
    # Exact match
    for seg in diarization_segments:
        if seg["start"] <= midpoint <= seg["end"]:
            return seg["speaker"]

    # Nearest segment by midpoint distance (fallback)
    if diarization_segments:
        closest = min(
            diarization_segments,
            key=lambda s: abs((s["start"] + s["end"]) / 2 - midpoint),
        )
        return closest["speaker"]

    return "Speaker 1"
