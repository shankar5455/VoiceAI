# 🎙️ HearMeAI — Speech AI: Transcription, TTS, Voice Cloning & Conversion

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Whisper](https://img.shields.io/badge/ASR-OpenAI%20Whisper-00C853.svg)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A hackathon-ready, end-to-end **Speech AI** web application featuring transcription,
text-to-speech, voice cloning, voice conversion, and a training pipeline — all running
**locally** on CPU without any API keys.

---

## 📌 Project Overview

HearMeAI is a five-in-one Speech AI platform built on a Streamlit UI:

| Module | Technology |
|---|---|
| Automatic Speech Recognition (ASR) | OpenAI Whisper (pretrained, local) |
| Speaker Diarization | MFCC + K-Means clustering (scikit-learn) |
| Text-to-Speech (TTS) | gTTS (online) / pyttsx3 (offline) |
| Voice Cloning | Pitch-shift synthesis via librosa |
| Voice Conversion | F0-based pitch transfer + time-stretch |
| Training Pipeline | Dataset validation + config generation |

---

## ✨ Features

### 🎙️ Transcription
- **Audio input** — upload `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac` *or* record from microphone
- **Accurate ASR** — OpenAI Whisper (tiny / base / small / medium model selectable)
- **Speaker diarization** — identifies 2–6 speakers; auto-detects count via silhouette score
- **Colour-coded transcript** — each speaker rendered in a distinct colour
- **Timestamps** — every utterance shows `[MM:SS.ss → MM:SS.ss]`
- **Speaker statistics** — speaking time and percentage per speaker
- **Download transcript** — export plain-text `.txt` file
- **Multi-language** — auto-detect or specify English, French, Spanish, German, Hindi, Arabic

### 🔊 Text-to-Speech
- Convert any text to speech using **gTTS** (online, MP3) or **pyttsx3** (offline, WAV)
- Auto-fallback mode: tries pyttsx3 first, then falls back to gTTS
- Adjustable speech rate and language selection
- In-browser audio playback and download

### 🎭 Voice Cloning
- Upload a short reference audio clip (≥ 3 s) from a target speaker
- Estimates the speaker's mean fundamental frequency (F0) using librosa
- Synthesises requested text with gTTS and pitch-shifts it to match the reference
- Outputs WAV audio approximating the target speaker's pitch profile

### 🔄 Voice Conversion
- Transform a source audio file towards a target speaker's voice
- Automatic F0-based pitch shift when a target reference is provided
- Manual pitch shift (−24 … +24 semitones) when no reference is available
- Optional time-stretch / speed control (0.25× – 4.0×)
- Outputs converted WAV audio

### 🧠 Advanced Training
- Upload a labelled dataset ZIP archive for validation
- Validates audio file count, transcript files, and structure
- Generates a portable **JSON training configuration** for TTS or ASR fine-tuning
- Simulated training progress for UI demonstration purposes
- **CPU-optimised** — runs on any modern laptop; no GPU required

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit 1.28+ |
| ASR model | OpenAI Whisper (via `openai-whisper`) |
| Deep learning | PyTorch (CPU inference) |
| Audio features | librosa |
| Clustering | scikit-learn KMeans |
| Audio I/O | soundfile, pydub |
| Text-to-Speech | gTTS, pyttsx3 |
| Microphone | audio-recorder-streamlit |
| Data | pandas |

---

## 📁 Project Structure

```
HearMeAI/
├── app.py              # Main Streamlit application (5 tabs)
├── asr.py              # Speech-to-text with OpenAI Whisper
├── diarization.py      # Speaker diarization (MFCC + K-Means)
├── tts.py              # Text-to-Speech (gTTS + pyttsx3 backends)
├── voice_cloning.py    # Voice cloning via pitch-shift synthesis
├── voice_conversion.py # Voice conversion via F0 transfer + time-stretch
├── training.py         # Dataset validation and training config generator
├── utils.py            # Audio helpers, formatting, rendering
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.10** or higher
- **ffmpeg** (required by Whisper and pydub for MP3 support)

Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### Python dependencies

```bash
# Clone the repository
git clone https://github.com/shankar5455/HearMeAI.git
cd HearMeAI

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The first run will download the selected Whisper model weights
> (~75 MB for `base`).  Subsequent runs reuse the cached weights.

---

## 🚀 How to Run

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## 🎬 Usage

### 🎙️ Transcription Tab
1. **Upload** a `.wav` or `.mp3` file using the *Upload Audio File* tab, **or**
   click the microphone button in the *Record from Microphone* tab.
2. *(Optional)* Adjust the Whisper model size, number of speakers, and language
   in the left sidebar.
3. Click **🚀 Process Audio**.
4. View the colour-coded, timestamped transcript and speaker statistics.
5. Click **⬇️ Download Transcript (.txt)** to save the results.

### 🔊 Text-to-Speech Tab
1. Type or paste any text into the input box.
2. Choose a TTS backend: **Auto**, **gTTS** (online), or **pyttsx3** (offline).
3. Select a language and speech rate, then click **Generate Speech**.
4. Play back and download the generated audio file.

### 🎭 Voice Cloning Tab
1. Upload a reference audio clip from the target speaker (≥ 3 s).
2. Enter the text you want synthesised in the target speaker's voice.
3. Click **Clone Voice** to generate pitch-adjusted speech.
4. Play back and download the output WAV.

### 🔄 Voice Conversion Tab
1. Upload the source audio file you want to transform.
2. *(Optional)* Upload a target reference audio to guide pitch transfer.
3. Adjust pitch shift (semitones) and speed factor as needed.
4. Click **Convert Voice** and download the result.

### 🧠 Advanced Training Tab
1. Upload a ZIP dataset archive (audio files + transcript `.txt`).
2. Review the validation report (file counts and warnings).
3. Configure model type, epochs, batch size, and sample rate.
4. Download the generated **JSON training config** to use on a GPU machine.

---

## 📸 Sample Transcript Output

```
[00:00.00 → 00:03.50]  Speaker 1: Hello, welcome to HearMeAI.
[00:03.60 → 00:07.20]  Speaker 2: Thanks! This is really impressive.
[00:07.40 → 00:11.10]  Speaker 1: It uses Whisper for transcription and clustering for speakers.
[00:11.20 → 00:14.80]  Speaker 2: Can it handle background noise?
```

---

## 🔬 How It Works

### Speech-to-Text (ASR)

1. Audio is converted to a **16 kHz mono WAV** using pydub/ffmpeg.
2. Whisper's `transcribe()` method is called with `word_timestamps=True`.
3. The output is a list of segments, each with `start`, `end`, and `text`.

### Speaker Diarization

1. A **sliding window** (1.5 s window, 0.5 s hop) scans the audio.
2. Each window's **MFCC + Δ-MFCC** features are extracted via librosa.
3. Features are **normalised** with `StandardScaler`.
4. **K-Means clustering** groups windows into speaker clusters.
   - When *auto-detect* is enabled, the number of clusters is chosen by
     maximising the **silhouette score** over k = 2 … 4.
5. Consecutive windows with the same cluster label are **merged** into speaker
   segments; a majority-vote smoothing removes isolated outliers.
6. Each Whisper segment is assigned the speaker whose diarization segment covers
   its midpoint.

### Text-to-Speech (TTS)

1. Text is validated and passed to the chosen backend.
2. **pyttsx3** (offline): invokes the OS TTS engine (espeak-ng / SAPI5 / NSS)
   and saves a WAV file.
3. **gTTS** (online): streams audio from Google's TTS API and returns MP3 bytes.
4. The *auto* mode tries pyttsx3 first and falls back to gTTS on failure.

### Voice Cloning

1. The reference audio is loaded and its **mean F0** is estimated with
   `librosa.pyin`.
2. The target text is synthesised with gTTS.
3. The synthesised speech's mean F0 is computed and compared to the reference.
4. `librosa.effects.pitch_shift` is applied (clamped to ±12 semitones) to align
   the pitch to the reference speaker.

### Voice Conversion

1. Source audio is loaded at 22 050 Hz mono.
2. If a target reference is provided, both mean F0 values are computed and the
   pitch-shift delta is derived (clamped to ±24 semitones).
3. `librosa.effects.pitch_shift` transforms the source pitch.
4. An optional `librosa.effects.time_stretch` adjusts playback speed.

---

## 🚧 Future Improvements

- Real-time streaming transcription with Whisper + WebSocket
- Pyannote.audio integration for neural-network–based diarization
- Word-level speaker labels (token-level alignment)
- Speaker voice profiles / recognition across sessions
- Translation mode (transcribe + translate to English)
- Neural TTS with Coqui TTS / VITS for higher voice quality
- Actual fine-tuning pipeline execution on GPU / cloud
- Docker container for one-command deployment

---

