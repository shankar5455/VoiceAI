# 🎙️ HearMeAI — Real-Time Speech-to-Text with Speaker Identification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Whisper](https://img.shields.io/badge/ASR-OpenAI%20Whisper-00C853.svg)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A hackathon-ready, end-to-end **Speech AI** web application that transcribes audio and
labels different speakers — all running **locally** on CPU without any API keys.

---

## 📌 Project Overview

HearMeAI combines two core Speech AI tasks:

| Task | Technology |
|---|---|
| Automatic Speech Recognition (ASR) | OpenAI Whisper (pretrained, local) |
| Speaker Diarization | MFCC + K-Means clustering (scikit-learn) |

Users can upload an audio file or record directly in the browser. HearMeAI will
produce a colour-coded, timestamped transcript with speaker labels that can be
downloaded as a `.txt` file.

---

## ✨ Features

- 🎵 **Audio input** — upload `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac` *or* record from microphone
- 🗣️ **Accurate ASR** — OpenAI Whisper (tiny / base / small / medium model selectable)
- 👥 **Speaker diarization** — identifies 2–6 speakers; auto-detects count via silhouette score
- 🌈 **Colour-coded transcript** — each speaker rendered in a distinct colour
- ⏱️ **Timestamps** — every utterance shows `[MM:SS.ss → MM:SS.ss]`
- 📊 **Speaker statistics** — speaking time and percentage per speaker
- ⬇️ **Download transcript** — export plain-text `.txt` file
- 🌐 **Multi-language** — auto-detect or specify English, French, Spanish, German, Hindi, Arabic
- 💻 **CPU optimised** — runs on any modern laptop; no GPU required

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
| Microphone | audio-recorder-streamlit |

---

## 📁 Project Structure

```
HearMeAI/
├── app.py              # Main Streamlit application
├── asr.py              # Speech-to-text with OpenAI Whisper
├── diarization.py      # Speaker diarization (MFCC + K-Means)
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

1. **Upload** a `.wav` or `.mp3` file using the *Upload Audio File* tab, **or**
   click the microphone button in the *Record from Microphone* tab.
2. *(Optional)* Adjust the Whisper model size, number of speakers, and language
   in the left sidebar.
3. Click **🚀 Process Audio**.
4. View the colour-coded, timestamped transcript and speaker statistics.
5. Click **⬇️ Download Transcript (.txt)** to save the results.

---

## 📸 Sample Output

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

---

## 🚧 Future Improvements

- Real-time streaming transcription with Whisper + WebSocket
- Pyannote.audio integration for neural-network–based diarization
- Word-level speaker labels (token-level alignment)
- Speaker voice profiles / recognition across sessions
- Translation mode (transcribe + translate to English)
- Docker container for one-command deployment

---

## 📄 License

MIT © 2024 HearMeAI Contributors
