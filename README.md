# 🎙️ HearMeAI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Whisper](https://img.shields.io/badge/ASR-OpenAI%20Whisper-00C853.svg)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

HearMeAI is an end-to-end **Speech AI** web application built with Streamlit. It transcribes audio, identifies different speakers, converts text to speech, and supports voice cloning and voice conversion — all running **locally** on CPU without any API keys.

---

## ✨ Features

- 🗣️ **Transcription** — upload or record audio and get a timestamped, colour-coded transcript with speaker labels
- 🔊 **Text-to-Speech** — convert text to speech using gTTS (online) or pyttsx3 (offline)
- 🎭 **Voice Cloning** — generate speech that matches the pitch of a reference speaker
- 🔄 **Voice Conversion** — shift the pitch and speed of any audio to match a target voice
- 🌐 **Multi-language** — auto-detect or choose from English, French, Spanish, German, Hindi, Arabic
- 💻 **CPU-friendly** — no GPU required

---

## 📁 Project Structure

```
HearMeAI/
├── app.py              # Main Streamlit application
├── asr.py              # Speech-to-text using OpenAI Whisper
├── diarization.py      # Speaker diarization (MFCC + K-Means)
├── tts.py              # Text-to-Speech (gTTS / pyttsx3)
├── voice_cloning.py    # Voice cloning via pitch matching
├── voice_conversion.py # Voice conversion (pitch shift + time stretch)
├── utils.py            # Audio helpers and formatting utilities
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Prerequisites

- Python **3.10** or higher
- **ffmpeg** installed and available on your PATH

Install ffmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shankar5455/HearMeAI.git
cd HearMeAI
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download Whisper model weights (~75 MB for `base`). Subsequent runs reuse the cached weights.

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## 🎬 Usage

### Transcription
1. Go to the **Transcription** tab.
2. Upload a `.wav`, `.mp3`, `.m4a`, `.ogg`, or `.flac` file, or record directly from your microphone.
3. Optionally adjust the Whisper model size, number of speakers, and language in the sidebar.
4. Click **Process Audio** to get a colour-coded, timestamped transcript.
5. Download the transcript as a `.txt` file.

### Text-to-Speech
1. Go to the **Text-to-Speech** tab.
2. Enter or paste text, choose an engine and language, then click **Generate Speech**.

### Voice Cloning
1. Go to the **Voice Cloning** tab.
2. Upload a short reference audio clip from the target speaker.
3. Enter the text to synthesise and click **Clone Voice**.

### Voice Conversion
1. Go to the **Voice Conversion** tab.
2. Upload a source audio file and optionally a target reference audio.
3. Adjust pitch and speed settings, then click **Convert Voice**.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| ASR | OpenAI Whisper |
| Deep learning | PyTorch (CPU) |
| Audio features | librosa |
| Speaker clustering | scikit-learn KMeans |
| TTS | gTTS, pyttsx3 |
| Audio I/O | soundfile, pydub |
| Microphone | audio-recorder-streamlit |

---

