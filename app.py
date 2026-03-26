"""
app.py — HearMeAI: Real-Time Speech-to-Text Web App with Speaker Identification.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from asr import load_model, transcribe_audio
from diarization import perform_diarization, merge_transcript_diarization
from utils import (
    save_uploaded_file,
    save_audio_bytes,
    convert_to_wav,
    load_audio,
    create_transcript_text,
    render_colored_transcript,
    speaker_stats,
    get_speaker_color,
    cleanup_temp_file,
    format_timestamp,
)

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="HearMeAI – Speech AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
        /* Dark card-style containers */
        .result-box {
            background-color: #1E1E2E;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }
        /* Speaker colour badges */
        .speaker-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: bold;
            margin-right: 6px;
        }
        /* Metric cards */
        .metric-card {
            background: #2A2A3E;
            border-radius: 8px;
            padding: 14px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Sidebar — settings
# ──────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/000000/microphone.png",
        width=64,
    )
    st.title("HearMeAI ⚙️")
    st.markdown("---")

    st.subheader("🤖 ASR Model")
    model_size = st.selectbox(
        "Whisper model size",
        options=["tiny", "base", "small", "medium"],
        index=1,
        help="'base' is recommended for CPU.  Larger models are more accurate but slower.",
    )

    st.subheader("🗣️ Diarization")
    auto_detect = st.checkbox("Auto-detect number of speakers", value=True)
    if not auto_detect:
        n_speakers = st.slider("Number of speakers", min_value=2, max_value=6, value=2)
    else:
        n_speakers = None

    st.subheader("🌐 Language")
    language_options = {
        "Auto-detect": None,
        "English": "en",
        "French": "fr",
        "Spanish": "es",
        "German": "de",
        "Hindi": "hi",
        "Arabic": "ar",
    }
    lang_label = st.selectbox("Transcription language", list(language_options.keys()))
    language = language_options[lang_label]

    st.markdown("---")
    st.caption("Built with Whisper + Streamlit 🚀")


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.title("🎙️ HearMeAI")
st.subheader("Real-Time Speech-to-Text with Speaker Identification")
st.markdown(
    "Upload an audio file **or** record directly from your microphone.  "
    "HearMeAI will transcribe the speech and identify different speakers."
)
st.markdown("---")

# ──────────────────────────────────────────────
# Audio input tabs
# ──────────────────────────────────────────────

tab_upload, tab_record = st.tabs(["📁 Upload Audio File", "🎤 Record from Microphone"])

audio_path: str | None = None  # will be set in whichever tab is used

with tab_upload:
    st.markdown("#### Upload a **.wav** or **.mp3** file")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        st.success(f"✅ File **{uploaded_file.name}** ready for processing.")
        # Persist file to disk for processing
        audio_path = save_uploaded_file(uploaded_file)

with tab_record:
    st.markdown("#### Record audio directly in your browser")
    try:
        from audio_recorder_streamlit import audio_recorder

        audio_bytes = audio_recorder(
            text="Click to start / stop recording",
            recording_color="#FF6B6B",
            neutral_color="#4ECDC4",
            icon_name="microphone",
            icon_size="2x",
        )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            st.success("✅ Recording captured — click **Process** below.")
            audio_path = save_audio_bytes(audio_bytes, suffix=".wav")
    except ImportError:
        st.info(
            "Install **audio-recorder-streamlit** to enable microphone recording:  \n"
            "`pip install audio-recorder-streamlit`"
        )

# ──────────────────────────────────────────────
# Process button
# ──────────────────────────────────────────────

st.markdown("---")
process_col, _ = st.columns([1, 4])
process_btn = process_col.button("🚀 Process Audio", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# Processing pipeline
# ──────────────────────────────────────────────

if process_btn:
    if audio_path is None:
        st.warning("⚠️ Please upload a file or record audio before clicking **Process**.")
        st.stop()

    wav_path: str | None = None  # path to the converted WAV (may be a temp file)

    try:
        # ── Step 1: Convert to WAV if needed ──────────────────────────────
        with st.status("🔄 Preparing audio…", expanded=True) as status:
            st.write("Converting audio to 16 kHz mono WAV…")
            wav_path = convert_to_wav(audio_path)
            st.write("✅ Audio ready.")

            # ── Step 2: Load ASR model ─────────────────────────────────────
            st.write(f"Loading Whisper **{model_size}** model…")
            model, device = load_model(model_size)
            st.write(f"✅ Model loaded on **{device.upper()}**.")

            # ── Step 3: Speech-to-text ─────────────────────────────────────
            st.write("Transcribing speech… (this may take a moment on CPU)")
            asr_result = transcribe_audio(model, wav_path, language=language)
            st.write(f"✅ Transcription complete — detected language: **{asr_result['language']}**")

            # ── Step 4: Speaker diarization ────────────────────────────────
            st.write("Identifying speakers…")
            audio_array, sr = load_audio(wav_path, sr=16000)
            diar_segments = perform_diarization(audio_array, sr, n_speakers=n_speakers)
            detected_speakers = {s["speaker"] for s in diar_segments}
            st.write(f"✅ Detected **{len(detected_speakers)} speaker(s)**.")

            # ── Step 5: Merge results ──────────────────────────────────────
            st.write("Merging transcript with speaker labels…")
            merged = merge_transcript_diarization(asr_result["segments"], diar_segments)
            status.update(label="✅ Processing complete!", state="complete", expanded=False)

        # ── Display results ────────────────────────────────────────────────
        if not merged:
            st.error("❌ No speech detected in the audio.  Please try a different file.")
            st.stop()

        st.markdown("---")
        st.header("📝 Transcript")

        # Coloured transcript
        html = render_colored_transcript(merged)
        st.markdown(
            f'<div class="result-box">{html}</div>',
            unsafe_allow_html=True,
        )

        # ── Speaker legend ─────────────────────────────────────────────────
        speakers = sorted({s["speaker"] for s in merged})
        legend_parts = []
        for sp in speakers:
            color = get_speaker_color(sp)
            legend_parts.append(
                f'<span class="speaker-badge" style="background:{color};color:#111;">{sp}</span>'
            )
        st.markdown(
            "<p><strong>Legend:</strong> " + " ".join(legend_parts) + "</p>",
            unsafe_allow_html=True,
        )

        # ── Statistics ─────────────────────────────────────────────────────
        st.markdown("---")
        st.header("📊 Speaker Statistics")
        stats = speaker_stats(merged)
        total_time = sum(stats.values())

        stat_cols = st.columns(len(stats))
        for col, (speaker, duration) in zip(stat_cols, stats.items()):
            pct = (duration / total_time * 100) if total_time > 0 else 0
            col.metric(
                label=speaker,
                value=format_timestamp(duration),
                delta=f"{pct:.1f}% of audio",
            )

        # ── Download ───────────────────────────────────────────────────────
        st.markdown("---")
        transcript_text = create_transcript_text(merged)
        st.download_button(
            label="⬇️ Download Transcript (.txt)",
            data=transcript_text,
            file_name="hearmeai_transcript.txt",
            mime="text/plain",
        )

    except Exception as exc:
        st.error(f"❌ An error occurred during processing:\n\n`{exc}`")
        st.exception(exc)

    finally:
        # Clean up temporary files
        if wav_path:
            cleanup_temp_file(wav_path)
        if audio_path:
            cleanup_temp_file(audio_path)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.markdown("---")
st.caption(
    "HearMeAI · Powered by [OpenAI Whisper](https://github.com/openai/whisper) · "
    "Built with [Streamlit](https://streamlit.io)"
)
