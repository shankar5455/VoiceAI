"""
app.py — VoiceAI: Speech AI Web App with Transcription, TTS, Voice Cloning,
         Voice Conversion and Training.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import time

import pandas as pd
import streamlit as st

from asr import load_model, transcribe_audio
from diarization import perform_diarization, merge_transcript_diarization
from tts import synthesize_speech
from voice_cloning import clone_voice
from voice_conversion import convert_voice
from training import validate_dataset, generate_training_config, mock_training_step
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
    page_title="VoiceAI – Speech AI Suite",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS - Modern Design
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        /* Global Styles */
        * {
            font-family: 'Poppins', sans-serif;
        }

        /* Main container */
        .main {
            background: #0a0e1a;
        }

        /* Card styling */
        .modern-card {
            background: linear-gradient(135deg, rgba(0,198,255,0.08) 0%, rgba(0,114,255,0.05) 100%);
            backdrop-filter: blur(12px);
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(0,198,255,0.18);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .modern-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 16px 40px rgba(0,198,255,0.12);
        }

        /* Hero section */
        .hero-section {
            background: linear-gradient(135deg, #0a0e1a 0%, #0d1b3e 50%, #0a1628 100%);
            border-radius: 22px;
            padding: 48px 40px;
            margin-bottom: 30px;
            text-align: center;
            color: white;
            box-shadow: 0 0 60px rgba(0,198,255,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
            border: 1px solid rgba(0,198,255,0.2);
            position: relative;
            overflow: hidden;
        }

        .hero-section::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(ellipse at center, rgba(0,198,255,0.06) 0%, transparent 60%);
            pointer-events: none;
        }

        .hero-title {
            font-size: 3.2rem;
            font-weight: 800;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #00c6ff 0%, #4facfe 50%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -1px;
        }

        .hero-subtitle {
            font-size: 1.15rem;
            opacity: 0.8;
            margin-bottom: 0;
            color: #b0c4de;
            font-weight: 300;
        }

        /* Result box styling */
        .result-box {
            background: rgba(10, 14, 26, 0.95);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(0,198,255,0.12);
        }

        /* Metric card styling */
        .metric-card {
            background: linear-gradient(135deg, rgba(0,198,255,0.12) 0%, rgba(0,114,255,0.08) 100%);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(0,198,255,0.18);
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: scale(1.05);
            background: linear-gradient(135deg, rgba(0,198,255,0.22) 0%, rgba(0,114,255,0.16) 100%);
            box-shadow: 0 8px 24px rgba(0,198,255,0.15);
        }

        /* Speaker badge styling */
        .speaker-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
            transition: all 0.2s ease;
        }

        .speaker-badge:hover {
            transform: scale(1.05);
            filter: brightness(1.15);
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            background: rgba(0,198,255,0.05);
            padding: 8px;
            border-radius: 14px;
            border: 1px solid rgba(0,198,255,0.1);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 8px 20px;
            font-weight: 500;
            transition: all 0.3s ease;
            color: #8899bb;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(0,198,255,0.1);
            color: #00c6ff;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
            color: white !important;
            box-shadow: 0 4px 14px rgba(0,198,255,0.35);
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 28px;
            font-weight: 600;
            font-family: 'Poppins', sans-serif;
            letter-spacing: 0.3px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 14px rgba(0,198,255,0.3);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,198,255,0.4);
        }

        /* Upload area styling */
        .upload-area {
            border: 2px dashed rgba(0,198,255,0.35);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .upload-area:hover {
            border-color: #00c6ff;
            background: rgba(0,198,255,0.04);
        }

        /* Progress bar styling */
        .stProgress > div > div {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
        }

        /* Status message styling */
        .stAlert {
            border-radius: 12px;
            border-left: 4px solid;
        }

        /* Footer styling */
        .footer {
            text-align: center;
            padding: 22px;
            background: linear-gradient(135deg, rgba(0,198,255,0.06) 0%, rgba(0,114,255,0.04) 100%);
            border-radius: 14px;
            margin-top: 30px;
            border: 1px solid rgba(0,198,255,0.12);
        }

        /* Feature card styling */
        .feature-card {
            background: linear-gradient(135deg, rgba(0,198,255,0.1) 0%, rgba(0,114,255,0.07) 100%);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 18px 16px;
            text-align: center;
            transition: all 0.3s ease;
            border: 1px solid rgba(0,198,255,0.15);
        }

        .feature-card:hover {
            transform: translateY(-5px);
            background: linear-gradient(135deg, rgba(0,198,255,0.18) 0%, rgba(0,114,255,0.14) 100%);
            box-shadow: 0 12px 30px rgba(0,198,255,0.12);
        }

        /* Code block styling */
        .stCodeBlock {
            border-radius: 12px;
        }

        /* Audio player styling */
        audio {
            border-radius: 12px;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Sidebar — settings
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 24px 0 16px 0;">
            <div style="font-size: 52px; filter: drop-shadow(0 0 12px rgba(0,198,255,0.5));">🔊</div>
            <h2 style="background: linear-gradient(135deg, #00c6ff, #0072ff);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text; margin-top: 10px; font-weight: 800; letter-spacing: -0.5px;">
                VoiceAI
            </h2>
            <p style="color: rgba(176,196,222,0.75); font-size: 13px; margin-top: 4px;">Advanced Speech AI Suite</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.subheader("🤖 ASR Model")
    model_size = st.selectbox(
        "Whisper model size",
        options=["tiny", "base", "small", "medium"],
        index=1,
        help="'base' is recommended for CPU. Larger models are more accurate but slower.",
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
    st.markdown(
        """
        <div style="text-align: center; color: rgba(176,196,222,0.45); font-size: 11px; padding: 4px 0;">
            <p style="margin: 0;">Powered by Whisper &amp; Streamlit</p>
            <p style="margin: 4px 0 0 0;">Version 2.0</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Hero Section
# ──────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-title">🔊 VoiceAI</div>
        <div class="hero-subtitle">Next-Generation Speech AI Suite</div>
        <div style="margin-top: 22px;">
            <span style="background: rgba(0,198,255,0.18); border: 1px solid rgba(0,198,255,0.3);
                         padding: 5px 14px; border-radius: 20px; font-size: 12px; color: #b0e8ff;">
                ✨ Real-time Transcription
            </span>
            <span style="background: rgba(0,198,255,0.18); border: 1px solid rgba(0,198,255,0.3);
                         padding: 5px 14px; border-radius: 20px; font-size: 12px; margin-left: 8px; color: #b0e8ff;">
                🎭 Voice Cloning
            </span>
            <span style="background: rgba(0,198,255,0.18); border: 1px solid rgba(0,198,255,0.3);
                         padding: 5px 14px; border-radius: 20px; font-size: 12px; margin-left: 8px; color: #b0e8ff;">
                🔄 Voice Conversion
            </span>
            <span style="background: rgba(0,198,255,0.18); border: 1px solid rgba(0,198,255,0.3);
                         padding: 5px 14px; border-radius: 20px; font-size: 12px; margin-left: 8px; color: #b0e8ff;">
                🔊 TTS
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Main tabs with icons
# ──────────────────────────────────────────────

(
    tab_transcription,
    tab_tts,
    tab_cloning,
    tab_conversion,
    tab_advanced,
) = st.tabs([
    "🎙️ Transcription",
    "🔊 Text-to-Speech",
    "🎭 Voice Cloning",
    "🔄 Voice Conversion",
    "🧠 Advanced Training",
])

# ══════════════════════════════════════════════
# TAB 1 — Transcription
# ══════════════════════════════════════════════

with tab_transcription:
    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 32px;">🎯</div>
                <strong>High Accuracy</strong>
                <p style="font-size: 12px; margin-top: 8px;">Powered by OpenAI Whisper</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 32px;">👥</div>
                <strong>Speaker Diarization</strong>
                <p style="font-size: 12px; margin-top: 8px;">Identify different speakers</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # ── Audio input ────────────────────────────────────────────────────────
    tab_upload, tab_record = st.tabs(["📁 Upload Audio File", "🎤 Record from Microphone"])

    audio_path: str | None = None

    with tab_upload:
        st.markdown("### 📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            st.audio(uploaded_file, format=uploaded_file.type)
            st.success(f"✅ File **{uploaded_file.name}** uploaded successfully!")
            audio_path = save_uploaded_file(uploaded_file)

    with tab_record:
        st.markdown("### 🎤 Record Audio")
        try:
            from audio_recorder_streamlit import audio_recorder

            audio_bytes = audio_recorder(
                text="Click to start/stop recording",
                recording_color="#FF6B6B",
                neutral_color="#4ECDC4",
                icon_name="microphone",
                icon_size="2x",
            )
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                st.success("✅ Recording captured!")
                audio_path = save_audio_bytes(audio_bytes, suffix=".wav")
        except ImportError:
            st.info(
                "💡 **Install audio-recorder-streamlit** to enable microphone recording:  \n"
                "`pip install audio-recorder-streamlit`"
            )

    # ── Process button ─────────────────────────────────────────────────────
    st.markdown("---")
    process_col, _, _ = st.columns([1, 1, 2])
    process_btn = process_col.button("🚀 Process Audio", type="primary", use_container_width=True)

    # ── Processing pipeline ────────────────────────────────────────────────
    if process_btn:
        if audio_path is None:
            st.warning("⚠️ Please upload a file or record audio before clicking **Process**.")
            st.stop()

        wav_path: str | None = None

        try:
            with st.status("🔄 Processing Audio...", expanded=True) as status:
                st.write("🎵 Converting audio to 16 kHz mono WAV...")
                wav_path = convert_to_wav(audio_path)
                st.write("✅ Audio ready")

                st.write("🤖 Loading Whisper model...")
                model, device = load_model(model_size)
                st.write(f"✅ Model loaded on **{device.upper()}**")

                st.write("📝 Transcribing speech...")
                asr_result = transcribe_audio(model, wav_path, language=language)
                st.write(f"✅ Transcription complete — detected language: **{asr_result['language']}**")

                st.write("👥 Identifying speakers...")
                audio_array, sr = load_audio(wav_path, sr=16000)
                diar_segments = perform_diarization(audio_array, sr, n_speakers=n_speakers)
                detected_speakers = {s["speaker"] for s in diar_segments}
                st.write(f"✅ Detected **{len(detected_speakers)} speaker(s)**")

                st.write("🔗 Merging transcript with speaker labels...")
                merged = merge_transcript_diarization(asr_result["segments"], diar_segments)
                status.update(label="✅ Processing complete!", state="complete", expanded=False)

            if not merged:
                st.error("❌ No speech detected in the audio. Please try a different file.")
                st.stop()

            # Save transcript to session state
            transcript_text = create_transcript_text(merged)
            st.session_state["last_transcript"] = transcript_text
            st.session_state["last_full_text"] = asr_result.get("full_text", "")

            st.markdown("---")
            st.markdown("## 📝 Transcript")
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            html = render_colored_transcript(merged)
            st.markdown(html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Speaker legend
            speakers = sorted({s["speaker"] for s in merged})
            legend_parts = []
            for sp in speakers:
                color = get_speaker_color(sp)
                legend_parts.append(
                    f'<span class="speaker-badge" style="background:{color};color:#fff;">{sp}</span>'
                )
            st.markdown(
                "<p><strong>Speaker Legend:</strong> " + " ".join(legend_parts) + "</p>",
                unsafe_allow_html=True,
            )

            # Speaker statistics
            st.markdown("---")
            st.markdown("## 📊 Speaker Statistics")
            stats = speaker_stats(merged)
            total_time = sum(stats.values())

            stat_cols = st.columns(len(stats))
            for col, (speaker, duration) in zip(stat_cols, stats.items()):
                pct = (duration / total_time * 100) if total_time > 0 else 0
                col.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size: 20px; font-weight: 600;">{speaker}</div>
                        <div style="font-size: 28px; font-weight: 700; margin: 10px 0;">{format_timestamp(duration)}</div>
                        <div style="font-size: 12px; opacity: 0.7;">{pct:.1f}% of audio</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Download transcript
            st.markdown("---")
            col_download, _ = st.columns([1, 3])
            with col_download:
                st.download_button(
                    label="⬇️ Download Transcript (.txt)",
                    data=transcript_text,
                    file_name="voiceai_transcript.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"❌ An error occurred during processing:\n\n`{exc}`")
            st.exception(exc)

        finally:
            if wav_path:
                cleanup_temp_file(wav_path)
            if audio_path:
                cleanup_temp_file(audio_path)


# ══════════════════════════════════════════════
# TAB 2 — Text-to-Speech
# ══════════════════════════════════════════════

with tab_tts:
    st.markdown("## 🔊 Text-to-Speech")
    st.markdown(
        """
        <div class="modern-card">
            Convert text to natural-sounding speech with multiple TTS engines.
            Choose between online (gTTS) or offline (pyttsx3) synthesis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pre-fill from last transcript if available
    default_text = st.session_state.get("last_full_text", "")

    tts_text = st.text_area(
        "📝 Text to convert",
        value=default_text,
        height=150,
        placeholder="Enter or paste text here…",
    )

    col_backend, col_lang, col_rate = st.columns(3)

    with col_backend:
        tts_backend = st.selectbox(
            "🎛️ TTS Engine",
            options=["auto", "gtts", "pyttsx3"],
            index=0,
            help="**auto** — tries offline first, then online. **gtts** — Google TTS (higher quality). **pyttsx3** — fully offline.",
        )

    with col_lang:
        tts_lang_options = {
            "English": "en",
            "French": "fr",
            "Spanish": "es",
            "German": "de",
            "Hindi": "hi",
            "Arabic": "ar",
        }
        tts_lang_label = st.selectbox(
            "🌐 Language (gTTS)",
            list(tts_lang_options.keys()),
            help="Language used by gTTS. Ignored when using pyttsx3.",
        )
        tts_lang = tts_lang_options[tts_lang_label]

    with col_rate:
        tts_rate = st.slider(
            "⚡ Speech Rate (pyttsx3)",
            min_value=80,
            max_value=300,
            value=150,
            step=10,
            help="Words per minute. Only affects pyttsx3 engine.",
        )

    if st.button("🔊 Convert to Speech", type="primary"):
        if not tts_text.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("🎵 Synthesising speech..."):
                try:
                    audio_bytes, ext = synthesize_speech(
                        tts_text,
                        backend=tts_backend,
                        lang=tts_lang,
                        rate=tts_rate,
                    )
                    mime = "audio/wav" if ext == ".wav" else "audio/mpeg"
                    st.success("✅ Speech synthesised successfully!")
                    st.audio(audio_bytes, format=mime)
                    st.download_button(
                        label=f"⬇️ Download Audio ({ext})",
                        data=audio_bytes,
                        file_name=f"voiceai_tts{ext}",
                        mime=mime,
                    )
                except Exception as exc:
                    st.error(f"❌ TTS failed: {exc}")
                    st.exception(exc)


# ══════════════════════════════════════════════
# TAB 3 — Voice Cloning
# ══════════════════════════════════════════════

with tab_cloning:
    st.markdown("## 🎭 Voice Cloning")
    st.markdown(
        """
        <div class="modern-card">
            Clone any voice with just a reference sample! Upload a clear voice recording and 
            enter the text you want spoken in that voice.
            <br><br>
            <strong>💡 Tip:</strong> For best results, use a reference audio with at least 5 seconds of clear speech.
        </div>
        """,
        unsafe_allow_html=True,
    )

    clone_ref_file = st.file_uploader(
        "📎 Upload Reference Voice Sample",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="clone_ref",
        help="Upload a clear voice recording (WAV/MP3, ≥5 seconds recommended)",
    )
    if clone_ref_file:
        st.audio(clone_ref_file, format=clone_ref_file.type)

    clone_text = st.text_area(
        "📝 Text to Synthesise",
        height=120,
        placeholder="Enter the text you want spoken in the reference voice…",
        key="clone_text",
    )

    clone_lang_options = {
        "English": "en",
        "French": "fr",
        "Spanish": "es",
        "German": "de",
        "Hindi": "hi",
        "Arabic": "ar",
    }
    clone_lang_label = st.selectbox(
        "🌐 Language",
        list(clone_lang_options.keys()),
        key="clone_lang",
    )
    clone_lang = clone_lang_options[clone_lang_label]

    if st.button("🎭 Clone & Synthesise", type="primary"):
        if clone_ref_file is None:
            st.warning("⚠️ Please upload a reference voice sample.")
        elif not clone_text.strip():
            st.warning("⚠️ Please enter some text to synthesise.")
        else:
            ref_path: str | None = None
            try:
                ref_path = save_uploaded_file(clone_ref_file)
                with st.spinner("🎭 Analysing reference voice and synthesising..."):
                    cloned_bytes = clone_voice(clone_text, ref_path, lang=clone_lang)
                st.success("✅ Voice cloning complete!")
                st.audio(cloned_bytes, format="audio/wav")
                st.download_button(
                    label="⬇️ Download Cloned Audio (.wav)",
                    data=cloned_bytes,
                    file_name="voiceai_cloned.wav",
                    mime="audio/wav",
                )
            except Exception as exc:
                st.error(f"❌ Voice cloning failed: {exc}")
                st.exception(exc)
            finally:
                if ref_path:
                    cleanup_temp_file(ref_path)


# ══════════════════════════════════════════════
# TAB 4 — Voice Conversion
# ══════════════════════════════════════════════

with tab_conversion:
    st.markdown("## 🔄 Voice Conversion")
    st.markdown(
        """
        <div class="modern-card">
            Transform voices with advanced pitch and speed manipulation. Upload a source voice 
            and optionally a target voice to match its characteristics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    src_file = st.file_uploader(
        "🎵 Upload **Source** Audio (the voice to convert)",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="vc_src",
    )
    if src_file:
        st.audio(src_file, format=src_file.type)

    tgt_file = st.file_uploader(
        "🎯 Upload **Target** Voice Sample (optional — leave blank for manual shift)",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="vc_tgt",
        help="Upload a target voice to automatically match its pitch",
    )
    if tgt_file:
        st.audio(tgt_file, format=tgt_file.type)

    col_pitch, col_speed = st.columns(2)

    with col_pitch:
        pitch_shift = st.slider(
            "🎵 Pitch Shift (semitones)",
            min_value=-24,
            max_value=24,
            value=0,
            step=1,
            help="Ignored when a target voice is uploaded",
        )

    with col_speed:
        speed_factor = st.slider(
            "⏱️ Speed Factor",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="1.0 = original speed, >1 speeds up, <1 slows down",
        )

    if st.button("🔄 Convert Voice", type="primary"):
        if src_file is None:
            st.warning("⚠️ Please upload a source audio file.")
        else:
            src_path: str | None = None
            tgt_path: str | None = None
            try:
                src_path = save_uploaded_file(src_file)
                if tgt_file is not None:
                    tgt_path = save_uploaded_file(tgt_file)

                with st.spinner("🔄 Converting voice..."):
                    converted_bytes = convert_voice(
                        source_audio_path=src_path,
                        target_audio_path=tgt_path,
                        pitch_shift_semitones=float(pitch_shift),
                        speed_factor=float(speed_factor),
                    )

                st.success("✅ Voice conversion complete!")
                st.audio(converted_bytes, format="audio/wav")
                st.download_button(
                    label="⬇️ Download Converted Audio (.wav)",
                    data=converted_bytes,
                    file_name="voiceai_converted.wav",
                    mime="audio/wav",
                )
            except Exception as exc:
                st.error(f"❌ Voice conversion failed: {exc}")
                st.exception(exc)
            finally:
                if src_path:
                    cleanup_temp_file(src_path)
                if tgt_path:
                    cleanup_temp_file(tgt_path)


# ══════════════════════════════════════════════
# TAB 5 — Advanced: Training
# ══════════════════════════════════════════════

with tab_advanced:
    st.markdown("## 🧠 Advanced: Train Custom Model")
    st.markdown(
        """
        <div class="modern-card">
            Create your own custom speech models! Upload your dataset, configure training parameters,
            and let the system guide you through the training process.
            <br><br>
            <strong>⚠️ Note:</strong> Full model training requires significant computational resources.
            This section provides a guided pipeline with simulation capabilities.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📂 Step 1 — Upload Dataset")
    st.markdown(
        """
        Upload a **ZIP archive** containing:
        - `audio/` folder with `.wav`/`.mp3` utterances
        - `transcripts.txt` file with lines in `filename|text` format
        """
    )

    dataset_zip = st.file_uploader(
        "Upload dataset ZIP",
        type=["zip"],
        key="training_zip",
    )

    if dataset_zip is not None:
        zip_path: str | None = None
        try:
            zip_path = save_uploaded_file(dataset_zip)
            summary = validate_dataset(zip_path)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size: 28px;">📄</div>
                        <div style="font-size: 24px; font-weight: 700;">{summary['total_files']}</div>
                        <div>Total Files</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size: 28px;">🎵</div>
                        <div style="font-size: 24px; font-weight: 700;">{summary['audio_files']}</div>
                        <div>Audio Files</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_c:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size: 28px;">📝</div>
                        <div style="font-size: 24px; font-weight: 700;">{summary['text_files']}</div>
                        <div>Text Files</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if summary["warnings"]:
                for w in summary["warnings"]:
                    st.warning(f"⚠️ {w}")
            else:
                st.success("✅ Dataset validation passed!")
        except Exception as exc:
            st.error(f"❌ Dataset validation failed: {exc}")
        finally:
            if zip_path:
                cleanup_temp_file(zip_path)

    st.markdown("---")
    st.markdown("### ⚙️ Step 2 — Configure Training")

    col_model, col_epochs, col_batch = st.columns(3)

    with col_model:
        train_model_type = st.selectbox(
            "Model Type",
            options=["tts", "asr"],
            help="**tts** — Text-to-Speech model, **asr** — Automatic Speech Recognition",
        )

    with col_epochs:
        train_epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=1000,
            value=100,
            step=10,
        )

    with col_batch:
        train_batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=128,
            value=16,
            step=4,
        )

    config_json = generate_training_config(
        model_type=train_model_type,
        epochs=int(train_epochs),
        batch_size=int(train_batch_size),
    )

    with st.expander("📄 View Generated Configuration"):
        st.code(config_json, language="json")
        st.download_button(
            label="⬇️ Download Config (.json)",
            data=config_json,
            file_name="voiceai_training_config.json",
            mime="application/json",
        )

    st.markdown("---")
    st.markdown("### 🚀 Step 3 — Start Training (Simulation)")

    demo_epochs = st.slider("Simulation Epochs", min_value=5, max_value=30, value=10)

    if st.button("▶️ Start Training", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()

        loss_history: list[float] = []
        acc_history: list[float] = []

        for ep in range(1, demo_epochs + 1):
            step = mock_training_step(ep, demo_epochs)
            loss_history.append(step["loss"])
            acc_history.append(step["accuracy"])

            progress_bar.progress(int(step["progress_pct"]))
            status_text.markdown(
                f"""
                <div class="modern-card">
                    <strong>Epoch {ep}/{demo_epochs}</strong><br>
                    Loss: <code>{step['loss']:.4f}</code> | 
                    Accuracy: <code>{step['accuracy']:.4f}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Update live chart
            chart_data = pd.DataFrame(
                {"Loss": loss_history, "Accuracy": acc_history},
                index=range(1, ep + 1),
            )
            chart_placeholder.line_chart(chart_data)

            time.sleep(0.3)

        progress_bar.progress(100)
        status_text.success(
            f"""
            ✅ Simulation complete! Final Results:
            - Loss: `{loss_history[-1]:.4f}`
            - Accuracy: `{acc_history[-1]:.4f}`
            """
        )
        st.info(
            "💡 **Next Steps:** Export the configuration above and use it with "
            "[Coqui TTS](https://github.com/coqui-ai/TTS) or "
            "[Whisper fine-tuning](https://github.com/openai/whisper) for real training.",
            icon="💡",
        )


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <p style="margin: 0;">🔊 <strong>VoiceAI</strong> — Advanced Speech AI Suite</p>
        <p style="margin: 5px 0 0 0; font-size: 12px; opacity: 0.7;">
            Powered by OpenAI Whisper · Built with Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)