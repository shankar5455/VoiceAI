"""
app.py — HearMeAI: Speech AI Web App with Transcription, TTS, Voice Cloning,
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
st.subheader("Speech AI — Transcription · TTS · Voice Cloning · Voice Conversion")
st.markdown("---")

# ──────────────────────────────────────────────
# Main tabs
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
# TAB 1 — Transcription (existing feature)
# ══════════════════════════════════════════════

with tab_transcription:
    st.markdown(
        "Upload an audio file **or** record directly from your microphone.  "
        "HearMeAI will transcribe the speech and identify different speakers."
    )

    # ── Audio input ────────────────────────────────────────────────────────
    tab_upload, tab_record = st.tabs(["📁 Upload Audio File", "🎤 Record from Microphone"])

    audio_path: str | None = None

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

    # ── Process button ─────────────────────────────────────────────────────
    st.markdown("---")
    process_col, _ = st.columns([1, 4])
    process_btn = process_col.button("🚀 Process Audio", type="primary", use_container_width=True)

    # ── Processing pipeline ────────────────────────────────────────────────
    if process_btn:
        if audio_path is None:
            st.warning("⚠️ Please upload a file or record audio before clicking **Process**.")
            st.stop()

        wav_path: str | None = None

        try:
            with st.status("🔄 Preparing audio…", expanded=True) as status:
                st.write("Converting audio to 16 kHz mono WAV…")
                wav_path = convert_to_wav(audio_path)
                st.write("✅ Audio ready.")

                st.write(f"Loading Whisper **{model_size}** model…")
                model, device = load_model(model_size)
                st.write(f"✅ Model loaded on **{device.upper()}**.")

                st.write("Transcribing speech… (this may take a moment on CPU)")
                asr_result = transcribe_audio(model, wav_path, language=language)
                st.write(f"✅ Transcription complete — detected language: **{asr_result['language']}**")

                st.write("Identifying speakers…")
                audio_array, sr = load_audio(wav_path, sr=16000)
                diar_segments = perform_diarization(audio_array, sr, n_speakers=n_speakers)
                detected_speakers = {s["speaker"] for s in diar_segments}
                st.write(f"✅ Detected **{len(detected_speakers)} speaker(s)**.")

                st.write("Merging transcript with speaker labels…")
                merged = merge_transcript_diarization(asr_result["segments"], diar_segments)
                status.update(label="✅ Processing complete!", state="complete", expanded=False)

            if not merged:
                st.error("❌ No speech detected in the audio.  Please try a different file.")
                st.stop()

            # Save transcript to session state for use in TTS tab
            transcript_text = create_transcript_text(merged)
            st.session_state["last_transcript"] = transcript_text
            st.session_state["last_full_text"] = asr_result.get("full_text", "")

            st.markdown("---")
            st.header("📝 Transcript")

            html = render_colored_transcript(merged)
            st.markdown(
                f'<div class="result-box">{html}</div>',
                unsafe_allow_html=True,
            )

            # Speaker legend
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

            # Speaker statistics
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

            # Download transcript
            st.markdown("---")
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
            if wav_path:
                cleanup_temp_file(wav_path)
            if audio_path:
                cleanup_temp_file(audio_path)


# ══════════════════════════════════════════════
# TAB 2 — Text-to-Speech
# ══════════════════════════════════════════════

with tab_tts:
    st.header("🔊 Text-to-Speech")
    st.markdown(
        "Convert text to spoken audio.  "
        "You can type your own text or use the transcript generated in the **Transcription** tab."
    )

    # Pre-fill from last transcript if available
    default_text = st.session_state.get("last_full_text", "")

    tts_text = st.text_area(
        "Text to convert",
        value=default_text,
        height=150,
        placeholder="Enter or paste text here…",
    )

    col_backend, col_lang, col_rate = st.columns(3)

    with col_backend:
        tts_backend = st.selectbox(
            "TTS engine",
            options=["auto", "gtts", "pyttsx3"],
            index=0,
            help=(
                "**auto** — tries offline (pyttsx3) first, then online (gTTS).  \n"
                "**gtts** — Google TTS (requires internet, higher quality).  \n"
                "**pyttsx3** — fully offline (requires espeak on Linux)."
            ),
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
            "Language (gTTS)",
            list(tts_lang_options.keys()),
            help="Language used by gTTS. Ignored when using pyttsx3.",
        )
        tts_lang = tts_lang_options[tts_lang_label]

    with col_rate:
        tts_rate = st.slider(
            "Speech rate (pyttsx3)",
            min_value=80,
            max_value=300,
            value=150,
            step=10,
            help="Words per minute. Only affects the pyttsx3 engine.",
        )

    if st.button("🔊 Convert to Speech", type="primary"):
        if not tts_text.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("Synthesising speech…"):
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
                        file_name=f"hearmeai_tts{ext}",
                        mime=mime,
                    )
                except Exception as exc:
                    st.error(f"❌ TTS failed: {exc}")
                    st.exception(exc)


# ══════════════════════════════════════════════
# TAB 3 — Voice Cloning
# ══════════════════════════════════════════════

with tab_cloning:
    st.header("🎭 Voice Cloning (Basic)")
    st.markdown(
        "Upload a **reference voice sample** and enter the text you want "
        "spoken in that voice.  "
        "The system will analyse the speaker's pitch and apply it to the synthesised speech."
    )
    st.info(
        "ℹ️ This is a lightweight, CPU-friendly approximation.  "
        "It matches pitch but not timbre or accent.",
        icon="ℹ️",
    )

    clone_ref_file = st.file_uploader(
        "📎 Upload reference voice sample (WAV / MP3, ideally ≥5 s of clear speech)",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="clone_ref",
    )
    if clone_ref_file:
        st.audio(clone_ref_file, format=clone_ref_file.type)

    clone_text = st.text_area(
        "Text to synthesise in the reference voice",
        height=120,
        placeholder="Enter the text you want spoken…",
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
        "Language",
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
                with st.spinner("Analysing reference voice and synthesising…"):
                    cloned_bytes = clone_voice(clone_text, ref_path, lang=clone_lang)
                st.success("✅ Voice cloning complete!")
                st.audio(cloned_bytes, format="audio/wav")
                st.download_button(
                    label="⬇️ Download Cloned Audio (.wav)",
                    data=cloned_bytes,
                    file_name="hearmeai_cloned.wav",
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
    st.header("🔄 Voice Conversion")
    st.markdown(
        "Transform the pitch and/or speed of a source audio clip.  "
        "Optionally upload a **target voice** sample to automatically match its pitch."
    )

    src_file = st.file_uploader(
        "📎 Upload **source** audio (the voice to convert)",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="vc_src",
    )
    if src_file:
        st.audio(src_file, format=src_file.type)

    tgt_file = st.file_uploader(
        "📎 Upload **target** voice sample (optional — leave blank for manual shift)",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="vc_tgt",
    )
    if tgt_file:
        st.audio(tgt_file, format=tgt_file.type)

    col_pitch, col_speed = st.columns(2)

    with col_pitch:
        pitch_shift = st.slider(
            "Manual pitch shift (semitones)",
            min_value=-24,
            max_value=24,
            value=0,
            step=1,
            help="Ignored when a target voice is uploaded.",
        )

    with col_speed:
        speed_factor = st.slider(
            "Speed factor",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="1.0 = original speed.  >1 speeds up, <1 slows down.",
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

                with st.spinner("Converting voice…"):
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
                    file_name="hearmeai_converted.wav",
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
    st.header("🧠 Advanced: Train Custom Model")
    st.markdown(
        "This section provides a **guided placeholder** for training custom TTS or ASR models.  \n"
        "Actual model training requires a GPU; the steps below show the pipeline and generate "
        "a configuration you can use in a full training environment."
    )
    st.warning(
        "⚠️ **Note:** Training is computationally expensive. "
        "The 'Start Training' button below runs a *simulated* demonstration only.",
        icon="⚠️",
    )

    st.markdown("---")
    st.subheader("📂 Step 1 — Upload Dataset (Optional)")
    st.markdown(
        "Upload a **ZIP archive** containing:  \n"
        "- An `audio/` folder with `.wav` / `.mp3` utterances.  \n"
        "- A `transcripts.txt` file with lines in `filename|text` format."
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
            col_a.metric("Total files", summary["total_files"])
            col_b.metric("Audio files", summary["audio_files"])
            col_c.metric("Text files", summary["text_files"])

            if summary["warnings"]:
                for w in summary["warnings"]:
                    st.warning(f"⚠️ {w}")
            else:
                st.success("✅ Dataset looks good!")
        except Exception as exc:
            st.error(f"❌ Dataset validation failed: {exc}")
        finally:
            if zip_path:
                cleanup_temp_file(zip_path)

    st.markdown("---")
    st.subheader("⚙️ Step 2 — Configure Training")

    col_model, col_epochs, col_batch = st.columns(3)

    with col_model:
        train_model_type = st.selectbox(
            "Model type",
            options=["tts", "asr"],
            help="**tts** — Glow-TTS architecture.  **asr** — QuartzNet architecture.",
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
            "Batch size",
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

    with st.expander("📄 View generated training configuration"):
        st.code(config_json, language="json")
        st.download_button(
            label="⬇️ Download Config (.json)",
            data=config_json,
            file_name="hearmeai_training_config.json",
            mime="application/json",
        )

    st.markdown("---")
    st.subheader("🚀 Step 3 — Start Training (Simulation)")
    st.markdown(
        "Click **Start Training** to run a *simulated* training loop that "
        "demonstrates what the progress would look like."
    )

    demo_epochs = st.slider("Simulation epochs", min_value=5, max_value=30, value=10)

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
                f"**Epoch {ep}/{demo_epochs}** — "
                f"Loss: `{step['loss']:.4f}` | "
                f"Accuracy: `{step['accuracy']:.4f}`"
            )

            # Update live chart
            chart_data = pd.DataFrame(
                {"Loss": loss_history, "Accuracy": acc_history},
                index=range(1, ep + 1),
            )
            chart_placeholder.line_chart(chart_data)

            time.sleep(0.3)  # brief pause to animate the simulation

        progress_bar.progress(100)
        status_text.success(
            f"✅ Simulation complete! Final — "
            f"Loss: `{loss_history[-1]:.4f}` | "
            f"Accuracy: `{acc_history[-1]:.4f}`"
        )
        st.info(
            "To run real training, export the configuration above and use it with "
            "[Coqui TTS](https://github.com/coqui-ai/TTS) or "
            "[Whisper fine-tuning](https://github.com/openai/whisper).",
            icon="💡",
        )


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.markdown("---")
st.caption(
    "HearMeAI · Powered by [OpenAI Whisper](https://github.com/openai/whisper) · "
    "Built with [Streamlit](https://streamlit.io)"
)
