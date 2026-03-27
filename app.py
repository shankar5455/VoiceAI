"""
app.py — VoiceAI: Speech AI Web App with Transcription, TTS, Voice Cloning,
         Voice Conversion and Training.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import time
from typing import Optional, Dict, Any, Tuple

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

# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="VoiceAI – Speech AI Suite",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Styling
# ============================================================================

def apply_custom_css():
    """Apply clean, minimal custom CSS styling"""
    st.markdown(
        """
        <style>
            /* Base */
            .stApp { background: #f8fafc; }
            
            /* Cards */
            .card {
                background: white;
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                border: 1px solid #e2e8f0;
                transition: all 0.2s ease;
            }
            .card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                border-color: #cbd5e1;
            }
            
            /* Hero */
            .hero {
                text-align: center;
                padding: 32px 24px;
                margin-bottom: 32px;
                background: white;
                border-radius: 24px;
                border: 1px solid #e2e8f0;
            }
            .hero h1 {
                font-size: 2.5rem;
                font-weight: 600;
                margin: 0;
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .hero p {
                font-size: 1rem;
                color: #64748b;
                margin-top: 8px;
            }
            
            /* Badges */
            .badge-container {
                margin-top: 16px;
                display: flex;
                justify-content: center;
                gap: 8px;
                flex-wrap: wrap;
            }
            .badge {
                background: #f1f5f9;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.75rem;
                color: #475569;
                font-weight: 500;
            }
            
            /* Metrics */
            .metric {
                background: #f8fafc;
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                border: 1px solid #e2e8f0;
            }
            .metric-value {
                font-size: 28px;
                font-weight: 600;
                color: #0f172a;
                margin: 8px 0;
            }
            .metric-label {
                font-size: 13px;
                color: #64748b;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            /* Speaker badges */
            .speaker-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 500;
                margin-right: 8px;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 4px;
                background: #f1f5f9;
                border-radius: 12px;
                padding: 4px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 8px;
                padding: 8px 20px;
                font-weight: 500;
                color: #475569;
            }
            .stTabs [aria-selected="true"] {
                background: white;
                color: #0f172a;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            }
            
            /* Buttons */
            .stButton > button {
                background: #0f172a;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 20px;
                font-weight: 500;
                transition: all 0.2s;
            }
            .stButton > button:hover {
                background: #1e293b;
                transform: translateY(-1px);
            }
            
            /* Features */
            .feature {
                background: #f8fafc;
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                border: 1px solid #e2e8f0;
                transition: all 0.2s;
            }
            .feature:hover {
                background: white;
                border-color: #cbd5e1;
            }
            .feature-icon { font-size: 28px; margin-bottom: 8px; }
            .feature-title { font-weight: 600; color: #0f172a; margin: 8px 0 4px 0; }
            .feature-desc { font-size: 12px; color: #64748b; }
            
            /* Footer */
            .footer {
                text-align: center;
                padding: 24px;
                margin-top: 40px;
                border-top: 1px solid #e2e8f0;
                color: #64748b;
                font-size: 13px;
            }
            
            /* Misc */
            hr { margin: 24px 0; border: none; border-top: 1px solid #e2e8f0; }
            audio { border-radius: 8px; width: 100%; }
            .stAlert { border-radius: 10px; border-left-width: 3px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# Session State Management
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "last_transcript": "",
        "last_full_text": "",
        "audio_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================================================
# Sidebar Components
# ============================================================================

def render_sidebar() -> Tuple[str, Optional[int], Optional[str]]:
    """Render sidebar and return configuration values"""
    with st.sidebar:
        st.markdown("### 🎙️ VoiceAI")
        st.markdown("---")
        
        # ASR Model
        st.markdown("**ASR Model**")
        model_size = st.selectbox(
            "Whisper model",
            options=["tiny", "base", "small", "medium"],
            index=1,
        )
        
        # Diarization
        st.markdown("**Diarization**")
        auto_detect = st.checkbox("Auto-detect speakers", value=True)
        n_speakers = None if auto_detect else st.slider(
            "Number of speakers", min_value=2, max_value=6, value=2
        )
        
        # Language
        st.markdown("**Language**")
        language_options = {
            "Auto-detect": None,
            "English": "en",
            "French": "fr",
            "Spanish": "es",
            "German": "de",
            "Hindi": "hi",
            "Arabic": "ar",
        }
        lang_label = st.selectbox("Language", list(language_options.keys()))
        language = language_options[lang_label]
        
        st.markdown("---")
        st.caption("v2.0 • Powered by Whisper")
    
    return model_size, n_speakers, language

# ============================================================================
# Hero Section
# ============================================================================

def render_hero():
    """Render hero section"""
    st.markdown(
        """
        <div class="hero">
            <h1>VoiceAI</h1>
            <p>Speech AI suite with transcription, synthesis, and voice transformation</p>
            <div class="badge-container">
                <span class="badge">🎯 Transcription</span>
                <span class="badge">🗣️ Diarization</span>
                <span class="badge">🔊 TTS</span>
                <span class="badge">🎭 Voice Cloning</span>
                <span class="badge">🔄 Voice Conversion</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# Transcription Tab
# ============================================================================

def render_transcription_tab(model_size: str, n_speakers: Optional[int], language: Optional[str]):
    """Render transcription tab with audio input and processing"""
    
    
    # Audio input
    audio_path = render_audio_input()
    
    # Process button
    if st.button("Process Audio", type="primary"):
        if audio_path is None:
            st.warning("Please upload or record audio first")
            return
        
        process_audio(audio_path, model_size, n_speakers, language)

def render_audio_input() -> Optional[str]:
    """Render audio upload and recording interface"""
    tab_upload, tab_record = st.tabs(["Upload Audio", "Record Audio"])
    audio_path = None
    
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            st.audio(uploaded_file, format=uploaded_file.type)
            audio_path = save_uploaded_file(uploaded_file)
    
    with tab_record:
        try:
            from audio_recorder_streamlit import audio_recorder
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#ef4444",
                neutral_color="#64748b",
                icon_name="microphone",
                icon_size="2x",
            )
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                audio_path = save_audio_bytes(audio_bytes, suffix=".wav")
        except ImportError:
            st.info("Install `audio-recorder-streamlit` for microphone recording")
    
    return audio_path

def process_audio(audio_path: str, model_size: str, n_speakers: Optional[int], language: Optional[str]):
    """Process audio through transcription pipeline"""
    wav_path = None
    
    try:
        with st.spinner("Processing..."):
            # Convert and load
            wav_path = convert_to_wav(audio_path)
            
            # Transcribe
            model, device = load_model(model_size)
            asr_result = transcribe_audio(model, wav_path, language=language)
            
            # Diarize
            audio_array, sr = load_audio(wav_path, sr=16000)
            diar_segments = perform_diarization(audio_array, sr, n_speakers=n_speakers)
            
            # Merge
            merged = merge_transcript_diarization(asr_result["segments"], diar_segments)
        
        if not merged:
            st.error("No speech detected")
            return
        
        # Save to session
        transcript_text = create_transcript_text(merged)
        st.session_state["last_transcript"] = transcript_text
        st.session_state["last_full_text"] = asr_result.get("full_text", "")
        
        # Display results
        display_transcript_results(merged, transcript_text)
        
    except Exception as exc:
        st.error(f"Error: {exc}")
    finally:
        if wav_path:
            cleanup_temp_file(wav_path)
        if audio_path:
            cleanup_temp_file(audio_path)

def display_transcript_results(merged: list, transcript_text: str):
    """Display transcript results with speaker information"""
    # Transcript
    st.markdown("### Transcript")
    html = render_colored_transcript(merged)
    st.markdown(html, unsafe_allow_html=True)
    
    # Speaker legend
    speakers = sorted({s["speaker"] for s in merged})
    legend_html = "".join(
        f'<span class="speaker-badge" style="background:{get_speaker_color(sp)};color:#fff;">{sp}</span>'
        for sp in speakers
    )
    st.markdown(f"**Speakers:** {legend_html}", unsafe_allow_html=True)
    
    # Statistics
    st.markdown("### Speaker Statistics")
    stats = speaker_stats(merged)
    total_time = sum(stats.values())
    
    cols = st.columns(len(stats))
    for col, (speaker, duration) in zip(cols, stats.items()):
        pct = (duration / total_time * 100) if total_time > 0 else 0
        col.markdown(
            f"""
            <div class="metric">
                <div class="metric-label">{speaker}</div>
                <div class="metric-value">{format_timestamp(duration)}</div>
                <div class="metric-label">{pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Download
    st.download_button(
        label="Download Transcript",
        data=transcript_text,
        file_name="transcript.txt",
        mime="text/plain",
    )

# ============================================================================
# TTS Tab
# ============================================================================

def render_tts_tab():
    """Render Text-to-Speech tab"""
    default_text = st.session_state.get("last_full_text", "")
    
    tts_text = st.text_area(
        "Text to synthesize",
        value=default_text,
        height=120,
        placeholder="Enter text to convert to speech...",
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        backend = st.selectbox("Engine", options=["auto", "gtts", "pyttsx3"], index=0)
    
    with col2:
        lang_options = {
            "English": "en", "French": "fr", "Spanish": "es",
            "German": "de", "Hindi": "hi", "Arabic": "ar",
        }
        lang_label = st.selectbox("Language", list(lang_options.keys()))
        language = lang_options[lang_label]
    
    with col3:
        rate = st.slider("Speech rate", min_value=80, max_value=300, value=150, step=10)
    
    if st.button("Generate Speech", type="primary"):
        if not tts_text.strip():
            st.warning("Please enter text")
        else:
            with st.spinner("Generating..."):
                try:
                    audio_bytes, ext = synthesize_speech(
                        tts_text, backend=backend, lang=language, rate=rate
                    )
                    mime = "audio/wav" if ext == ".wav" else "audio/mpeg"
                    st.success("Speech generated")
                    st.audio(audio_bytes, format=mime)
                    st.download_button(
                        label="Download Audio",
                        data=audio_bytes,
                        file_name=f"speech{ext}",
                        mime=mime,
                    )
                except Exception as exc:
                    st.error(f"Error: {exc}")

# ============================================================================
# Voice Cloning Tab
# ============================================================================

def render_cloning_tab():
    """Render Voice Cloning tab"""
    st.markdown(
        """
        <div class="card">
            <strong>How it works:</strong> Upload a clear reference voice sample (at least 5 seconds)
            and enter the text you want spoken in that voice.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    ref_file = st.file_uploader(
        "Reference voice sample",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="clone_ref",
    )
    if ref_file:
        st.audio(ref_file, format=ref_file.type)
    
    text = st.text_area(
        "Text to synthesize",
        height=100,
        placeholder="Enter text...",
        key="clone_text",
    )
    
    lang_options = {
        "English": "en", "French": "fr", "Spanish": "es",
        "German": "de", "Hindi": "hi", "Arabic": "ar",
    }
    lang_label = st.selectbox("Language", list(lang_options.keys()), key="clone_lang")
    language = lang_options[lang_label]
    
    if st.button("Clone Voice", type="primary"):
        if ref_file is None:
            st.warning("Please upload a reference voice")
        elif not text.strip():
            st.warning("Please enter text")
        else:
            ref_path = None
            try:
                ref_path = save_uploaded_file(ref_file)
                with st.spinner("Cloning voice..."):
                    cloned_bytes = clone_voice(text, ref_path, lang=language)
                st.success("Voice cloned successfully")
                st.audio(cloned_bytes, format="audio/wav")
                st.download_button(
                    label="Download Audio",
                    data=cloned_bytes,
                    file_name="cloned_voice.wav",
                    mime="audio/wav",
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
            finally:
                if ref_path:
                    cleanup_temp_file(ref_path)

# ============================================================================
# Voice Conversion Tab
# ============================================================================

def render_conversion_tab():
    """Render Voice Conversion tab"""
    src_file = st.file_uploader(
        "Source audio (voice to convert)",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="vc_src",
    )
    if src_file:
        st.audio(src_file, format=src_file.type)
    
    tgt_file = st.file_uploader(
        "Target voice (optional)",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="vc_tgt",
        help="Upload to automatically match pitch",
    )
    if tgt_file:
        st.audio(tgt_file, format=tgt_file.type)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pitch_shift = st.slider(
            "Pitch shift (semitones)",
            min_value=-24, max_value=24, value=0, step=1,
            disabled=tgt_file is not None,
        )
    
    with col2:
        speed_factor = st.slider(
            "Speed factor", min_value=0.5, max_value=2.0, value=1.0, step=0.1
        )
    
    if st.button("Convert Voice", type="primary"):
        if src_file is None:
            st.warning("Please upload source audio")
        else:
            src_path = tgt_path = None
            try:
                src_path = save_uploaded_file(src_file)
                if tgt_file is not None:
                    tgt_path = save_uploaded_file(tgt_file)
                
                with st.spinner("Converting..."):
                    converted_bytes = convert_voice(
                        source_audio_path=src_path,
                        target_audio_path=tgt_path,
                        pitch_shift_semitones=float(pitch_shift),
                        speed_factor=float(speed_factor),
                    )
                
                st.success("Conversion complete")
                st.audio(converted_bytes, format="audio/wav")
                st.download_button(
                    label="Download Audio",
                    data=converted_bytes,
                    file_name="converted_voice.wav",
                    mime="audio/wav",
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
            finally:
                if src_path:
                    cleanup_temp_file(src_path)
                if tgt_path:
                    cleanup_temp_file(tgt_path)

# ============================================================================
# Training Tab
# ============================================================================

def render_training_tab():
    """Render Training tab"""
    st.markdown(
        """
        <div class="card">
            <strong>Custom Model Training</strong><br>
            Upload your dataset to train custom speech models. Requires a ZIP file with audio files
            and transcripts in the correct format.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Dataset upload
    st.markdown("### Dataset")
    dataset_zip = st.file_uploader("Upload ZIP archive", type=["zip"], key="training_zip")
    
    if dataset_zip is not None:
        validate_dataset_upload(dataset_zip)
    
    # Configuration
    st.markdown("### Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox("Model type", options=["tts", "asr"])
    with col2:
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100, step=10)
    with col3:
        batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=16, step=4)
    
    config = generate_training_config(
        model_type=model_type,
        epochs=int(epochs),
        batch_size=int(batch_size),
    )
    
    with st.expander("View configuration"):
        st.code(config, language="json")
        st.download_button(
            label="Download config",
            data=config,
            file_name="training_config.json",
            mime="application/json",
        )
    
    # Simulation
    st.markdown("### Training Simulation")
    demo_epochs = st.slider("Simulation epochs", min_value=5, max_value=30, value=10)
    
    if st.button("Start Simulation", type="primary"):
        run_training_simulation(demo_epochs)

def validate_dataset_upload(zip_file):
    """Validate uploaded dataset ZIP"""
    zip_path = None
    try:
        zip_path = save_uploaded_file(zip_file)
        summary = validate_dataset(zip_path)
        
        col1, col2, col3 = st.columns(3)
        metrics = [
            ("Total Files", summary['total_files']),
            ("Audio Files", summary['audio_files']),
            ("Transcripts", summary['text_files']),
        ]
        for col, (label, value) in zip([col1, col2, col3], metrics):
            col.markdown(
                f"""
                <div class="metric">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        if summary["warnings"]:
            for w in summary["warnings"]:
                st.warning(w)
        else:
            st.success("Dataset ready for training")
    
    except Exception as exc:
        st.error(f"Validation failed: {exc}")
    finally:
        if zip_path:
            cleanup_temp_file(zip_path)

def run_training_simulation(epochs: int):
    """Run training simulation with live updates"""
    progress_bar = st.progress(0)
    status = st.empty()
    chart = st.empty()
    
    losses = []
    accuracies = []
    
    for ep in range(1, epochs + 1):
        step = mock_training_step(ep, epochs)
        losses.append(step["loss"])
        accuracies.append(step["accuracy"])
        
        progress_bar.progress(int(step["progress_pct"]))
        status.text(
            f"Epoch {ep}/{epochs} | Loss: {step['loss']:.4f} | Acc: {step['accuracy']:.4f}"
        )
        
        chart_data = pd.DataFrame({
            "Loss": losses,
            "Accuracy": accuracies,
        }, index=range(1, ep + 1))
        chart.line_chart(chart_data)
        
        time.sleep(0.2)
    
    progress_bar.progress(100)
    status.success(
        f"Complete! Final loss: {losses[-1]:.4f}, accuracy: {accuracies[-1]:.4f}"
    )

# ============================================================================
# Footer
# ============================================================================

def render_footer():
    """Render footer section"""
    st.markdown(
        """
        <div class="footer">
            VoiceAI — Speech AI Suite • Powered by Whisper & Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point"""
    # Initialize
    apply_custom_css()
    init_session_state()
    
    # Render sidebar and get config
    model_size, n_speakers, language = render_sidebar()
    
    # Render hero
    render_hero()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Transcription", "Text-to-Speech", "Voice Cloning", "Voice Conversion", "Training"
    ])
    
    with tab1:
        render_transcription_tab(model_size, n_speakers, language)
    
    with tab2:
        render_tts_tab()
    
    with tab3:
        render_cloning_tab()
    
    with tab4:
        render_conversion_tab()
    
    with tab5:
        render_training_tab()
    
    # Footer
    render_footer()

# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    main()