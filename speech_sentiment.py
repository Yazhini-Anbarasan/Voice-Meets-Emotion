"""
South Indian Speech Sentiment Analyzer — Streamlit Web App

Installation:
   pip install openai-whisper transformers torch streamlit audio-recorder-streamlit deep-translator scipy numpy

Run:
   streamlit run speech_sentiment.py
"""

import os
import sys
import io
import wave
import subprocess
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ── Auto-detect FFmpeg on Windows ──────────────────────────────────────────
def setup_ffmpeg():
    """Find ffmpeg and add its directory to PATH so Whisper can use it."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return True
    except FileNotFoundError:
        pass

    search_dirs = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
        "C:\\ffmpeg",
        os.path.join(os.environ.get("ProgramFiles", ""), "ffmpeg"),
    ]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            if "ffmpeg.exe" in files:
                os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                return True
    return False


if sys.platform == "win32":
    setup_ffmpeg()

# ── Imports (after ffmpeg setup) ───────────────────────────────────────────
import streamlit as st
import whisper
import torch
import numpy as np
from transformers import pipeline
from deep_translator import GoogleTranslator
from audio_recorder_streamlit import audio_recorder

# ── Language map ───────────────────────────────────────────────────────────
LANG_MAP = {
    "ta": "Tamil", "ml": "Malayalam", "te": "Telugu",
    "kn": "Kannada", "en": "English", "hi": "Hindi",
    "bn": "Bengali", "gu": "Gujarati", "mr": "Marathi",
    "ur": "Urdu", "pa": "Punjabi",
}

LANG_FLAGS = {
    "Tamil": "🇮🇳", "Malayalam": "🇮🇳", "Telugu": "🇮🇳",
    "Kannada": "🇮🇳", "English": "🇬🇧", "Hindi": "🇮🇳",
}

SENTIMENT_EMOJI = {
    "Positive": "😊", "Negative": "😞", "Neutral": "😐",
}

SENTIMENT_COLOR = {
    "Positive": "#00c853", "Negative": "#ff1744", "Neutral": "#ffc107",
}


# ── Cached model loading (runs only once) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return whisper.load_model("medium")  # 'medium' is much better for South Indian languages


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )


# ── Processing functions ──────────────────────────────────────────────────
def transcribe_audio(audio_path, model, language=None):
    """Transcribe audio file and return result dict."""
    # Use fp16 only if CUDA is available
    use_fp16 = torch.cuda.is_available()
    opts = {"fp16": use_fp16}
    if language and language != "auto":
        opts["language"] = language  # Force specific language for accuracy
    result = model.transcribe(audio_path, **opts)
    return result


def convert_audio_bytes_to_wav(audio_bytes):
    """Convert raw audio recorder bytes to a proper WAV file, return path."""
    try:
        # The audio_recorder returns WAV bytes — write them directly
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
    except Exception:
        # Fallback: write raw bytes
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name


def translate_text(text, source_lang):
    """Translate text to English using Google Translate."""
    try:
        translated = GoogleTranslator(source=source_lang, target="en").translate(text)
        return translated
    except Exception as e:
        return f"[Translation error: {e}]"


def analyze_sentiment(text, model):
    """Run sentiment analysis and return label + score."""
    if not text or text.startswith("["):
        return "Neutral", 0.5
    result = model(text)
    label = result[0]["label"].lower()
    score = result[0]["score"]
    if "positive" in label:
        return "Positive", score
    elif "negative" in label:
        return "Negative", score
    return "Neutral", score


# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="South Indian Speech Sentiment Analyzer",
    page_icon="🎙️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #888;
        font-size: 1.05rem;
        margin-top: 0;
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .result-card .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #999;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    .result-card .value {
        font-size: 1.15rem;
        color: #f0f0f0;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Sentiment badge */
    .sentiment-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        font-size: 1.6rem;
        font-weight: 700;
        padding: 0.8rem 1.6rem;
        border-radius: 50px;
        margin-top: 0.4rem;
    }

    /* Confidence bar */
    .confidence-container {
        margin-top: 0.6rem;
    }
    .confidence-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
        margin-top: 0.3rem;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 1.5rem 0;
    }

    /* Input section */
    .input-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* Pipeline */
    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 1rem;
        flex-wrap: wrap;
    }
    .pipeline-step {
        background: rgba(102,126,234,0.15);
        color: #667eea;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(102,126,234,0.25);
    }
    .pipeline-arrow {
        color: #555;
        font-size: 1.1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #555;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎙️ South Indian Speech Sentiment Analyzer</h1>
    <p>Analyze speech sentiment in Tamil · Malayalam · Telugu and more</p>
</div>
""", unsafe_allow_html=True)

# ── Pipeline Visualization ────────────────────────────────────────────────
st.markdown("""
<div class="pipeline">
    <span class="pipeline-step">🎤 Speech Input</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">📝 Whisper STT</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">🌐 Translation</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">🧠 Sentiment</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">📊 Result</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────
with st.spinner("⏳ Loading AI Models (first run takes a moment)..."):
    whisper_model = load_whisper_model()
    sentiment_pipeline = load_sentiment_model()

# ── Input Section ─────────────────────────────────────────────────────────
st.markdown("### 🎧 Choose Input Method")

# Language selector — lets the user tell Whisper what language they are speaking
lang_options = {
    "🔍 Auto Detect": "auto",
    "🇮🇳 Tamil": "ta",
    "🇮🇳 Malayalam": "ml",
    "🇮🇳 Telugu": "te",
    "🇮🇳 Kannada": "kn",
    "🇮🇳 Hindi": "hi",
    "🇬🇧 English": "en",
}
selected_lang_label = st.selectbox(
    "🌐 Select the language you are speaking:",
    options=list(lang_options.keys()),
    index=0,
    help="Selecting the correct language greatly improves transcription accuracy for South Indian languages.",
)
selected_lang_code = lang_options[selected_lang_label]

tab_mic, tab_file = st.tabs(["🎤 Record from Microphone", "📁 Upload Audio File"])

audio_bytes = None
audio_source = None

with tab_mic:
    st.markdown("Click the microphone button below to start recording:")
    recorded_audio = audio_recorder(
        text="",
        recording_color="#667eea",
        neutral_color="#888",
        icon_size="2x",
        pause_threshold=3.0,
    )
    if recorded_audio:
        audio_bytes = recorded_audio
        audio_source = "mic"
        st.audio(recorded_audio, format="audio/wav")
        st.success("✅ Audio recorded! Click **Analyze** below.")

with tab_file:
    uploaded_file = st.file_uploader(
        "Upload a .wav or .mp3 audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Supports Tamil, Malayalam, Telugu, and other languages",
    )
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        audio_source = "file"
        st.audio(audio_bytes, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        st.success(f"✅ File **{uploaded_file.name}** loaded! Click **Analyze** below.")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Analyze Button ────────────────────────────────────────────────────────
if audio_bytes:
    if st.button("🚀  Analyze Sentiment", use_container_width=True, type="primary"):

        # Save audio to a temp file for Whisper
        if audio_source == "mic":
            tmp_path = convert_audio_bytes_to_wav(audio_bytes)
        else:
            suffix = f".{uploaded_file.name.split('.')[-1]}" if audio_source == "file" else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

        try:
            # Step 1 — Transcribe (pass the language hint for better accuracy)
            with st.spinner("🗣️  Transcribing speech (this may take a moment)..."):
                result = transcribe_audio(tmp_path, whisper_model, language=selected_lang_code)
                original_text = result.get("text", "").strip()
                lang_code = result.get("language", "unknown")
                # If user selected a specific language, use that as the detected language
                if selected_lang_code != "auto":
                    lang_code = selected_lang_code
                lang_name = LANG_MAP.get(lang_code, lang_code.capitalize())

            # Step 2 — Translate
            english_text = original_text
            if lang_code != "en" and original_text:
                with st.spinner("🌐 Translating to English..."):
                    english_text = translate_text(original_text, lang_code)

            # Step 3 — Sentiment
            with st.spinner("🧠 Analyzing sentiment..."):
                sentiment, confidence = analyze_sentiment(english_text, sentiment_pipeline)

            # ── Display Results ───────────────────────────────────────
            st.markdown("### 📊 Analysis Results")

            # Transcribed text card
            st.markdown(f"""
            <div class="result-card">
                <div class="label">📝 Transcribed Text (Original)</div>
                <div class="value">{original_text if original_text else "[No speech detected]"}</div>
            </div>
            """, unsafe_allow_html=True)

            # Language + Translation row
            col1, col2 = st.columns(2)
            with col1:
                flag = LANG_FLAGS.get(lang_name, "🌍")
                st.markdown(f"""
                <div class="result-card">
                    <div class="label">🌍 Detected Language</div>
                    <div class="value">{flag} {lang_name}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if lang_code != "en":
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="label">🔄 English Translation</div>
                        <div class="value">{english_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="label">🔄 Translation</div>
                        <div class="value" style="color:#666;">Not needed (already English)</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Sentiment result card
            emoji = SENTIMENT_EMOJI.get(sentiment, "😐")
            color = SENTIMENT_COLOR.get(sentiment, "#ffc107")
            conf_pct = int(confidence * 100)

            st.markdown(f"""
            <div class="result-card" style="text-align:center;">
                <div class="label">💬 Sentiment Result</div>
                <div class="sentiment-badge" style="color:{color}; background:rgba(255,255,255,0.04);">
                    {emoji} {sentiment}
                </div>
                <div class="confidence-container">
                    <div class="label" style="margin-top:1rem;">Confidence: {conf_pct}%</div>
                    <div class="confidence-bg">
                        <div class="confidence-fill" style="width:{conf_pct}%; background:{color};"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
            st.info("💡 Make sure FFmpeg is installed and the audio file is valid.")

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

else:
    st.info("👆 Record audio or upload a file above to get started!")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    Built with ❤️ using Whisper · HuggingFace Transformers · Google Translate · Streamlit<br>
    South Indian Speech Sentiment Analyzer © 2026
</div>
""", unsafe_allow_html=True)
