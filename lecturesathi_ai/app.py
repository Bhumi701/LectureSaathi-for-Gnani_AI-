import io
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from google import genai
import streamlit.components.v1 as components

load_dotenv()

STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
STT_MODEL = "prisma-v3.0"
TTS_MODEL = "timbre-v2.5"
MAX_CHUNK_MS = 25_000

AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

STOPWORDS = set("""
a an the and or but if then is are was were be been being to of in on at for from
by with as into about this that these those it its we you your they them their
he she his her i me my our us what which who when where why how can could should
would will may might do does did have has had not no yes so than too very also
there here their because while during after before over under between through
एक और या लेकिन अगर तो है हैं था थी थे हो के की का को में पर से तक लिए यह वह
आप हम मैं क्या कैसे क्यों कब कहाँ इस उस इन उन भी नहीं तो फिर साथ लिए
""".split())

def get_api_key():
    key = os.getenv("GNANI_API_KEY") or os.getenv("GNANI_API_KEY_ID")
    if not key:
        try:
            if "GNANI_API_KEY" in st.secrets:
                key = st.secrets["GNANI_API_KEY"]
            elif "GNANI_API_KEY_ID" in st.secrets:
                key = st.secrets["GNANI_API_KEY_ID"]
        except Exception:
            pass
    return key

# --- Groq Client for AI Tutor (Qwen 3.8) ---
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    try:
        if not api_key and "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

def call_groq_tutor(prompt, system_prompt="You are a helpful AI lecture assistant."):
    client = get_groq_client()
    if not client:
        return None
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"Groq Error: {exc}"

# --- Google Gemini Client for Mind Map ---
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    try:
        if not api_key and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        elif not api_key and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def call_gemini_mindmap(prompt, system_prompt="You are an expert visual diagram designer."):
    client = get_gemini_client()
    if not client:
        return None
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as exc:
        return f"Gemini API Error: {exc}"

def st_mermaid(mermaid_code):
    """Renders Mermaid code as an interactive graphical diagram using HTML/JS component."""
    html_code = f"""
    <div class="mermaid" style="display: flex; justify-content: center; background: white; padding: 20px; border-radius: 10px;">
        {mermaid_code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    """
    components.html(html_code, height=550, scrolling=True)

def api_error(response):
    try:
        body = response.json()
        return body.get("message") or body.get("error", {}).get("message") or str(body)
    except Exception:
        return response.text[:500]

def transcribe_one_chunk(wav_bytes, api_key, mode, topic=""):
    headers = {"X-API-Key-ID": api_key}
    data = {
        "model": STT_MODEL,
        "language_code": "indic-auto",
        "format": mode,
    }
    if topic.strip():
        data["topic"] = topic.strip()[:1600]

    files = {"audio_file": ("lecture_chunk.wav", wav_bytes, "audio/wav")}
    try:
        response = requests.post(
            STT_URL, headers=headers, data=data, files=files, timeout=120
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach Gnani Prisma: {exc}") from exc

    if not response.ok:
        raise RuntimeError(
            f"Prisma returned HTTP {response.status_code}: {api_error(response)}"
        )

    payload = response.json()
    if payload.get("success") is False:
        raise RuntimeError(f"Prisma error: {api_error(response)}")

    transcript = payload.get("transcript") or payload.get("text")
    if not transcript:
        raise RuntimeError(f"Prisma response did not contain transcript: {payload}")

    return {
        "text": transcript.strip(),
        "language": payload.get("detected_language_code", "unknown"),
        "request_id": payload.get("request_id", ""),
        "model": payload.get("model", STT_MODEL),
    }

def audio_from_upload(uploaded_file):
    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix == "mp4":
        fmt = "mp4"
    elif suffix == "m4a":
        fmt = "m4a"
    else:
        fmt = suffix or "wav"
    try:
        audio = AudioSegment.from_file(io.BytesIO(raw), format=fmt)
    except Exception as exc:
        raise RuntimeError(
            "Unable to read this file. Try MP3, WAV, M4A or MP4. "
            "For MP4, ensure the file contains an audio track."
        ) from exc
    if audio.channels > 1:
        audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000).set_sample_width(2)
    return audio

def transcribe_audio(audio, api_key, mode, topic, progress_callback=None):
    duration = len(audio)
    chunks = []
    for start in range(0, duration, MAX_CHUNK_MS):
        end = min(start + MAX_CHUNK_MS, duration)
        chunk = audio[start:end]
        buffer = io.BytesIO()
        chunk.export(buffer, format="wav")
        result = transcribe_one_chunk(buffer.getvalue(), api_key, mode, topic)
        chunks.append({
            "start_ms": start,
            "end_ms": end,
            "start": format_time(start),
            "end": format_time(end),
            "text": result["text"],
            "language": result["language"],
            "request_id": result["request_id"],
        })
        if progress_callback:
            progress_callback(min(end / max(duration, 1), 1.0))
    return chunks

def format_time(ms):
    total_seconds = int(ms / 1000)
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"

def get_transcript(chunks):
    return "\n".join(c["text"] for c in chunks if c["text"])

def synthesize_speech(text, api_key, language, voice):
    payload = {
        "text": text[:1800],
        "model": TTS_MODEL,
        "language": language,
        "voice": voice,
        "speed": 1.0,
        "audio_config": {
            "encoding": "linear_pcm",
            "container": "wav",
            "num_channels": 1,
            "sample_rate": 48000,
            "sample_width": 2,
        },
    }
    try:
        response = requests.post(
            TTS_URL,
            headers={"X-API-Key-ID": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach Gnani Timbre: {exc}") from exc

    if not response.ok:
        raise RuntimeError(
            f"Timbre returned HTTP {response.status_code}: {api_error(response)}"
        )
    content_type = response.headers.get("content-type", "")
    if "json" in content_type.lower():
        raise RuntimeError(f"Timbre returned JSON instead of audio: {api_error(response)}")
    return response.content

def reset_session():
    for key in ["chunks", "transcript", "language", "filename", "audio_duration", "mindmap_result"]:
        st.session_state.pop(key, None)

st.set_page_config(
    page_title="LectureSathi AI",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
.main {background: #f7f8fc;}
.block-container {padding-top: 2rem; max-width: 1180px;}
.hero {
  padding: 1.4rem 1.6rem; border-radius: 18px;
  background: linear-gradient(120deg,#102a43,#245b83);
  color: white; margin-bottom: 1.2rem;
}
.hero h1 {color:white; margin:0; font-size:2.3rem;}
.hero p {color:#e4edf5; margin:.35rem 0 0 0;}
</style>
<div class="hero">
  <h1>🎓 LectureSathi AI</h1>
  <p>Your multilingual lecture companion — powered by Gnani Prisma, Timbre, Groq Qwen 3.8 & Google Gemini.</p>
</div>
""", unsafe_allow_html=True)

api_key = get_api_key()
if not api_key:
    st.error("Gnani API key not found. Add GNANI_API_KEY to your .env file or Streamlit Secrets.")
    st.code("GNANI_API_KEY=your_gnani_api_key")
    st.stop()

with st.sidebar:
    st.header("Lecture settings")
    mode = st.selectbox(
        "Prisma transcript style",
        ["transcribe", "codemix", "verbatim", "transliterate"],
        help="Prisma v3.0 supports all four styles.",
    )
    topic = st.text_input(
        "Lecture topic (optional)",
        placeholder="e.g. C Programming, DBMS, AI",
    )
    st.divider()
    st.markdown("**Models used**")
    st.write("🎙️ Prisma v3.0 · Speech-to-text")
    st.write("🧠 Qwen 3.8 27B (Groq) · AI Tutor")
    st.write("📊 Gemini 2.5 Flash · Graphical Mind Map")
    st.write("🔊 Timbre v2.5 · Text-to-speech")

st.subheader("1 · Upload a lecture")
uploaded = st.file_uploader(
    "Upload an audio or video file",
    type=["wav", "mp3", "m4a", "ogg", "flac", "aac", "mp4"],
    help="Upload your lecture audio or short recording.",
)

col_a, col_b = st.columns([1, 4])
with col_a:
    process_clicked = st.button("Transcribe lecture", type="primary", use_container_width=True, disabled=uploaded is None)
with col_b:
    st.caption("LectureSathi automatically segments audio clips for Gnani Prisma REST API.")

if process_clicked and uploaded is not None:
    reset_session()
    try:
        with st.spinner("Preparing audio..."):
            audio = audio_from_upload(uploaded)
        total_chunks = max(1, (len(audio) + MAX_CHUNK_MS - 1) // MAX_CHUNK_MS)
        progress = st.progress(0.0, text=f"Sending audio segments to Gnani Prisma...")
        chunks = transcribe_audio(
            audio, api_key, mode, topic,
            progress_callback=lambda value: progress.progress(value, text=f"Transcribing lecture · {int(value*100)}%"),
        )
        transcript = get_transcript(chunks)
        st.session_state["chunks"] = chunks
        st.session_state["transcript"] = transcript
        st.session_state["language"] = next((c["language"] for c in chunks if c["language"] != "unknown"), "unknown")
        st.session_state["filename"] = uploaded.name
        st.session_state["audio_duration"] = len(audio)
        st.success("Lecture transcribed successfully with Gnani Prisma.")
    except Exception as exc:
        st.error(str(exc))

if "transcript" not in st.session_state:
    st.info("Upload a lecture and select **Transcribe lecture** to begin.")
    st.stop()

chunks = st.session_state["chunks"]
transcript = st.session_state["transcript"]
language = st.session_state["language"]

m1, m2, m3 = st.columns(3)
m1.metric("Detected language", language)
m2.metric("Audio duration", format_time(st.session_state["audio_duration"]))
m3.metric("Transcript segments", len(chunks))

tab1, tab2, tab3, tab4 = st.tabs(["Transcript", "Ask the lecture (AI Tutor)", "Graphical Mind Map & Notes", "Listen back"])

with tab1:
    st.subheader("Lecture transcript")
    st.caption(f"{st.session_state['filename']} · Prisma v3.0 · {mode}")
    st.download_button(
        "Download transcript (.txt)",
        transcript,
        file_name="lecturesathi_transcript.txt",
        mime="text/plain",
    )
    for chunk in chunks:
        with st.expander(f"{chunk['start']} – {chunk['end']} · {chunk['language']}"):
            st.write(chunk["text"])

with tab2:
    st.subheader("🤖 Ask your lecture (AI Tutor via Qwen 3.8)")
    st.caption("Ask questions about the lecture. Qwen will analyze the transcript and explain clearly with examples.")
    
    question = st.text_input(
        "What are you trying to understand from this lecture?",
        placeholder="e.g. Explain this concept simply with a real-life analogy...",
        key="groq_lecture_question",
    )
    
    if st.button("Ask AI Tutor", type="primary", key="ask_groq_btn"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Analyzing lecture context with Qwen 3.8 (Groq)..."):
                prompt = f"""You are LectureSathi AI, an expert educational tutor. 
Here is the full transcript of the lecture uploaded by the student:
---
{transcript}
---

Student Question / Request: {question}

Please provide:
1. A clear, easy-to-understand explanation answering the question.
2. Real-life examples or simpler analogies.
3. Direct references or quotes from the transcript where appropriate.
4. 2 relevant follow-up questions the student might want to consider."""
                
                answer = call_groq_tutor(prompt, system_prompt="You are a clear, patient academic tutor.")
                if answer:
                    st.markdown("### 💡 AI Tutor Explanation")
                    st.markdown(answer)
                else:
                    st.error("Could not generate a response. Please verify your GROQ_API_KEY.")

with tab3:
    st.subheader("📊 Visual Graphical Mind Map & Notes")
    st.caption("Generated as a rendered flowchart diagram using Google Gemini.")
    
    if st.button("Generate Graphical Mind Map", type="primary", key="gen_mindmap_btn"):
        with st.spinner("Generating visual mind map & structured notes with Google Gemini..."):
            prompt = f"""Analyze the following lecture transcript and create two things:
1. A valid **Mermaid.js flowchart** code block starting with `graph TD` representing the core theme and branching sub-concepts visually. 
   - Use clean alphanumeric node IDs (like A, B, C1, C2).
   - Put node labels inside quotes, e.g., A["Core Topic"] --> B["Subtopic"].
   - Do NOT include syntax errors or special characters unescaped. Wrap the code block strictly in ```mermaid ... ```.
2. Structured revision notes below it with key definitions and quick takeaways.

Transcript:
---
{transcript}
---
"""
            result_output = call_gemini_mindmap(prompt, system_prompt="You are an expert visual diagram designer and study assistant.")
            if result_output:
                st.session_state["mindmap_result"] = result_output
            else:
                st.error("Failed to generate mind map. Check your GEMINI_API_KEY.")

    if "mindmap_result" in st.session_state:
        output_text = st.session_state["mindmap_result"]
        
        if "```mermaid" in output_text:
            parts = output_text.split("```mermaid")
            intro_text = parts[0]
            mermaid_and_rest = parts[1].split("```")
            mermaid_code = mermaid_and_rest[0].strip()
            remaining_text = mermaid_and_rest[1] if len(mermaid_and_rest) > 1 else ""
            
            st.markdown(intro_text)
            st.subheader("🗺️ Rendered Graphical Flowchart")
            # Render visually as an image/diagram via HTML component!
            st_mermaid(mermaid_code)
            
            st.markdown("### 📝 Detailed Revision Notes")
            st.markdown(remaining_text)
        else:
            st.markdown(output_text)
            
        st.download_button(
            "Download mind map & notes",
            output_text,
            file_name="lecturesathi_mindmap.txt",
            mime="text/plain",
        )
    else:
        st.info("Click the button above to generate a graphical node-based mind map from your transcript.")

with tab4:
    st.subheader("Listen to the transcript")
    st.caption("Timbre converts transcript text into spoken audio.")
    default_text = transcript[:1200]
    speech_text = st.text_area("Text to speak", value=default_text, height=160, max_chars=1800)
    
    language_options = {
        "English (India)": "en-IN",
        "Hindi": "hi-IN",
        "Punjabi": "pa-IN",
        "Bengali": "bn-IN",
        "Marathi": "mr-IN",
        "Gujarati": "gu-IN",
        "Kannada": "kn-IN",
        "Telugu": "te-IN",
        "Tamil": "ta-IN",
        "Malayalam": "ml-IN",
    }
    selected_label = st.selectbox(
        "Speech language",
        list(language_options.keys()),
    )
    voice = st.selectbox("Timbre voice", ["Nalini", "Pranav", "Yashvi", "Shlok"])
    if st.button("Generate spoken audio", type="primary"):
        if not speech_text.strip():
            st.warning("Enter some text to speak.")
        else:
            try:
                with st.spinner("Generating speech with Gnani Timbre..."):
                    audio_bytes = synthesize_speech(
                        speech_text, api_key, language_options[selected_label], voice
                    )
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    "Download spoken audio",
                    audio_bytes,
                    file_name="lecturesathi_audio.wav",
                    mime="audio/wav",
                )
            except Exception as exc:
                st.error(str(exc))

st.divider()
st.caption("LectureSathi AI · Powered by Gnani Prisma v3.0, Timbre v2.5, Groq Qwen 3.8 & Google Gemini.")