# 🎓 LectureSathi AI

> Transform lectures into transcripts, AI-powered explanations, visual mind maps, revision notes, and spoken summaries.

LectureSathi AI is a multilingual learning assistant that enables students understand lectures more effectively. Upload an audio or video lecture, generate transcripts using speech recognition, interact with an AI tutor, create graphical mind maps, and convert notes back into speech.

---

## ✨ Features

### 🎙️ Lecture Transcription
- Upload lecture recordings in audio or video format.
- Automatic audio extraction and preprocessing.
- Supports:
  - MP3
  - WAV
  - M4A
  - AAC
  - OGG
  - FLAC
  - MP4

### 📝 Transcript Generation
- High-quality speech-to-text conversion using **Gnani Prisma v3.0**.
- Multiple transcription modes:
  - Transcribe
  - Codemix
  - Verbatim
  - Transliterate

### 🤖 AI Tutor
Powered by **Qwen 3.8 27B via Groq**.

Students can:
- Ask questions from the uploaded lecture.
- Get concept explanations.
- Receive simple analogies and examples.
- Generate follow-up learning questions.
- Understand difficult topics quickly.

### 🗺️ Visual Mind Maps
Powered by **Google Gemini 2.5 Flash**.

Generates:
- Mermaid-based graphical flowcharts.
- Concept relationships.
- Revision-friendly notes.
- Structured topic breakdowns.

### 🔊 Text-to-Speech
Powered by **Gnani Timbre v2.5**.

Convert transcripts into natural speech and download them as audio files.

Supported Languages:
- English
- Hindi
- Punjabi
- Bengali
- Marathi
- Gujarati
- Kannada
- Telugu
- Tamil
- Malayalam

---

## 🏗️ System Architecture

```text
Audio/Video Upload
         │
         ▼
Gnani Prisma STT
         │
         ▼
Lecture Transcript
         │
 ┌───────┼────────┐
 ▼       ▼        ▼
AI Tutor Mind Map TTS
(Groq)  (Gemini) (Timbre)
```

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### AI Models & APIs

| Component | Technology |
|------------|------------|
| Speech-to-Text | Gnani Prisma v3.0 |
| AI Tutor | Qwen 3.8 27B (Groq) |
| Mind Map Generation | Gemini 2.5 Flash |
| Text-to-Speech | Gnani Timbre v2.5 |

### Python Libraries

- streamlit
- requests
- openai
- google-genai
- numpy
- pydub
- imageio-ffmpeg
- scikit-learn
- python-dotenv

---

## 📂 Repository Structure

```text
LectureSaathi-for-Gnani_AI-/
│
├── README.md
├── .gitignore
│
└── lecturesathi_ai/
    ├── app.py
    ├── requirements.txt
    ├── runtime.txt
    ├── Procfile
    └── .streamlit/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/lecturesathi-ai.git
cd lecturesathi-ai
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GNANI_API_KEY=your_gnani_api_key

GROQ_API_KEY=your_groq_api_key

GEMINI_API_KEY=your_gemini_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Application will start at:

```text
http://localhost:8501
```

---

## 📖 How It Works

### Step 1
Upload an audio or video lecture.

### Step 2
LectureSathi extracts audio and sends it to Gnani Prisma for transcription.

### Step 3
A complete lecture transcript is generated.

### Step 4
Ask personalized questions using the AI Tutor.

### Step 5
Generate a visual mind map and revision notes.

### Step 6
Convert important notes into speech and download them.

---

## 🔒 Security

The following files contain API keys and should never be pushed to GitHub:

```text
.env
.streamlit/secrets.toml
```

These files are excluded using `.gitignore`.

---

## 🚀 Future Improvements

- PDF lecture notes export
- Quiz generation
- Flashcard generation
- Multi-document learning support
- Lecture summarization
- RAG-based knowledge retrieval
- Chat history persistence

---

## 👩‍💻 Author

```
Bhumi Bhardwaj & Rudra Pratap Singh
```
---

## ⭐ Support

If you found this project useful, consider starring the repository.

```bash
⭐ Star this repo
🍴 Fork it
🚀 Build something awesome
```
