# 🗣️ Vaani — Real-Time Audio Intelligence (Luddy Hackathon 2025)

**Vaani** is an advanced real-time audio processing tool built during the **Luddy Hackathon 2025**. It transforms both live and recorded audio into structured, searchable, and intelligent transcripts. From transcription and speaker diarization to summarization and sentiment analysis — Vaani makes audio data instantly useful.

---

## 🚀 Key Features

- 🎙️ **Live & File-based Transcription** using **Fast-Whisper**
- 🧑‍🤝‍🧑 **Speaker Diarization** using **Distilled-Whisper** and **PyAnnote-Audio**
- 💬 **Summarization & Sentiment Analysis** using **Hugging Face Transformers**
- ❓ **Question Answering (QnA)** from transcript content
- 🧠 Smart chunking for low-latency real-time feedback
- 🌐 **Gradio Interface** for seamless interaction

---

## 📁 Project Structure

```
Vaani_LuddyHack/
│
├── app.py                 # Main Gradio interface and orchestrator
├── config.py              # Configuration handling, API key loading, and utility constants
├── live_transcription.py  # Handles real-time audio input and Fast-Whisper transcription
├── summarization.py       # Summarization, sentiment analysis, and QnA using Hugging Face models
├── requirements.txt       # All required packages
└── .env                   # Stores your secret API keys (ignored in Git)
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Kratik-Rathi/Vaani_LuddyHack.git
cd Vaani_LuddyHack
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the root directory with the following content:

```
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

> 🔐 You may also need other authentication (e.g., for PyAnnote or WhisperX if using hosted models). Include those as needed.

⚠️ **Do not commit your `.env` file**. It contains sensitive keys.

### 5. Launch the App

```bash
python app.py
```

> Gradio will open a local web app in your browser. Start using Vaani immediately!

---

## 🧪 Example Use Cases

- 🎓 Transcribing and summarizing lectures or webinars
- 🧑‍💼 Real-time meeting notes with speaker-wise segmentation
- 🎙️ Podcast processing and topic extraction
- 🕵️‍♂️ Investigative conversation analysis with sentiment breakdown
- 📊 Generating QnA insights from long audio interviews

---

## 🌍 Tech Stack

- 🧠 [Fast-Whisper](https://github.com/guillaumekln/faster-whisper) — Lightweight Whisper model for fast transcription
- 🧑‍🤝‍🧑 [WhisperX](https://github.com/m-bain/whisperx) — Alignment & Diarization for Whisper
- 🔊 [PyAnnote-Audio](https://github.com/pyannote/pyannote-audio) — Speaker diarization from pretrained models
- 💬 [Hugging Face Transformers](https://huggingface.co/) — For summarization, sentiment, and Q&A
- 🌐 [Gradio](https://www.gradio.app/) — Fast and interactive frontend
- 🧰 PyDub, FFmpeg, Torch — Audio preprocessing and model support

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Fast-Whisper](https://github.com/guillaumekln/faster-whisper) for efficient transcription
- [WhisperX](https://github.com/m-bain/whisperx) for alignment and diarization
- [PyAnnote](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [Hugging Face](https://huggingface.co/) for state-of-the-art NLP models
- [Gradio](https://www.gradio.app/) for rapid UI development

