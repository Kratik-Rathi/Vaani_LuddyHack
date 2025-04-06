import requests
import logging
from groq import Groq
from pyannote.audio import Pipeline
from pydub import AudioSegment
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
from config import GROQ_API_KEY
from config import Huggingface_token

nltk.download("vader_lexicon")
logging.getLogger("pyannote").setLevel(logging.ERROR)

client = Groq(api_key=GROQ_API_KEY)
HF_TOKEN = Huggingface_token


def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    compound = score["compound"]
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 100)
    return f"{mins:02d}:{secs:02d}.{millis:02d}"


def transcribe_with_groq_whisper_and_diarization(audio_file_path, chunk_duration=10):
    print("\nLoading diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

    audio = AudioSegment.from_file(audio_file_path)
    duration_sec = len(audio) / 1000
    temp_chunk_path = "temp_chunk.wav"
    full_transcript = ""
    final_transcript = "\n📄 Final Transcript:\n" + "─" * 50 + "\n"

    for start in range(0, int(duration_sec), chunk_duration):
        end = min(start + chunk_duration, duration_sec)
        chunk_audio = audio[start * 1000:end * 1000]
        chunk_audio.export(temp_chunk_path, format="wav")

        try:
            diarization = pipeline(temp_chunk_path)
        except Exception as e:
            print(f"Diarization error: {e}")
            continue

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            global_start = start + turn.start
            global_end = start + turn.end
            segment = audio[int(global_start * 1000):int(global_end * 1000)]
            segment.export(temp_chunk_path, format="wav")

            with open(temp_chunk_path, "rb") as f:
                files = {"file": f}
                response = requests.post(
                    "https://api.groq.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {client.api_key}"},
                    files=files,
                    data={"model": "distilled-whisper", "language": "en"}
                )

            if response.status_code == 200:
                text = response.json()["text"].strip()
            else:
                text = ""

            if not text:
                continue

            sentiment = analyze_sentiment(text)
            speaker_label = f"Speaker {speaker}"
            start_time = format_timestamp(global_start)
            end_time = format_timestamp(global_end)

            print(f"\n🎤 {speaker_label}: {text}")
            print(f"💬 Sentiment: {sentiment}")

            final_transcript += f"{speaker_label}\nText: {text}\nSentiment: {sentiment}\n\n"
            full_transcript += f"\n🎤 {speaker_label}: {text}\n💬 Sentiment: {sentiment}\n"

    os.remove(temp_chunk_path)

    final_transcript += "─" * 50 + "\n✅ Done.\n"
    output_path = os.path.splitext(os.path.basename(audio_file_path))[0] + "_groq_diarized.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_transcript + "\n" + final_transcript)

    return full_transcript + "\n" + final_transcript


def summarize_transcript_with_groq(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    prompt = f"""
You are an AI assistant. Here's a meeting transcript:

{transcript}

Instructions:
1. Provide a structured summary.
2. Extract any decisions made or conclusions drawn.
3. List actionable items with who said what.
4. Rate the overall sentiment of the discussion and flag emotional moments.

Respond in clear bullet points.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",  # ✅ Using supported model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024
    )

    return response.choices[0].message.content


def ask_groq_question(question, file_path="live_transcript.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            context = f.read()

        if not context.strip():
            return "❗ No transcript found. Please transcribe audio first."

        messages = [
            {"role": "system", "content": f"You are a helpful assistant. Answer questions based only on this transcript:\n\n{context}"},
            {"role": "user", "content": question}
        ]

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.5,
            max_tokens=512
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error in Q&A: {str(e)}"