import os
import queue
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import whisper
import librosa
from scipy.spatial.distance import cosine
import tempfile
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Config
sample_rate = 16000
chunk_duration = 3  # seconds
chunk_samples = sample_rate * chunk_duration
max_buffer_chunks = 5
speaker_threshold = 0.25

# Globals
q = queue.Queue()
model = whisper.load_model("medium")
sentiment_analyzer = SentimentIntensityAnalyzer()

# Shared state
buffer = AudioSegment.silent(duration=0)
temp_chunks = []
speaker_profiles = {}
next_speaker_id = 0
last_buffer_speaker = None
document_text = ""

# Control flag for stopping stream
stop_thread = False

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

def get_mfcc(samples):
    if len(samples) < chunk_samples:
        return None
    mfcc = librosa.feature.mfcc(y=samples.astype(np.float32), sr=sample_rate, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def match_speaker(current_mfcc, speaker_profiles, threshold=0.3):
    min_dist = float('inf')
    matched_speaker = None
    for speaker_id, mfcc in speaker_profiles.items():
        dist = cosine(current_mfcc, mfcc)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            matched_speaker = speaker_id
    return matched_speaker

def analyze_audio_and_get_text(audio_seg, speaker_label=None):
    global document_text
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_seg.export(temp_audio.name, format="wav")
        try:
            result = model.transcribe(temp_audio.name, fp16=False, language="en", task="transcribe")
            text = result["text"].strip()
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            os.remove(temp_audio.name)
            return ""  # Skip this segment

        os.remove(temp_audio.name)

    sentiment_label = "Neutral"
    if text:
        sentiment_scores = sentiment_analyzer.polarity_scores(text)
        compound = sentiment_scores['compound']
        if compound >= 0.05:
            sentiment_label = "Positive"
        elif compound <= -0.03:
            sentiment_label = "Negative"

        document_text += f"Speaker: {text}\nSentiment: {sentiment_label}\n\n"
        return f"{text}  ({sentiment_label})"
    return ""


def start_transcription_stream(callback):
    global buffer, temp_chunks, last_buffer_speaker, speaker_profiles, next_speaker_id, document_text, stop_thread

    # Reset all state variables
    buffer = AudioSegment.silent(duration=0)
    temp_chunks = []
    speaker_profiles = {}
    next_speaker_id = 0
    last_buffer_speaker = None
    document_text = ""
    stop_thread = False

    print("🎙️ Live recording started...")

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16',
                            callback=audio_callback, blocksize=chunk_samples):
            while not stop_thread:
                audio_data = q.get()
                samples = audio_data.flatten()
                segment = AudioSegment(
                    samples.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1
                )

                mfcc = get_mfcc(samples)
                if mfcc is None:
                    continue

                current_speaker = match_speaker(mfcc, speaker_profiles, speaker_threshold)
                if current_speaker is None:
                    current_speaker = next_speaker_id
                    speaker_profiles[current_speaker] = mfcc
                    next_speaker_id += 1
                else:
                    speaker_profiles[current_speaker] = (speaker_profiles[current_speaker] + mfcc) / 2

                if last_buffer_speaker is None:
                    last_buffer_speaker = current_speaker

                if current_speaker != last_buffer_speaker or len(temp_chunks) >= max_buffer_chunks:
                    transcript = analyze_audio_and_get_text(buffer, last_buffer_speaker)
                    if transcript:
                        callback(transcript)

                    buffer = segment
                    temp_chunks = [samples]
                    last_buffer_speaker = current_speaker
                else:
                    buffer += segment
                    temp_chunks.append(samples)

    except Exception as e:
        print(f"❌ Error in live transcription: {e}")

    # Final dump
    if len(buffer) > 0:
        transcript = analyze_audio_and_get_text(buffer, last_buffer_speaker)
        if transcript:
            callback(transcript)

    print("🛑 Live transcription stopped.")

def stop_transcription():
    global stop_thread
    stop_thread = True


def print_callback(text):
    print(f"\n📢 Transcribed: {text}")
    print(f"[Live Update] {text}")


def main():
    global stop_thread
    try:
        start_transcription_stream(print_callback)
    except KeyboardInterrupt:
        stop_thread = True
        print("🛑 Keyboard interrupt received. Stopping...")


if __name__ == "__main__":
    main()
