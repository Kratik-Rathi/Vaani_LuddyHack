import gradio as gr
import threading
import os
from faster_whisper import WhisperModel
from live_transcription import start_transcription_stream, stop_transcription
from summarization import summarize_transcript_with_groq, ask_groq_question, transcribe_with_groq_whisper_and_diarization

# === Shared Variables ===
live_output = ""
live_recording_thread = None
is_live_on = False
model = WhisperModel("medium")

# Persistent state
file_transcript_text = ""
live_transcript_text = ""


def transcribe_with_faster_whisper_local(audio_file_path):
    segments, _ = model.transcribe(audio_file_path)
    return " ".join([seg.text.strip() for seg in segments])


def callback_update(text):
    global live_output, live_transcript_text
    live_output += f"{text}\n\n"
    live_transcript_text = live_output.strip()


def run_live_transcription():
    start_transcription_stream(callback_update)


def run_file_transcription(file, fast_mode):
    global file_transcript_text
    file_path = file.name
    if os.path.exists("file_transcript.txt"):
        os.remove("file_transcript.txt")

    file_transcript_text = ""  # Reset state

    if fast_mode:
        transcript = transcribe_with_faster_whisper_local(file_path)
    else:
        transcript = transcribe_with_groq_whisper_and_diarization(file_path)

    with open("file_transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    file_transcript_text = transcript.strip()
    return gr.update(value="✅ File processed", visible=True), transcript


def reset_file_inputs():
    global file_transcript_text
    file_transcript_text = ""
    if os.path.exists("file_transcript.txt"):
        os.remove("file_transcript.txt")
    return (
        gr.update(value=None),       # Reset file input
        gr.update(value="", visible=False),  # Reset status
        gr.update(value=""),         # Reset transcribed text
        gr.update(value="", visible=False),  # Reset summary output
        gr.update(value="")          # Reset QA output
    )


def start_or_stop_live():
    global live_output, live_recording_thread, is_live_on, live_transcript_text
    if not is_live_on:
        live_output = ""
        live_transcript_text = ""
        if os.path.exists("live_transcript.txt"):
            os.remove("live_transcript.txt")

        live_recording_thread = threading.Thread(target=run_live_transcription)
        live_recording_thread.start()
        is_live_on = True
        return "⏹️ Stop Live Transcription", "🎙️ Live transcription started..."
    else:
        stop_transcription()
        with open("live_transcript.txt", "w", encoding="utf-8") as f:
            f.write(live_transcript_text)
        is_live_on = False
        return "▶️ Start Live Transcription", "🛑 Transcription stopped"


def refresh_captions():
    return live_output


def file_summary():
    if not file_transcript_text:
        return gr.update(value="⚠️ No transcript file found.", visible=True)
    with open("file_transcript.txt", "w", encoding="utf-8") as f:
        f.write(file_transcript_text)
    summary = summarize_transcript_with_groq("file_transcript.txt")
    return gr.update(value=summary, visible=True)


def live_summary():
    if not live_transcript_text:
        return gr.update(value="⚠️ No live transcript yet.", visible=True)
    return gr.update(value=live_transcript_text.split("\n📄 Final Transcript:")[0], visible=True)


def handle_qa_file(question):
    return ask_groq_question(question, "file_transcript.txt")


def handle_qa_live(question):
    return ask_groq_question(question, "live_transcript.txt")


# === UI ===
with gr.Blocks() as demo:
    gr.Markdown("# 🎤 Vaani AI")

    with gr.Tabs() as tabs:
        with gr.Tab("Upload Audio File") as tab_file:
            file_input = gr.File(label="Upload Audio File", file_types=[".mp3", ".wav"])
            fast_mode = gr.Checkbox(label="⚡ Use fast transcription (no speaker diarization)", visible=False, value=True)

            transcribe_button = gr.Button("Transcribe Audio")
            reset_file_btn = gr.Button("🔁 Upload New File")

            transcribe_status = gr.Textbox(label="Status", visible=False)
            transcribed_text = gr.Textbox(label="📝 Transcribed Text", lines=10, interactive=False)

            generate_summary_button = gr.Button("📄 Generate Summary")
            summary_output = gr.Markdown(visible=False)

            qa_input_file = gr.Textbox(label="❓ Ask a question about the file transcript")
            qa_button_file = gr.Button("💬 Ask")
            qa_output_file = gr.Markdown()

            transcribe_button.click(
                run_file_transcription,
                inputs=[file_input, fast_mode],
                outputs=[transcribe_status, transcribed_text]
            )
            reset_file_btn.click(
                reset_file_inputs,
                outputs=[file_input, transcribe_status, transcribed_text, summary_output, qa_output_file]
            )
            generate_summary_button.click(file_summary, outputs=summary_output)
            qa_button_file.click(handle_qa_file, inputs=qa_input_file, outputs=qa_output_file)

        with gr.Tab("Live Transcription") as tab_live:
            gr.Markdown("## 🎙️ Live Transcription")

            live_toggle = gr.Button("▶️ Start Live Transcription")
            live_status = gr.Textbox(visible=True, label="Status")
            live_text = gr.Textbox(label="📝 Live Captions", lines=10, interactive=False)

            generate_summary_button_live = gr.Button("📄 Generate Summary")
            summary_output_local = gr.Markdown()

            qa_input_live = gr.Textbox(label="❓ Ask a question about the live transcript")
            qa_button_live = gr.Button("💬 Ask")
            qa_output_live = gr.Markdown()

            live_toggle.click(start_or_stop_live, outputs=[live_toggle, live_status])
            generate_summary_button_live.click(
                fn=lambda: summarize_transcript_with_groq("live_transcript.txt"),
                outputs=summary_output_local
            )
            qa_button_live.click(handle_qa_live, inputs=qa_input_live, outputs=qa_output_live)
            demo.load(fn=refresh_captions, every=2, outputs=[live_text])

if __name__ == "__main__":
    demo.queue().launch()
