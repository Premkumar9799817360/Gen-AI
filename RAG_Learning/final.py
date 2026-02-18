import os
import shutil
import subprocess
import re
import wave
import uuid
import av
import cv2
import pytesseract
import gradio as gr
import chromadb

from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
from llm_model import LLM_MODEL

# -----------------------------
# CONFIG
# -----------------------------
WORKDIR = "video_rag_data"
COLLECTION_NAME = "video_rag"

os.makedirs(WORKDIR, exist_ok=True)

GLOBAL_COLLECTION = None
GLOBAL_EMBEDDER = None
GLOBAL_LLM = None


# -----------------------------
# DOWNLOAD VIDEO
# -----------------------------
def download_video(url):
    output_path = os.path.join(WORKDIR, "video.mp4")
    subprocess.run(["yt-dlp", "-f", "mp4", "-o", output_path, url], check=True)
    return output_path


# -----------------------------
# EXTRACT AUDIO
# -----------------------------
def extract_audio(video_path):
    audio_path = os.path.join(WORKDIR, "audio.wav")

    container = av.open(video_path)
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    
    if audio_stream is None:
        container.close()
        return None

    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=16000
    )

    pcm = bytearray()
    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            frame = resampler.resample(frame)
            for f in frame:
                pcm.extend(f.to_ndarray().tobytes())

    container.close()

    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm)

    return audio_path


# -----------------------------
# VIDEO FRAME TEXT (OCR)
# -----------------------------
def extract_frame_text(video_path, interval=5):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # default fallback
    
    frame_text = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % (fps * interval) == 0:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray)
                if text.strip():
                    frame_text.append(text)
            except Exception as e:
                print(f"OCR error: {e}")

        count += 1

    cap.release()
    return " ".join(frame_text)


# -----------------------------
# TRANSCRIBE AUDIO
# -----------------------------
def transcribe(audio_path):
    if audio_path is None:
        return ""
    
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)

    return " ".join(seg.text for seg in segments)


# -----------------------------
# CLEAN + CHUNK
# -----------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )
    return splitter.split_text(text)


# -----------------------------
# STORE IN CHROMA
# -----------------------------
def store_chunks(chunks):
    client = chromadb.Client(Settings(persist_directory=WORKDIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(uuid.uuid4()) for _ in chunks]
    )

    return collection, embedder


# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve(query, collection, embedder, k=4):
    q_emb = embedder.encode(query).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    return res["documents"][0]


# -----------------------------
# ANSWER
# -----------------------------
def generate_answer(query, context):
    prompt = f"""
    you are a helpful assistant that answers questions based on the provided video context. 
    Use only the information in the context to answer the question. 
    If the answer is not in the context, say you don't know.
   Answer the question using the context below.

Context:
{context}

Question:
{query}
"""
    return LLM_MODEL(prompt)

# -----------------------------
# UI FUNCTIONS
# -----------------------------
def process_video(youtube_url, uploaded_video):
    global GLOBAL_COLLECTION, GLOBAL_EMBEDDER

    try:
        if not youtube_url and not uploaded_video:
            return "❌ Please provide a YouTube URL or upload a video"

        if youtube_url:
            video_path = download_video(youtube_url)
        else:
            video_path = os.path.join(WORKDIR, "uploaded.mp4")
            shutil.copy(uploaded_video.name, video_path)

        audio_path = extract_audio(video_path)
        audio_text = transcribe(audio_path)
        frame_text = extract_frame_text(video_path)

        full_text = clean_text(audio_text + " " + frame_text)
        
        if not full_text:
            return "❌ No text extracted from video"
        
        chunks = chunk_text(full_text)
        GLOBAL_COLLECTION, GLOBAL_EMBEDDER = store_chunks(chunks)

        return "✅ Video processed successfully. You can now ask questions."
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


def ask_question(question):
    try:
        if GLOBAL_COLLECTION is None:
            return "❌ Please process a video first"

        docs = retrieve(question, GLOBAL_COLLECTION, GLOBAL_EMBEDDER)
        return generate_answer(question, "\n".join(docs))
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Video RAG (Audio + Frame Text)")

    yt = gr.Textbox(label="YouTube URL (optional)")
    file = gr.File(label="Upload Video (optional)")

    status = gr.Textbox(label="Status", interactive=False)
    gr.Button("Process Video").click(process_video, [yt, file], status)

    q = gr.Textbox(label="Ask Question")
    a = gr.Textbox(label="Answer", lines=6, interactive=False)
    q.submit(ask_question, q, a)

if __name__ == "__main__":
    demo.launch()