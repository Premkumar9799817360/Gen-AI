import os
import shutil
import subprocess
import re
import wave
import uuid
import av
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


# -----------------------------
# DOWNLOAD VIDEO
# -----------------------------
def download_video(url):
    output_path = os.path.join(WORKDIR, "video.mp4")
    subprocess.run(
        ["yt-dlp", "-f", "mp4", "-o", output_path, url],
        check=True
    )
    return output_path


# -----------------------------
# EXTRACT AUDIO
# -----------------------------
def extract_audio(video_path):
    audio_path = os.path.join(WORKDIR, "audio.wav")

    container = av.open(video_path)
    audio_stream = next(s for s in container.streams if s.type == "audio")

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
# TRANSCRIBE
# -----------------------------
def transcribe(audio_path):
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)

    text = ""
    for seg in segments:
        text += seg.text + " "

    return text.strip()


# -----------------------------
# CLEAN + CHUNK
# -----------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    return splitter.split_text(text)


# -----------------------------
# STORE IN CHROMA
# -----------------------------
def store_chunks(chunks):
    client = chromadb.Client(
        Settings(persist_directory=WORKDIR)
    )

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
def retrieve(query, collection, embedder, k=3):
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

    if not youtube_url and not uploaded_video:
        return "❌ Provide YouTube URL or upload video"

    if youtube_url:
        video_path = download_video(youtube_url)
    else:
        video_path = os.path.join(WORKDIR, "uploaded.mp4")
        shutil.copy(uploaded_video.name, video_path)

    audio = extract_audio(video_path)
    text = clean_text(transcribe(audio))
    chunks = chunk_text(text)

    GLOBAL_COLLECTION, GLOBAL_EMBEDDER = store_chunks(chunks)

    return "✅ Video processed. Ask questions now."


def ask_question(question):
    if GLOBAL_COLLECTION is None:
        return "❌ Process video first"

    docs = retrieve(question, GLOBAL_COLLECTION, GLOBAL_EMBEDDER)
    return generate_answer(question, "\n".join(docs))


# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Video RAG System")

    yt = gr.Textbox(label="YouTube URL")
    file = gr.File(label="Upload Video")

    status = gr.Textbox(label="Status")
    gr.Button("Process").click(process_video, [yt, file], status)

    q = gr.Textbox(label="Ask Question")
    a = gr.Textbox(label="Answer", lines=8)
    q.submit(ask_question, q, a)

demo.launch() 