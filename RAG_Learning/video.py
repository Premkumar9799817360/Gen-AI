import os
import shutil
import subprocess
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import av
import wave
from faster_whisper import WhisperModel
from llm_model import LLM_MODEL


# -----------------------------
# CONFIG
# -----------------------------
VIDEO_URL = "https://www.youtube.com/watch?v=3kT2wAGT18A"
WORKDIR = "video_rag_data"
COLLECTION_NAME = "video_rag"

os.makedirs(WORKDIR, exist_ok=True)


# -----------------------------
# 1. DOWNLOAD VIDEO
# -----------------------------
def download_video(url):
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp not found. Install using: pip install yt-dlp")

    output_path = os.path.join(WORKDIR, "video.mp4")

    subprocess.run(
        ["yt-dlp", "-f", "mp4", "-o", output_path, url],
        check=True
    )

    return output_path


# -----------------------------
# 2. EXTRACT AUDIO (MoviePy)
# -----------------------------
def extract_audio(video_path):
    audio_path = os.path.join(WORKDIR, "audio.wav")

    container = av.open(video_path)

    audio_stream = next(
        (s for s in container.streams if s.type == "audio"), None
    )

    if audio_stream is None:
        raise RuntimeError("No audio stream found in video.")

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=16000
    )

    pcm_data = bytearray()

    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            frame = resampler.resample(frame)
            for f in frame:
                pcm_data.extend(f.to_ndarray().tobytes())

    container.close()

    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        wf.writeframes(pcm_data)

    return audio_path
# -----------------------------
# 3. TRANSCRIBE (WHISPER)
# -----------------------------
def transcribe(audio_path):
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    segments, _ = model.transcribe(audio_path)
    return " ".join(segment.text for segment in segments)


# -----------------------------
# 4. CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------
# 5. CHUNKING
# -----------------------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# -----------------------------
# 6. VECTOR DB + EMBEDDINGS
# -----------------------------
def store_chunks(chunks):
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, convert_to_numpy=True).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return collection, embedder


# -----------------------------
# 7. RETRIEVAL
# -----------------------------
def retrieve(query, collection, embedder, k=3):
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]


# -----------------------------
# 8. GENERATE ANSWER (LLM)
# -----------------------------
def generate_answer(query, context):
    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""
    return LLM_MODEL(prompt)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("⬇️ Downloading video...")
    video = download_video(VIDEO_URL)

    print("🎧 Extracting audio (MoviePy)...")
    audio = extract_audio(video)
    print("✅ Audio extracted:", audio)

    print("📝 Transcribing...")
    raw_text = transcribe(audio)

    print("🧹 Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("✂️ Chunking...")
    chunks = chunk_text(cleaned_text)

    print("📦 Storing in ChromaDB...")
    collection, embedder = store_chunks(chunks)

    query = input("\nAsk a question from the video: ")
    retrieved_chunks = retrieve(query, collection, embedder)

    print("\n🤖 Generating answer...")
    answer = generate_answer(query, "\n".join(retrieved_chunks))

    print("\n✅ ANSWER:\n", answer)


if __name__ == "__main__":
    main()