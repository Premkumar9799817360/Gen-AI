import os
import shutil
import subprocess
import re
import streamlit as st
import tempfile

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
WORKDIR = "video_rag_data"
COLLECTION_NAME = "video_rag"

os.makedirs(WORKDIR, exist_ok=True)


# -----------------------------
# PROCESSING FUNCTIONS
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
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_data)
    
    return audio_path


def transcribe(audio_path):
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )
    
    segments, _ = model.transcribe(audio_path)
    return " ".join(segment.text for segment in segments)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


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


def retrieve(query, collection, embedder, k=3):
    query_embedding = embedder.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    return results["documents"][0]


def generate_answer(query, context):
    prompt = f"""
You are a strict question-answering assistant.

RULES (VERY IMPORTANT):
1. Answer ONLY using the provided Context.
2. Do NOT use prior knowledge or make assumptions.
3. If the answer is not explicitly stated in the Context, reply exactly:
   "I don't know based on the provided context. Please rephrase your question."
4. Do NOT add explanations, examples, or extra information.
5. Keep the answer concise and factual.
6. If the question is unrelated to the Context, follow rule #3.

Context:
{context}

Question:
{query}

Answer:
"""
    return LLM_MODEL(prompt)


# -----------------------------
# STREAMLIT APP
# -----------------------------
def main():
    st.title("🎥 Video RAG - Ask Questions from Videos")
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'embedder' not in st.session_state:
        st.session_state.embedder = None
    
    # Input method selection
    input_method = st.radio("Select Input Method:", ["YouTube Link", "Upload Video"])
    
    video_path = None
    
    if input_method == "YouTube Link":
        youtube_url = st.text_input("Enter YouTube URL:")
        
        if st.button("Process Video") and youtube_url:
            try:
                with st.spinner("Downloading video..."):
                    video_path = download_video(youtube_url)
                
                with st.spinner("Extracting audio..."):
                    audio_path = extract_audio(video_path)
                
                with st.spinner("Transcribing audio..."):
                    raw_text = transcribe(audio_path)
                
                with st.spinner("Processing text..."):
                    cleaned_text = clean_text(raw_text)
                    chunks = chunk_text(cleaned_text)
                
                with st.spinner("Storing embeddings..."):
                    st.session_state.collection, st.session_state.embedder = store_chunks(chunks)
                
                st.session_state.processed = True
                st.success("✅ Video processed successfully!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:  # Upload Video
        uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if st.button("Process Video") and uploaded_file:
            try:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
                
                with st.spinner("Extracting audio..."):
                    audio_path = extract_audio(video_path)
                
                with st.spinner("Transcribing audio..."):
                    raw_text = transcribe(audio_path)
                
                with st.spinner("Processing text..."):
                    cleaned_text = clean_text(raw_text)
                    chunks = chunk_text(cleaned_text)
                
                with st.spinner("Storing embeddings..."):
                    st.session_state.collection, st.session_state.embedder = store_chunks(chunks)
                
                st.session_state.processed = True
                st.success("✅ Video processed successfully!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Q&A Section (only show after processing)
    if st.session_state.processed:
        st.markdown("---")
        st.subheader("💬 Ask Questions")
        
        question = st.text_input("Enter your question:")
        
        if st.button("Get Answer") and question:
            try:
                with st.spinner("Searching for answer..."):
                    retrieved_chunks = retrieve(
                        question, 
                        st.session_state.collection, 
                        st.session_state.embedder
                    )
                    
                    answer = generate_answer(question, "\n".join(retrieved_chunks))
                
                st.markdown("### Answer:")
                st.write(answer)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()