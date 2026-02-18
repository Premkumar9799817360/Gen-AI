"""
============================================================
FAISS VECTOR DATABASE – COMPLETE WORKING EXAMPLE
============================================================

This file shows:
1. How to create embeddings
2. How to create FAISS index
3. How to add vectors
4. How to store text mapping
5. How to retrieve relevant text

Used for:
- RAG
- Semantic Search
- Similarity Search

============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict


# ============================================================
# LOAD EMBEDDING MODEL (OPEN SOURCE)
# ============================================================

def load_embedding_model():
    """
    Loads sentence transformer embedding model

    Model:
        all-MiniLM-L6-v2
    Dimension:
        384
    """
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    model: SentenceTransformer,
    texts: List[str]
) -> np.ndarray:
    """
    Converts text into embeddings

    Output:
        NumPy array of shape (n_texts, dim)
    """
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embedding_dim: int):
    """
    Creates FAISS index

    Index type:
        IndexFlatIP (Inner Product)
        Works well with normalized embeddings
    """
    index = faiss.IndexFlatIP(embedding_dim)
    return index


# ============================================================
# ADD DOCUMENTS TO FAISS
# ============================================================

def add_documents(
    index,
    embeddings: np.ndarray
):
    """
    Adds vectors to FAISS index
    """
    index.add(embeddings)


# ============================================================
# STORE TEXT METADATA
# ============================================================

def create_text_store(texts: List[str]) -> Dict[int, str]:
    """
    Stores text with index positions
    """
    return {i: text for i, text in enumerate(texts)}


# ============================================================
# SEARCH FAISS INDEX
# ============================================================

def search_faiss(
    model: SentenceTransformer,
    index,
    text_store: Dict[int, str],
    query: str,
    top_k: int = 5
):
    """
    Performs similarity search

    Steps:
    1. Embed query
    2. Search FAISS index
    3. Map indices to original text
    """
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "text": text_store[idx],
            "score": float(score)
        })

    return results


def retrieve_top_k(
    query_embedding: np.ndarray,
    index,
    documents: list[str],
    top_k: int = 5
):
    """
    Retrieve top-k documents from FAISS
    """
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "text": documents[idx],
            "score": float(score)
        })

    return results

# ranker.py

from sentence_transformers import CrossEncoder

def load_reranker():
    """
    Cross encoder for reranking
    """
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return model


def rerank(query: str, retrieved_docs: list[dict], top_k: int = 3):
    """
    Rerank retrieved docs using cross encoder
    """
    pairs = [(query, doc["text"]) for doc in retrieved_docs]

    scores = load_reranker().predict(pairs)

    for i, score in enumerate(scores):
        retrieved_docs[i]["rerank_score"] = float(score)

    reranked = sorted(
        retrieved_docs,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


def generate_answer(query: str, context_docs: list[str]):
    tokenizer, model = load_llm()

    context = "\n\n".join(context_docs)

    prompt = f"""
You are an expert assistant.
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============================================================
# FULL PIPELINE EXAMPLE
# ============================================================

if __name__ == "__main__":

    # -----------------------------
    # Sample documents
    # -----------------------------
    documents = [
        "FAISS is a library for efficient similarity search",
        "RAG combines retrieval and generation",
        "Vector databases store embeddings",
        "Embeddings represent text as numbers",
        "Large language models use context"
    ]

    # -----------------------------
    # Load embedding model
    # -----------------------------
    model = load_embedding_model()

    # -----------------------------
    # Create embeddings
    # -----------------------------
    embeddings = create_embeddings(model, documents)

    # -----------------------------
    # Create FAISS index
    # -----------------------------
    dim = embeddings.shape[1]
    index = create_faiss_index(dim)

    # -----------------------------
    # Add embeddings to FAISS
    # -----------------------------
    add_documents(index, embeddings)

    # -----------------------------
    # Store original texts
    # -----------------------------
    text_store = create_text_store(documents)

    # -----------------------------
    # Query
    # -----------------------------
    query = "How does similarity search work?"

    results = search_faiss(
        model=model,
        index=index,
        text_store=text_store,
        query=query,
        top_k=3
    )
    retrieved = retrieve_top_k(query_embedding, index, documents, top_k=8)

    reranked = rerank(query, retrieved, top_k=3)

    context_docs = [doc["text"] for doc in reranked]

    answer = generate_answer(query, context_docs)

    # -----------------------------
    # Print results
    # -----------------------------
    for r in results:
        print(f"Score: {r['score']:.4f} | Text: {r['text']}")