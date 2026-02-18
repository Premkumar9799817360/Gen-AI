"""
============================================================
  VECTOR DB #3: CHROMADB (Open Source, Free)
  - Lightweight, embedded or client-server
  - Best for: local dev, prototyping, small-medium scale
  - No separate server required (runs in-process)
  - Persistent storage on disk or in-memory
============================================================

INSTALL:
    pip install chromadb openai

NO SIGN UP NEEDED for local mode — just install and run.

SERVER MODE (optional):
    pip install chromadb
    chroma run --path /db/chromadb   ← starts HTTP server on port 8000
    Then use: chromadb.HttpClient(host="localhost", port=8000)

ENV VARS:
    OPENAI_API_KEY = "sk-xxxx..."   (for embeddings/generation)
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ─────────────────────────────────────────────
# 1. CONNECT / CREATE CLIENT
#    Three modes:
#      A) In-memory (ephemeral, lost on restart)
#      B) Persistent (saved to disk)
#      C) HTTP client (connect to running server)
# ─────────────────────────────────────────────

# Mode A: In-memory (for testing)
# client = chromadb.Client()

# Mode B: Persistent on disk ← RECOMMENDED for local RAG
client = chromadb.PersistentClient(path="./chroma_db")   # folder created automatically

# Mode C: Remote server
# client = chromadb.HttpClient(host="localhost", port=8000)

print("Chroma version:", chromadb.__version__)
print("Collections:", client.list_collections())


# ─────────────────────────────────────────────
# 2. EMBEDDING FUNCTION
#    Chroma can auto-embed using its own EmbeddingFunction wrapper.
#    Supports: OpenAI, HuggingFace, Cohere, custom
# ─────────────────────────────────────────────

# Built-in OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small"   # dimension: 1536
)

# OR: HuggingFace (no API key needed, runs locally)
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# openai_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


# ─────────────────────────────────────────────
# 3. CREATE OR GET A COLLECTION
#    - name:               unique identifier
#    - embedding_function: auto-embed on add/query
#    - metadata:           collection-level settings
#    - get_or_create:      safe for repeated runs
# ─────────────────────────────────────────────

collection = client.get_or_create_collection(
    name="rag_documents",
    embedding_function=openai_ef,        # auto-embed everything
    metadata={"hnsw:space": "cosine"}    # distance metric: cosine | l2 | ip
)

print("Collection name:", collection.name)
print("Count:", collection.count())


# ─────────────────────────────────────────────
# 4. ADD DOCUMENTS
#    - documents: list of text strings (Chroma auto-embeds)
#    - ids:       unique string IDs (required, must be unique)
#    - metadatas: list of dicts (optional, for filtering)
#    - embeddings: supply pre-computed vectors (skips auto-embed)
# ─────────────────────────────────────────────

documents = [
    "ChromaDB is an open-source vector database for AI applications.",
    "RAG combines retrieval from a vector store with LLM generation.",
    "Embeddings are dense numerical representations of text meaning.",
    "Chroma supports cosine, L2, and inner product distance metrics.",
    "You can filter Chroma results by metadata fields.",
]

ids = [f"doc_{i}" for i in range(len(documents))]

metadatas = [
    {"source": "chroma_docs", "chapter": 1},
    {"source": "rag_paper",   "chapter": 1},
    {"source": "ml_blog",     "chapter": 2},
    {"source": "chroma_docs", "chapter": 2},
    {"source": "chroma_docs", "chapter": 3},
]

# Add with auto-embedding (uses openai_ef defined on the collection)
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)
print("Added", collection.count(), "documents")


# ─────────────────────────────────────────────
# 5. QUERY (SEMANTIC SEARCH / RETRIEVAL)
#    - query_texts:      Chroma auto-embeds these
#    - n_results:        how many results to return
#    - where:            metadata filter
#    - where_document:   text content filter
# ─────────────────────────────────────────────

def retrieve(query: str, n_results: int = 3, metadata_filter: dict = None) -> list[dict]:
    """
    Semantic search: Chroma embeds the query and finds closest documents.
    Returns list of dicts with text, id, distance, metadata.
    """
    results = collection.query(
        query_texts=[query],           # can pass multiple queries at once
        n_results=n_results,
        where=metadata_filter,         # e.g. {"source": {"$eq": "chroma_docs"}}
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for doc, meta, dist, doc_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0]
    ):
        hits.append({
            "id":       doc_id,
            "text":     doc,
            "metadata": meta,
            "distance": round(dist, 4)   # lower = more similar for cosine/L2
        })
    return hits


# ─────────────────────────────────────────────
# 6. QUERY WITH PRE-COMPUTED EMBEDDINGS
#    Use when you want explicit control over the embedding step
# ─────────────────────────────────────────────

def manual_get_embedding(text: str) -> list[float]:
    resp = oai_client.embeddings.create(input=[text], model="text-embedding-3-small")
    return resp.data[0].embedding

def retrieve_by_vector(query: str, n_results: int = 3) -> list[dict]:
    """Supply your own embedding instead of using the collection's EF."""
    query_vec = manual_get_embedding(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {"text": d, "distance": round(dist, 4)}
        for d, dist in zip(results["documents"][0], results["distances"][0])
    ]


# ─────────────────────────────────────────────
# 7. FULL RAG PIPELINE
# ─────────────────────────────────────────────

def rag_answer(question: str, n_results: int = 3) -> str:
    """
    1. Retrieve relevant chunks from Chroma
    2. Build context string
    3. Call OpenAI to generate answer
    """
    hits = retrieve(question, n_results=n_results)
    context = "\n\n".join([f"[{h['metadata']['source']}] {h['text']}" for h in hits])

    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────
# 8. UPDATE AND DELETE
# ─────────────────────────────────────────────

# Update documents (replaces text + re-embeds)
def update_document(doc_id: str, new_text: str, new_metadata: dict = None):
    collection.update(
        ids=[doc_id],
        documents=[new_text],
        metadatas=[new_metadata] if new_metadata else None
    )

# Upsert (insert or update)
def upsert_document(doc_id: str, text: str, metadata: dict):
    collection.upsert(
        ids=[doc_id],
        documents=[text],
        metadatas=[metadata]
    )

# Delete by ID
def delete_document(doc_id: str):
    collection.delete(ids=[doc_id])

# Delete by metadata filter
def delete_by_filter(metadata_filter: dict):
    # e.g. {"source": {"$eq": "ml_blog"}}
    collection.delete(where=metadata_filter)

# Get specific document by ID
def get_by_id(doc_id: str) -> dict:
    result = collection.get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])
    return {
        "text":      result["documents"][0],
        "metadata":  result["metadatas"][0],
        "embedding": result["embeddings"][0][:5]  # first 5 dims for display
    }

# List all collection names
def list_all_collections():
    return [c.name for c in client.list_collections()]

# Delete a whole collection
def delete_collection(name: str):
    client.delete_collection(name)

# Peek at first N documents
def peek(n: int = 5):
    return collection.peek(limit=n)


# ─────────────────────────────────────────────
# DEMO RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    question = "What is ChromaDB and how is it used for RAG?"

    print("\n=== Semantic Search ===")
    for hit in retrieve(question):
        print(f"  [dist={hit['distance']}] {hit['text']}")

    print("\n=== Filtered Search (chroma_docs only) ===")
    for hit in retrieve(question, metadata_filter={"source": {"$eq": "chroma_docs"}}):
        print(f"  {hit['text']}")

    print("\n=== RAG Answer ===")
    print(rag_answer(question))

    print("\n=== Get by ID ===")
    print(get_by_id("doc_0"))
