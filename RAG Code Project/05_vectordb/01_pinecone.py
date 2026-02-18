"""
============================================================
  VECTOR DB #1: PINECONE (Paid / Free Tier Available)
  - Fully managed, cloud-native vector database
  - Supports metadata filtering, namespaces
  - REST + gRPC API
============================================================

INSTALL:
    pip install pinecone-client openai

SIGN UP & GET API KEY:
    https://app.pinecone.io  →  API Keys tab
    Free tier: 1 index, 100k vectors

ENV VARS:
    PINECONE_API_KEY   = "pcsk_xxxx..."
    OPENAI_API_KEY     = "sk-xxxx..."   (only needed for embedding)
"""

import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# ─────────────────────────────────────────────
# 1. CONNECT TO PINECONE
# ─────────────────────────────────────────────

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# ─────────────────────────────────────────────
# 2. CREATE OR CONNECT TO AN INDEX
#    - name:       your index name (lowercase, no spaces)
#    - dimension:  must match your embedding model output
#                  text-embedding-3-small  → 1536
#                  text-embedding-ada-002  → 1536
#                  all-MiniLM-L6-v2        → 384
#    - metric:     "cosine" | "euclidean" | "dotproduct"
#    - spec:       where the index lives (serverless or pod)
# ─────────────────────────────────────────────

INDEX_NAME = "rag-demo"
DIMENSION  = 1536          # matches text-embedding-3-small
METRIC     = "cosine"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud="aws",        # "aws" | "gcp" | "azure"
            region="us-east-1"  # pick the region closest to you
        )
    )

# Get a handle to the index
index = pc.Index(INDEX_NAME)
print("Index stats:", index.describe_index_stats())


# ─────────────────────────────────────────────
# 3. GENERATE EMBEDDINGS WITH OPENAI
# ─────────────────────────────────────────────

oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    Convert a string into a vector (list of floats).
    The dimension MUST match the Pinecone index dimension.
    """
    text = text.replace("\n", " ")
    response = oai_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding   # returns list[float] of length 1536


# ─────────────────────────────────────────────
# 4. UPSERT (ADD / UPDATE VECTORS)
#    Each record is a tuple: (id, vector, metadata)
#    - id:       unique string identifier
#    - vector:   list of floats
#    - metadata: dict with any filterable fields
# ─────────────────────────────────────────────

documents = [
    {"id": "doc1", "text": "Pinecone is a managed vector database for AI applications.",   "source": "docs"},
    {"id": "doc2", "text": "RAG combines retrieval with language model generation.",         "source": "paper"},
    {"id": "doc3", "text": "Embeddings encode semantic meaning into numerical vectors.",     "source": "blog"},
]

vectors_to_upsert = []
for doc in documents:
    emb = get_embedding(doc["text"])
    vectors_to_upsert.append({
        "id":       doc["id"],
        "values":   emb,
        "metadata": {"text": doc["text"], "source": doc["source"]}
    })

# Upsert in batches (Pinecone recommends ≤100 per call)
index.upsert(vectors=vectors_to_upsert, namespace="default")
print("Upserted", len(vectors_to_upsert), "vectors")


# ─────────────────────────────────────────────
# 5. QUERY (SEMANTIC SEARCH / RETRIEVAL)
#    - vector:          the query embedding
#    - top_k:           how many results to return
#    - namespace:       logical partition (optional)
#    - filter:          metadata filter (optional)
#    - include_metadata: return stored metadata
# ─────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3, filter_dict: dict = None) -> list[dict]:
    """
    Convert query to embedding, then find the top_k closest vectors.
    Returns list of dicts with id, score, and metadata.
    """
    query_emb = get_embedding(query)

    results = index.query(
        vector=query_emb,
        top_k=top_k,
        namespace="default",
        filter=filter_dict,         # e.g. {"source": {"$eq": "docs"}}
        include_metadata=True
    )

    hits = []
    for match in results["matches"]:
        hits.append({
            "id":    match["id"],
            "score": round(match["score"], 4),   # cosine similarity: 0–1 (higher = more similar)
            "text":  match["metadata"]["text"],
            "source": match["metadata"]["source"]
        })
    return hits


# ─────────────────────────────────────────────
# 6. FULL RAG PIPELINE
#    Retrieve relevant chunks → build prompt → call LLM
# ─────────────────────────────────────────────

def rag_answer(question: str) -> str:
    """
    1. Embed the question
    2. Retrieve top-k relevant chunks from Pinecone
    3. Build a context string
    4. Call OpenAI Chat to generate an answer
    """
    # Step 1 & 2: Retrieve
    hits = retrieve(question, top_k=3)
    context = "\n\n".join([f"[{h['source']}] {h['text']}" for h in hits])

    # Step 3: Build prompt
    prompt = f"""Use the following context to answer the question.
If the context doesn't contain the answer, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

    # Step 4: Generate
    response = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# 7. OTHER IMPORTANT FUNCTIONS
# ─────────────────────────────────────────────

# DELETE individual vectors
def delete_vectors(ids: list[str]):
    index.delete(ids=ids, namespace="default")

# FETCH a vector by ID
def fetch_vector(vector_id: str):
    return index.fetch(ids=[vector_id], namespace="default")

# UPDATE metadata only (re-upsert same vector with new metadata)
def update_metadata(vector_id: str, new_metadata: dict, original_text: str):
    emb = get_embedding(original_text)
    index.upsert(vectors=[{"id": vector_id, "values": emb, "metadata": new_metadata}])

# LIST all namespaces
def list_namespaces():
    stats = index.describe_index_stats()
    return list(stats["namespaces"].keys())

# DELETE entire index (irreversible!)
def delete_index():
    pc.delete_index(INDEX_NAME)


# ─────────────────────────────────────────────
# DEMO RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    question = "What is RAG and how does it work?"
    print("\nQuestion:", question)
    print("\nRelevant chunks:")
    for hit in retrieve(question):
        print(f"  [{hit['score']}] {hit['text']}")
    print("\nRAG Answer:", rag_answer(question))
