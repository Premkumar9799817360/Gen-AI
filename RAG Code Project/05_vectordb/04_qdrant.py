"""
============================================================
  VECTOR DB #4: QDRANT (Open Source + Cloud Paid)
  - High-performance Rust-based vector database
  - Rich filtering with payload (metadata) conditions
  - Named vectors (multiple vector spaces per document)
  - Docker, cloud, or in-memory
============================================================

INSTALL:
    pip install qdrant-client openai

RUN LOCALLY WITH DOCKER:
    docker run -d -p 6333:6333 -p 6334:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage \
        qdrant/qdrant

QDRANT CLOUD:
    Sign up: https://cloud.qdrant.io
    Free tier: 1GB cluster, 1M vectors
    Get: QDRANT_URL       e.g. https://xyz.eu-central-1.aws.cloud.qdrant.io
         QDRANT_API_KEY   (from Qdrant Cloud console)

ENV VARS:
    QDRANT_URL     = "https://your-cluster.cloud.qdrant.io"
    QDRANT_API_KEY = "xxxx"
    OPENAI_API_KEY = "sk-xxxx..."
"""

import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition, MatchValue,
    UpdateStatus, SearchRequest
)
from openai import OpenAI

oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION       = 1536   # text-embedding-3-small output dimension
COLLECTION_NAME = "rag_documents"


# ─────────────────────────────────────────────
# 1. CONNECT TO QDRANT
#    Option A: In-memory (no persistence, for testing)
#    Option B: Local Docker
#    Option C: Qdrant Cloud
# ─────────────────────────────────────────────

# Option A: In-memory
# qdrant = QdrantClient(":memory:")

# Option B: Local Docker
# qdrant = QdrantClient(host="localhost", port=6333)

# Option C: Cloud
qdrant = QdrantClient(
    url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    api_key=os.environ.get("QDRANT_API_KEY")     # None for local
)

print("Qdrant connected. Collections:", [c.name for c in qdrant.get_collections().collections])


# ─────────────────────────────────────────────
# 2. CREATE COLLECTION
#    - vectors_config: defines the vector space
#      - size:     embedding dimension (must match your model!)
#      - distance: Distance.COSINE | Distance.EUCLID | Distance.DOT
# ─────────────────────────────────────────────

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=DIMENSION,
            distance=Distance.COSINE   # most common for text
        )
    )
    print("Collection created:", COLLECTION_NAME)
else:
    print("Collection already exists:", COLLECTION_NAME)


# ─────────────────────────────────────────────
# 3. GENERATE EMBEDDINGS
# ─────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Convert text to embedding vector using OpenAI."""
    text = text.replace("\n", " ")
    resp = oai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding


# ─────────────────────────────────────────────
# 4. UPSERT POINTS (INSERT / UPDATE)
#    Each point has:
#      - id:      integer OR UUID string
#      - vector:  list of floats (the embedding)
#      - payload: dict (like metadata) — searchable & filterable
# ─────────────────────────────────────────────

documents = [
    {"text": "Qdrant is a high-performance vector search engine written in Rust.",       "source": "docs",   "category": "database"},
    {"text": "RAG systems retrieve context before LLM generation for better accuracy.",  "source": "paper",  "category": "ai"},
    {"text": "Named vectors in Qdrant allow multiple embedding spaces per document.",     "source": "docs",   "category": "database"},
    {"text": "Payload filtering enables precise metadata-based search in Qdrant.",       "source": "docs",   "category": "database"},
    {"text": "OpenAI text-embedding-3-small produces 1536-dimensional vectors.",         "source": "openai", "category": "embeddings"},
]

def upsert_documents(docs: list[dict]):
    """Embed and upsert a list of documents."""
    points = []
    for i, doc in enumerate(docs):
        embedding = get_embedding(doc["text"])
        points.append(
            PointStruct(
                id=i,                         # integer ID (or use str(uuid.uuid4()) for UUID)
                vector=embedding,
                payload={                      # any JSON-serializable data
                    "text":     doc["text"],
                    "source":   doc["source"],
                    "category": doc["category"]
                }
            )
        )

    result = qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True       # wait for indexing to complete before returning
    )
    print("Upsert status:", result.status)

upsert_documents(documents)


# ─────────────────────────────────────────────
# 5. SEARCH (VECTOR SIMILARITY)
#    - query_vector:       the embedded query
#    - limit:              top-k results
#    - query_filter:       Filter on payload fields
#    - with_payload:       return payload (metadata)
#    - score_threshold:    only return results above this score
# ─────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3, category_filter: str = None) -> list[dict]:
    """
    Embed the query and search Qdrant for the nearest vectors.
    Optionally filter by the 'category' payload field.
    """
    query_vec = get_embedding(query)

    # Build optional filter
    search_filter = None
    if category_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="category",                    # payload field name
                    match=MatchValue(value=category_filter)
                )
            ]
        )

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,          # include the payload dict in results
        score_threshold=0.0         # filter out results below this cosine score
    )

    return [
        {
            "id":       hit.id,
            "score":    round(hit.score, 4),    # cosine similarity 0–1
            "text":     hit.payload["text"],
            "source":   hit.payload["source"],
            "category": hit.payload["category"]
        }
        for hit in results
    ]


# ─────────────────────────────────────────────
# 6. BATCH SEARCH (multiple queries at once)
# ─────────────────────────────────────────────

def batch_retrieve(queries: list[str], top_k: int = 3) -> list[list[dict]]:
    """Run multiple searches in one API call."""
    query_vecs = [get_embedding(q) for q in queries]
    requests = [
        SearchRequest(vector=vec, limit=top_k, with_payload=True)
        for vec in query_vecs
    ]
    batch_results = qdrant.search_batch(
        collection_name=COLLECTION_NAME,
        requests=requests
    )
    all_results = []
    for query_results in batch_results:
        all_results.append([
            {"text": hit.payload["text"], "score": round(hit.score, 4)}
            for hit in query_results
        ])
    return all_results


# ─────────────────────────────────────────────
# 7. FULL RAG PIPELINE
# ─────────────────────────────────────────────

def rag_answer(question: str) -> str:
    """
    1. Retrieve relevant chunks from Qdrant
    2. Build context string
    3. Generate answer with OpenAI
    """
    hits = retrieve(question, top_k=3)
    context = "\n\n".join([f"[{h['source']}] {h['text']}" for h in hits])

    prompt = f"""Use the following retrieved context to answer the question accurately.
If you cannot find the answer in the context, say "I don't know."

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
# 8. OTHER IMPORTANT FUNCTIONS
# ─────────────────────────────────────────────

# Retrieve a point by ID
def get_point(point_id: int) -> dict:
    results = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
        with_vectors=False
    )
    return results[0].payload if results else None

# Delete points by ID
def delete_points(point_ids: list[int]):
    from qdrant_client.models import PointIdsList
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=point_ids)
    )

# Delete points matching a filter
def delete_by_filter(category: str):
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )
    )

# Update payload only (no re-embedding)
def update_payload(point_id: int, new_payload: dict):
    qdrant.set_payload(
        collection_name=COLLECTION_NAME,
        payload=new_payload,
        points=[point_id]
    )

# Count total vectors
def count_vectors() -> int:
    info = qdrant.get_collection(COLLECTION_NAME)
    return info.vectors_count

# Scroll (list) all points with optional filter
def scroll_points(limit: int = 10) -> list[dict]:
    results, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        with_payload=True
    )
    return [{"id": p.id, "text": p.payload["text"]} for p in results]


# ─────────────────────────────────────────────
# DEMO RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    question = "What makes Qdrant fast for vector search?"

    print("\n=== Semantic Search ===")
    for hit in retrieve(question):
        print(f"  [score={hit['score']}] {hit['text']}")

    print("\n=== Filtered (database only) ===")
    for hit in retrieve(question, category_filter="database"):
        print(f"  {hit['text']}")

    print("\n=== RAG Answer ===")
    print(rag_answer(question))

    print("\n=== Collection size ===")
    print("Vectors:", count_vectors())
