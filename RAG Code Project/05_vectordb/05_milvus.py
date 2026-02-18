"""
============================================================
  VECTOR DB #5: MILVUS (Open Source + Zilliz Cloud Paid)
  - Battle-tested, highly scalable (billions of vectors)
  - Multiple index types (HNSW, IVF_FLAT, IVF_SQ8, etc.)
  - Supports dynamic fields, partitions, and replicas
  - Milvus Lite: runs in-process (like SQLite for vectors)
============================================================

INSTALL:
    pip install pymilvus openai

MILVUS LITE (no server needed — embedded in Python):
    from pymilvus import MilvusClient
    client = MilvusClient("./milvus.db")

DOCKER (full Milvus):
    docker run -d -p 19530:19530 -p 9091:9091 \
        milvusdb/milvus:v2.4.0 milvus run standalone

ZILLIZ CLOUD (managed Milvus):
    Sign up: https://cloud.zilliz.com
    Free: 1M vectors, 2 CUs
    Get: ZILLIZ_URI     e.g. https://in01-xxx.api.gcp-us-west1.zillizcloud.com:19541
         ZILLIZ_TOKEN   (from Zilliz console)

ENV VARS:
    MILVUS_URI    = "http://localhost:19530"          (local)
    ZILLIZ_URI    = "https://xyz.zillizcloud.com:..."  (cloud)
    ZILLIZ_TOKEN  = "xxxx"
    OPENAI_API_KEY = "sk-xxxx..."
"""

import os
from pymilvus import (
    MilvusClient,
    DataType,
    Collection,
    connections,
    utility
)
from openai import OpenAI

oai_client    = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EMBEDDING_DIM = 1536
COLLECTION_NAME = "rag_documents"


# ─────────────────────────────────────────────
# 1. CONNECT (Three Options)
# ─────────────────────────────────────────────

# Option A: Milvus Lite — embedded, no server needed (best for dev)
client = MilvusClient("./milvus_lite.db")   # stores to local file

# Option B: Full Milvus (Docker)
# client = MilvusClient(uri="http://localhost:19530")

# Option C: Zilliz Cloud
# client = MilvusClient(
#     uri=os.environ["ZILLIZ_URI"],
#     token=os.environ["ZILLIZ_TOKEN"]
# )

print("Connected to Milvus")
print("Collections:", client.list_collections())


# ─────────────────────────────────────────────
# 2. CREATE COLLECTION
#    Two approaches:
#      A) Quick (auto schema) — just specify dimension
#      B) Full schema — explicit field definitions
# ─────────────────────────────────────────────

# ── Approach A: Quick schema (simplest)
# client.create_collection(
#     collection_name=COLLECTION_NAME,
#     dimension=EMBEDDING_DIM
# )

# ── Approach B: Full custom schema
if COLLECTION_NAME in client.list_collections():
    client.drop_collection(COLLECTION_NAME)   # drop for clean demo

from pymilvus import MilvusClient
from pymilvus.orm.types import DataType

client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=client.create_schema(
        auto_id=False,               # we supply our own IDs
        enable_dynamic_field=True,   # allow adding extra fields at insert time
        description="RAG document store"
    ).add_field("id",        DataType.INT64,          is_primary=True)
     .add_field("vector",    DataType.FLOAT_VECTOR,   dim=EMBEDDING_DIM)
     .add_field("text",      DataType.VARCHAR,        max_length=4096)
     .add_field("source",    DataType.VARCHAR,        max_length=256)
     .add_field("category",  DataType.VARCHAR,        max_length=128),

    # Index built at creation time — defines ANN algorithm
    index_params=client.prepare_index_params().add_index(
        field_name="vector",
        index_type="HNSW",      # Options: HNSW (best recall) | IVF_FLAT | IVF_SQ8
        metric_type="COSINE",   # COSINE | L2 | IP (inner product)
        params={"M": 16, "efConstruction": 200}   # HNSW params
    )
)
print("Collection created:", COLLECTION_NAME)


# ─────────────────────────────────────────────
# 3. GENERATE EMBEDDINGS
# ─────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Convert text to 1536-dim vector using OpenAI."""
    text = text.replace("\n", " ")
    resp = oai_client.embeddings.create(input=[text], model="text-embedding-3-small")
    return resp.data[0].embedding


# ─────────────────────────────────────────────
# 4. INSERT DATA
#    Data format: list of dicts (column-based or row-based)
#    Each dict key must match a schema field name
# ─────────────────────────────────────────────

documents = [
    {"text": "Milvus is a cloud-native vector database built for scale.",              "source": "docs",   "category": "database"},
    {"text": "RAG systems reduce LLM hallucinations by grounding answers in data.",    "source": "paper",  "category": "ai"},
    {"text": "HNSW index offers best recall/speed tradeoff for vector search.",        "source": "docs",   "category": "indexing"},
    {"text": "Milvus Lite embeds the full Milvus engine into a Python process.",       "source": "docs",   "category": "database"},
    {"text": "Zilliz Cloud provides a fully managed Milvus service.",                  "source": "zilliz", "category": "cloud"},
]

def insert_documents(docs: list[dict]):
    data = []
    for i, doc in enumerate(docs):
        data.append({
            "id":       i,
            "vector":   get_embedding(doc["text"]),
            "text":     doc["text"],
            "source":   doc["source"],
            "category": doc["category"]
        })

    result = client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"Inserted {result['insert_count']} vectors. IDs: {result['ids']}")

insert_documents(documents)


# ─────────────────────────────────────────────
# 5. SEARCH (ANN VECTOR SEARCH)
#    - data:             query vector(s) — list of list[float]
#    - limit:            top-k results
#    - search_params:    algorithm-specific (ef for HNSW)
#    - filter:           boolean expression for metadata
#    - output_fields:    which fields to return
# ─────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3, category_filter: str = None) -> list[dict]:
    """
    Embed the query and run ANN search.
    Optionally filter results by 'category' field.
    """
    query_vec = get_embedding(query)

    # Filter expression syntax: SQL-like string
    filter_expr = f'category == "{category_filter}"' if category_filter else None

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],                  # can pass multiple queries
        limit=top_k,
        filter=filter_expr,
        search_params={"ef": 64},          # HNSW search param (higher = better recall)
        output_fields=["text", "source", "category"]
    )

    hits = []
    for hit in results[0]:                  # results[0] = first query's results
        hits.append({
            "id":       hit["id"],
            "distance": round(hit["distance"], 4),  # cosine: higher = more similar
            "text":     hit["entity"]["text"],
            "source":   hit["entity"]["source"],
            "category": hit["entity"]["category"]
        })
    return hits


# ─────────────────────────────────────────────
# 6. QUERY (EXACT MATCH — like a traditional DB)
#    For fetching specific IDs or metadata-only queries
# ─────────────────────────────────────────────

def query_by_filter(filter_expr: str, output_fields: list = None) -> list[dict]:
    """
    Non-vector query: filter by metadata expression.
    filter_expr examples:
      'source == "docs"'
      'category in ["database", "indexing"]'
      'id in [0, 1, 2]'
    """
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter=filter_expr,
        output_fields=output_fields or ["text", "source", "category"]
    )
    return results


# ─────────────────────────────────────────────
# 7. FULL RAG PIPELINE
# ─────────────────────────────────────────────

def rag_answer(question: str) -> str:
    """
    1. Retrieve from Milvus
    2. Build context
    3. Generate with OpenAI
    """
    hits = retrieve(question, top_k=3)
    context = "\n\n".join([f"[{h['source']}] {h['text']}" for h in hits])

    prompt = f"""Use the context to answer accurately. Say "I don't know" if the answer
is not in the context.

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

# Get a point by ID
def get_by_id(point_id: int) -> dict:
    return client.get(collection_name=COLLECTION_NAME, ids=[point_id])

# Delete by IDs
def delete_by_ids(ids: list[int]):
    client.delete(collection_name=COLLECTION_NAME, ids=ids)

# Delete by filter expression
def delete_by_filter(filter_expr: str):
    client.delete(collection_name=COLLECTION_NAME, filter=filter_expr)

# Upsert (insert or replace)
def upsert_document(doc_id: int, text: str, source: str, category: str):
    client.upsert(
        collection_name=COLLECTION_NAME,
        data=[{
            "id":       doc_id,
            "vector":   get_embedding(text),
            "text":     text,
            "source":   source,
            "category": category
        }]
    )

# Count total entities
def count_entities() -> int:
    return client.get_collection_stats(COLLECTION_NAME)["row_count"]

# Get collection info
def collection_info():
    return client.describe_collection(COLLECTION_NAME)


# ─────────────────────────────────────────────
# DEMO RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    question = "What index type should I use in Milvus for best recall?"

    print("\n=== ANN Vector Search ===")
    for hit in retrieve(question):
        print(f"  [dist={hit['distance']}] {hit['text']}")

    print("\n=== Filtered Search (indexing category) ===")
    for hit in retrieve(question, category_filter="indexing"):
        print(f"  {hit['text']}")

    print("\n=== Exact Filter Query ===")
    for r in query_by_filter('source == "docs"'):
        print(f"  {r['text']}")

    print("\n=== RAG Answer ===")
    print(rag_answer(question))

    print("\n=== Total vectors ===")
    print(count_entities())
