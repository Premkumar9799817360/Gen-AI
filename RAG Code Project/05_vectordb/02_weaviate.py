"""
============================================================
  VECTOR DB #2: WEAVIATE (Open Source + Cloud Paid)
  - Open source, GraphQL + REST API
  - Built-in vectorization modules (OpenAI, Cohere, HuggingFace)
  - Hybrid search (vector + BM25 keyword)
  - Docker or Weaviate Cloud (WCS)
============================================================

INSTALL:
    pip install weaviate-client openai

RUN LOCALLY WITH DOCKER:
    docker run -d \
      -p 8080:8080 -p 50051:50051 \
      -e ENABLE_MODULES=text2vec-openai \
      -e OPENAI_APIKEY=your_key \
      cr.weaviate.io/semitechnologies/weaviate:latest

WEAVIATE CLOUD (WCS):
    Sign up: https://console.weaviate.cloud
    Free sandbox: 14-day trial
    Get: WEAVIATE_URL  (e.g. https://xyz.weaviate.network)
         WEAVIATE_API_KEY

ENV VARS:
    WEAVIATE_URL     = "https://your-cluster.weaviate.network"
    WEAVIATE_API_KEY = "xxxx"  (cloud only, leave empty for local)
    OPENAI_API_KEY   = "sk-xxxx..."
"""

import os
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure
from openai import OpenAI

oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ─────────────────────────────────────────────
# 1. CONNECT TO WEAVIATE
#    Option A: Local Docker
#    Option B: Weaviate Cloud (WCS)
# ─────────────────────────────────────────────

# Option A — Local
# client = weaviate.connect_to_local()   # default: localhost:8080

# Option B — Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ["WEAVIATE_URL"],
    auth_credentials=weaviate.auth.AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
    headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]}  # for built-in vectorizer
)

print("Weaviate ready:", client.is_ready())


# ─────────────────────────────────────────────
# 2. CREATE A COLLECTION (= "class" / "table")
#    Define schema with properties and vectorizer
# ─────────────────────────────────────────────

COLLECTION_NAME = "Document"

# Delete if exists (for clean demo)
if client.collections.exists(COLLECTION_NAME):
    client.collections.delete(COLLECTION_NAME)

collection = client.collections.create(
    name=COLLECTION_NAME,
    # Built-in vectorizer: Weaviate calls OpenAI automatically on insert
    vectorizer_config=Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"
    ),
    # OR use no vectorizer if you supply your own vectors:
    # vectorizer_config=Configure.Vectorizer.none(),

    generative_config=Configure.Generative.openai(model="gpt-4o-mini"),  # for RAG

    properties=[
        Property(name="text",   data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
        Property(name="chunk_index", data_type=DataType.INT),
    ]
)
print("Collection created:", COLLECTION_NAME)


# ─────────────────────────────────────────────
# 3. GENERATE EMBEDDINGS (manual, if no built-in vectorizer)
# ─────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Generate embedding using OpenAI manually."""
    text = text.replace("\n", " ")
    resp = oai_client.embeddings.create(input=[text], model="text-embedding-3-small")
    return resp.data[0].embedding


# ─────────────────────────────────────────────
# 4. INSERT OBJECTS
#    When using built-in vectorizer: just pass properties, Weaviate vectorizes
#    When using none vectorizer:     pass vector= explicitly
# ─────────────────────────────────────────────

documents = [
    {"text": "Weaviate is an open-source vector database with hybrid search.",   "source": "docs",  "chunk_index": 0},
    {"text": "RAG retrieves relevant context before generating an LLM response.", "source": "paper", "chunk_index": 0},
    {"text": "Embeddings map text into high-dimensional vector spaces.",           "source": "blog",  "chunk_index": 0},
    {"text": "Weaviate supports GraphQL queries for flexible data retrieval.",     "source": "docs",  "chunk_index": 1},
]

# METHOD A: Built-in vectorizer (Weaviate vectorizes automatically)
with collection.batch.dynamic() as batch:
    for doc in documents:
        batch.add_object(properties=doc)
        # Optionally supply your own vector:
        # batch.add_object(properties=doc, vector=get_embedding(doc["text"]))

print("Inserted", len(documents), "documents")


# ─────────────────────────────────────────────
# 5. QUERY — THREE MODES
# ─────────────────────────────────────────────

# -- 5A: NEAR TEXT (vector search using the vectorizer)
def near_text_search(query: str, limit: int = 3) -> list[dict]:
    """
    Semantic search: finds objects whose vectors are closest to the query.
    The built-in vectorizer embeds the query automatically.
    """
    results = collection.query.near_text(
        query=query,
        limit=limit,
        return_metadata=wvc.query.MetadataQuery(distance=True),
        return_properties=["text", "source"]
    )
    return [
        {"text": obj.properties["text"],
         "source": obj.properties["source"],
         "distance": obj.metadata.distance}
        for obj in results.objects
    ]


# -- 5B: NEAR VECTOR (supply your own vector)
def near_vector_search(query_text: str, limit: int = 3) -> list[dict]:
    """Use this when vectorizer_config = none."""
    query_vec = get_embedding(query_text)
    results = collection.query.near_vector(
        near_vector=query_vec,
        limit=limit,
        return_metadata=wvc.query.MetadataQuery(distance=True),
        return_properties=["text", "source"]
    )
    return [
        {"text": obj.properties["text"],
         "source": obj.properties["source"],
         "distance": obj.metadata.distance}
        for obj in results.objects
    ]


# -- 5C: HYBRID SEARCH (vector + BM25 keyword, combined score)
def hybrid_search(query: str, alpha: float = 0.7, limit: int = 3) -> list[dict]:
    """
    alpha=1.0 → pure vector search
    alpha=0.0 → pure BM25 keyword search
    alpha=0.7 → 70% vector, 30% BM25 (good default)
    """
    results = collection.query.hybrid(
        query=query,
        alpha=alpha,
        limit=limit,
        return_metadata=wvc.query.MetadataQuery(score=True),
        return_properties=["text", "source"]
    )
    return [
        {"text": obj.properties["text"],
         "source": obj.properties["source"],
         "score": obj.metadata.score}
        for obj in results.objects
    ]


# ─────────────────────────────────────────────
# 6. FULL RAG WITH WEAVIATE GENERATIVE MODULE
#    Weaviate can call the LLM for you — "Generate" queries
# ─────────────────────────────────────────────

def rag_generate(question: str) -> str:
    """
    Uses Weaviate's built-in RAG:
    1. Retrieves top-k objects via vector search
    2. Passes them directly to OpenAI for a grouped answer
    """
    result = collection.generate.near_text(
        query=question,
        limit=3,
        grouped_task=f"Answer this question using the provided context: {question}"
    )
    return result.generated


# ─────────────────────────────────────────────
# 7. MANUAL RAG PIPELINE
# ─────────────────────────────────────────────

def rag_manual(question: str) -> str:
    hits = near_text_search(question, limit=3)
    context = "\n\n".join([f"[{h['source']}] {h['text']}" for h in hits])
    prompt = f"""Context:\n{context}\n\nQuestion: {question}\nAnswer:"""
    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────
# 8. OTHER IMPORTANT FUNCTIONS
# ─────────────────────────────────────────────

# Filter by metadata
def filtered_search(query: str, source_filter: str) -> list[dict]:
    from weaviate.classes.query import Filter
    results = collection.query.near_text(
        query=query,
        limit=3,
        filters=Filter.by_property("source").equal(source_filter),
        return_properties=["text", "source"]
    )
    return [{"text": o.properties["text"]} for o in results.objects]

# Count objects
def count_objects() -> int:
    return collection.aggregate.over_all(total_count=True).total_count

# Delete an object by UUID
def delete_object(uuid: str):
    collection.data.delete_by_id(uuid)

# Close the client when done
def close():
    client.close()


# ─────────────────────────────────────────────
# DEMO RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    query = "How does hybrid search work in vector databases?"

    print("\n=== Near Text Search ===")
    for r in near_text_search(query):
        print(f"  [{r['distance']:.4f}] {r['text']}")

    print("\n=== Hybrid Search ===")
    for r in hybrid_search(query):
        print(f"  [score={r['score']:.4f}] {r['text']}")

    print("\n=== RAG Answer ===")
    print(rag_generate(query))

    client.close()
