# 🔍 Vector Databases — Complete RAG Guide (Top 6)

A hands-on reference for every major vector database: how to connect, pass embeddings, insert data, and run full RAG pipelines. Each database has its own Python file with fully commented code.

---

## 📁 Files

| File | Database | Type | Best For |
|------|----------|------|----------|
| `01_pinecone.py` | **Pinecone** | ☁️ Managed SaaS | Production, simplicity |
| `02_weaviate.py` | **Weaviate** | 🐳 Open Source + Cloud | Hybrid search, built-in vectorizer |
| `03_chromadb.py` | **ChromaDB** | 🆓 Open Source | Local dev, prototyping |
| `04_qdrant.py` | **Qdrant** | 🆓 Open Source + Cloud | Performance, rich filtering |
| `05_milvus.py` | **Milvus** | 🆓 Open Source + Cloud | Billion-scale, enterprise |
| `06_pgvector.py` | **pgvector** | 🆓 PostgreSQL Extension | SQL+vector, existing Postgres |

---

## 🚀 Quick Start (Any Database)

```bash
# 1. Install base dependencies
pip install openai

# 2. Set your OpenAI key (for embeddings)
export OPENAI_API_KEY="sk-xxxx..."

# 3. Install the specific DB client (see each section below)
# 4. Set DB-specific env vars
# 5. Run the Python file
python 01_pinecone.py
```

---

## 🧠 How RAG Works (Overview)

```
User Question
      │
      ▼
┌─────────────────┐
│  Embed Query    │  ← convert text to vector (e.g. 1536 floats)
└────────┬────────┘
         │  query vector
         ▼
┌─────────────────┐
│  Vector Search  │  ← find top-k nearest vectors in the DB
└────────┬────────┘
         │  relevant chunks
         ▼
┌─────────────────┐
│  Build Prompt   │  ← "Context: <chunks>\n\nQuestion: <question>"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Generate  │  ← GPT-4o-mini, Claude, etc.
└────────┬────────┘
         │
         ▼
      Answer
```

---

## 1️⃣ Pinecone

**Type:** Fully managed SaaS  
**Free tier:** 1 index, 100k vectors  
**Pricing:** https://www.pinecone.io/pricing  
**Dashboard:** https://app.pinecone.io

### Install
```bash
pip install pinecone-client openai
```

### Environment Variables
```bash
export PINECONE_API_KEY="pcsk_xxxx..."
export OPENAI_API_KEY="sk-xxxx..."
```

### Connect & Create Index
```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_API_KEY")

pc.create_index(
    name="my-index",
    dimension=1536,          # must match embedding model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("my-index")
```

### Pass Embeddings & Upsert
```python
# Generate embedding
import openai
response = openai.embeddings.create(input=["Hello world"], model="text-embedding-3-small")
vector = response.data[0].embedding   # list of 1536 floats

# Upsert: (id, vector, metadata)
index.upsert(vectors=[{
    "id":       "doc1",
    "values":   vector,
    "metadata": {"text": "Hello world", "source": "blog"}
}])
```

### Query (Retrieve)
```python
results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)
# results["matches"] → list of {id, score, metadata}
```

### Key Functions
| Function | What It Does |
|----------|-------------|
| `pc.create_index()` | Create a new index |
| `index.upsert()` | Insert or update vectors |
| `index.query()` | Vector similarity search |
| `index.fetch()` | Get vector by ID |
| `index.delete()` | Delete vectors by ID |
| `index.describe_index_stats()` | Count vectors, namespaces |

---

## 2️⃣ Weaviate

**Type:** Open source + Weaviate Cloud (WCS)  
**Free tier:** 14-day sandbox on WCS  
**Docker image:** `cr.weaviate.io/semitechnologies/weaviate:latest`  
**Docs:** https://weaviate.io/developers/weaviate

### Install
```bash
pip install weaviate-client openai
```

### Environment Variables
```bash
export WEAVIATE_URL="https://your-cluster.weaviate.network"
export WEAVIATE_API_KEY="xxxx"        # only for Weaviate Cloud
export OPENAI_API_KEY="sk-xxxx..."
```

### Connect
```python
import weaviate

# Local Docker
client = weaviate.connect_to_local()

# Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="https://xyz.weaviate.network",
    auth_credentials=weaviate.auth.AuthApiKey("YOUR_KEY"),
    headers={"X-OpenAI-Api-Key": "sk-xxxx"}
)
```

### Run with Docker
```bash
docker run -d \
  -p 8080:8080 -p 50051:50051 \
  -e ENABLE_MODULES=text2vec-openai \
  -e OPENAI_APIKEY=your_openai_key \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

### Create Collection & Pass Embeddings
```python
from weaviate.classes.config import Configure, Property, DataType

# With built-in vectorizer (Weaviate calls OpenAI automatically)
collection = client.collections.create(
    name="Document",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(model="text-embedding-3-small"),
    properties=[Property(name="text", data_type=DataType.TEXT)]
)

# Insert — no manual embedding needed!
collection.data.insert({"text": "Hello world"})

# Query
results = collection.query.near_text(query="what is RAG", limit=3)
```

### Key Functions
| Function | What It Does |
|----------|-------------|
| `client.collections.create()` | Create schema + collection |
| `collection.data.insert()` | Add single object |
| `collection.batch.dynamic()` | Batch insert |
| `collection.query.near_text()` | Vector search (auto-embed) |
| `collection.query.near_vector()` | Vector search (manual embed) |
| `collection.query.hybrid()` | Vector + BM25 keyword search |
| `collection.generate.near_text()` | Built-in RAG (calls LLM too!) |

---

## 3️⃣ ChromaDB

**Type:** Open source (embedded or server)  
**Free tier:** Fully free, runs locally  
**No account needed** for local mode  
**Docs:** https://docs.trychroma.com

### Install
```bash
pip install chromadb openai
```

### Environment Variables
```bash
export OPENAI_API_KEY="sk-xxxx..."    # only for embeddings
```

### Connect (Three Modes)
```python
import chromadb

# In-memory (ephemeral)
client = chromadb.Client()

# Persistent on disk ← recommended
client = chromadb.PersistentClient(path="./chroma_db")

# Remote server
client = chromadb.HttpClient(host="localhost", port=8000)
```

### Start Server (optional)
```bash
pip install chromadb
chroma run --path /db/chromadb   # starts on port 8000
```

### Create Collection & Pass Embeddings
```python
from chromadb.utils import embedding_functions

# Auto-embed with OpenAI
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-xxxx...",
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="my_docs",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}
)

# Add — Chroma embeds automatically
collection.add(
    documents=["Hello world", "RAG is cool"],
    ids=["id1", "id2"],
    metadatas=[{"source": "blog"}, {"source": "paper"}]
)

# Query
results = collection.query(query_texts=["what is RAG"], n_results=3)
# results["documents"][0] → list of text strings
# results["distances"][0] → list of distances
```

### Key Functions
| Function | What It Does |
|----------|-------------|
| `client.get_or_create_collection()` | Create or open collection |
| `collection.add()` | Insert documents |
| `collection.query()` | Vector similarity search |
| `collection.get()` | Fetch by ID |
| `collection.update()` | Update existing document |
| `collection.upsert()` | Insert or update |
| `collection.delete()` | Delete by ID or filter |
| `collection.count()` | Count all documents |

---

## 4️⃣ Qdrant

**Type:** Open source + Qdrant Cloud  
**Free tier:** 1 GB cluster free forever on cloud  
**Docker image:** `qdrant/qdrant`  
**Cloud:** https://cloud.qdrant.io  
**Docs:** https://qdrant.tech/documentation

### Install
```bash
pip install qdrant-client openai
```

### Environment Variables
```bash
export QDRANT_URL="https://xyz.cloud.qdrant.io"
export QDRANT_API_KEY="xxxx"
export OPENAI_API_KEY="sk-xxxx..."
```

### Connect
```python
from qdrant_client import QdrantClient

# In-memory
client = QdrantClient(":memory:")

# Local Docker
client = QdrantClient(host="localhost", port=6333)

# Qdrant Cloud
client = QdrantClient(
    url="https://xyz.cloud.qdrant.io",
    api_key="YOUR_API_KEY"
)
```

### Run with Docker
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### Create Collection & Pass Embeddings
```python
from qdrant_client.models import Distance, VectorParams, PointStruct

client.create_collection(
    collection_name="my_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Insert (you generate the embedding yourself)
embedding = openai_client.embeddings.create(...).data[0].embedding
client.upsert(
    collection_name="my_docs",
    points=[PointStruct(id=0, vector=embedding, payload={"text": "...", "source": "docs"})]
)

# Query
results = client.search(
    collection_name="my_docs",
    query_vector=query_embedding,
    limit=3,
    with_payload=True
)
# results[i].score → cosine similarity
# results[i].payload → metadata dict
```

### Key Functions
| Function | What It Does |
|----------|-------------|
| `client.create_collection()` | Create vector space |
| `client.upsert()` | Insert or update points |
| `client.search()` | ANN vector search |
| `client.search_batch()` | Multiple searches in one call |
| `client.retrieve()` | Fetch by ID |
| `client.delete()` | Delete by ID or filter |
| `client.scroll()` | Page through all records |
| `client.set_payload()` | Update metadata without re-embedding |

---

## 5️⃣ Milvus

**Type:** Open source + Zilliz Cloud  
**Free tier:** Milvus Lite (local), Zilliz Cloud (1M free vectors)  
**Docker image:** `milvusdb/milvus`  
**Cloud:** https://cloud.zilliz.com  
**Docs:** https://milvus.io/docs

### Install
```bash
pip install pymilvus openai
```

### Environment Variables
```bash
export ZILLIZ_URI="https://in01-xxx.api.gcp-us-west1.zillizcloud.com:19541"
export ZILLIZ_TOKEN="xxxx"
export OPENAI_API_KEY="sk-xxxx..."
```

### Connect
```python
from pymilvus import MilvusClient

# Milvus Lite (no server needed — saves to .db file)
client = MilvusClient("./milvus.db")

# Local Docker
client = MilvusClient(uri="http://localhost:19530")

# Zilliz Cloud
client = MilvusClient(
    uri="https://xyz.zillizcloud.com:19541",
    token="YOUR_TOKEN"
)
```

### Run with Docker
```bash
docker run -d -p 19530:19530 -p 9091:9091 \
    milvusdb/milvus:v2.4.0 milvus run standalone
```

### Create Collection & Pass Embeddings
```python
# Quick (auto-schema)
client.create_collection(collection_name="my_docs", dimension=1536)

# Insert
embedding = openai_client.embeddings.create(...).data[0].embedding
client.insert(
    collection_name="my_docs",
    data=[{"id": 0, "vector": embedding, "text": "...", "source": "docs"}]
)

# Search
results = client.search(
    collection_name="my_docs",
    data=[query_embedding],
    limit=3,
    output_fields=["text", "source"],
    filter='source == "docs"'          # SQL-like filter
)
# results[0][i]["entity"] → metadata dict
# results[0][i]["distance"] → cosine score
```

### Key Functions
| Function | What It Does |
|----------|-------------|
| `client.create_collection()` | Create vector collection |
| `client.insert()` | Add vectors |
| `client.upsert()` | Insert or replace |
| `client.search()` | ANN vector search |
| `client.query()` | Exact filter query (no vector) |
| `client.get()` | Fetch by ID |
| `client.delete()` | Delete by ID or filter |
| `client.get_collection_stats()` | Row count, info |

---

## 6️⃣ pgvector (PostgreSQL)

**Type:** PostgreSQL extension (open source)  
**Free tier:** Fully free; cloud options (Supabase, Neon) have free tiers  
**Docker image:** `pgvector/pgvector:pg16`  
**Docs:** https://github.com/pgvector/pgvector

### Install
```bash
pip install psycopg2-binary pgvector openai
```

### Environment Variables
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/mydb"
export OPENAI_API_KEY="sk-xxxx..."
```

### Run with Docker
```bash
docker run -d \
    --name pgvector \
    -e POSTGRES_USER=raguser \
    -e POSTGRES_PASSWORD=ragpass \
    -e POSTGRES_DB=ragdb \
    -p 5432:5432 \
    pgvector/pgvector:pg16
```

### Connect
```python
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgresql://raguser:ragpass@localhost:5432/ragdb")
register_vector(conn)     # ← REQUIRED: registers pgvector type with psycopg2
cursor = conn.cursor()
```

### Setup & Pass Embeddings
```sql
-- Run once in psql as superuser:
CREATE EXTENSION IF NOT EXISTS vector;
```

```python
# Create table with vector column
cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id        SERIAL PRIMARY KEY,
        text      TEXT,
        embedding vector(1536)    -- dimension must match model!
    );
""")

# Create HNSW index
cursor.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops);
""")

# Insert with embedding
embedding = openai_client.embeddings.create(...).data[0].embedding
cursor.execute(
    "INSERT INTO documents (text, embedding) VALUES (%s, %s)",
    ("Hello world", embedding)     # psycopg2 + register_vector handles the conversion
)
conn.commit()

# Query (cosine distance: <=> — lower is more similar)
cursor.execute("""
    SELECT text, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 3
""", (query_embedding, query_embedding))

rows = cursor.fetchall()   # [(text, similarity), ...]
```

### Distance Operators
| Operator | Distance Type | Use When |
|----------|--------------|----------|
| `<=>` | Cosine | Text similarity (most common) |
| `<->` | L2 Euclidean | Image embeddings, spatial data |
| `<#>` | Negative inner product | When vectors are normalized |

### Key Advantage
pgvector lets you combine vector search with full SQL:
```sql
-- Vector search + JOIN + WHERE filter in ONE query
SELECT d.text, a.author_name, 1 - (d.embedding <=> $1) AS sim
FROM documents d
JOIN authors a ON d.author_id = a.id
WHERE d.published = true
  AND d.created_at > '2024-01-01'
ORDER BY d.embedding <=> $1
LIMIT 5;
```

---

## ⚖️ Comparison Table

| | Pinecone | Weaviate | ChromaDB | Qdrant | Milvus | pgvector |
|--|---------|----------|----------|--------|--------|---------|
| **Hosting** | Cloud only | Self + Cloud | Local + Server | Self + Cloud | Self + Cloud | Self + Cloud |
| **Free tier** | 100k vectors | 14-day trial | Unlimited local | 1GB cloud | Local unlimited | Unlimited local |
| **Setup difficulty** | ⭐ Easiest | ⭐⭐ Easy | ⭐ Easiest | ⭐⭐ Easy | ⭐⭐⭐ Medium | ⭐⭐ Easy |
| **Scale** | Millions | Millions | Thousands–Millions | Millions | Billions | Millions |
| **Hybrid search** | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ (SQL) |
| **Built-in vectorizer** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **SQL support** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Full SQL |
| **Language** | Python, JS, Go | Python, JS, Go | Python, JS | Python, Rust, Go | Python, Go, Java | Any Postgres client |

---

## 💡 Which Database Should I Choose?

| Situation | Best Choice |
|-----------|-------------|
| Just starting out, local dev | **ChromaDB** — zero setup, free |
| Need production managed service | **Pinecone** — easiest to scale |
| Already have PostgreSQL | **pgvector** — add vector search to existing DB |
| Need hybrid vector + keyword | **Weaviate** or **Qdrant** |
| Billions of vectors, enterprise | **Milvus** |
| Open source + best performance | **Qdrant** |
| Built-in auto-embedding | **Weaviate** |

---

## 🔄 Standard Embedding Models

| Model | Dimension | Notes |
|-------|-----------|-------|
| `text-embedding-3-small` | 1536 | OpenAI, fast, cheap |
| `text-embedding-3-large` | 3072 | OpenAI, highest quality |
| `text-embedding-ada-002` | 1536 | OpenAI, legacy |
| `all-MiniLM-L6-v2` | 384 | HuggingFace, free, local |
| `BAAI/bge-large-en-v1.5` | 1024 | HuggingFace, very good quality |

> ⚠️ **Critical:** The dimension in your vector DB index MUST match the model you use.

---

## 📦 All Dependencies

```bash
# Core
pip install openai

# Pinecone
pip install pinecone-client

# Weaviate
pip install weaviate-client

# ChromaDB
pip install chromadb

# Qdrant
pip install qdrant-client

# Milvus
pip install pymilvus

# pgvector
pip install psycopg2-binary pgvector
```
