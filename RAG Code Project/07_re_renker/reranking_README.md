# 🔄 Re-Ranking Techniques — Complete RAG Reference

Everything you need to know about re-ranking: what it is, why it matters, all major techniques, and exactly how each one works in code.

---

## 📁 Files

| File | Contents |
|------|----------|
| `reranking_techniques.py` | All 11 re-ranking techniques with full implementations + universal `rerank()` function |
| `README.md` | This guide |

---

## 🤔 What Is Re-Ranking?

In a RAG pipeline, the first retrieval step uses fast **vector search** to grab the top-K documents. But "closest vector" doesn't always mean "most relevant to the question."

**Re-ranking** takes those initial results and applies a more powerful (but slower) model to re-score and re-order them before sending to the LLM.

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  Step 1: Initial Retrieval  │  ← Fast ANN vector search
│  Returns top-100 documents  │    (milliseconds)
└─────────────┬───────────────┘
              │ 100 candidate docs
              ▼
┌─────────────────────────────┐
│  Step 2: Re-Ranking         │  ← Powerful model, only on 100 docs
│  Returns top-5 documents    │    (50–500ms depending on method)
└─────────────┬───────────────┘
              │ 5 highly relevant docs
              ▼
┌─────────────────────────────┐
│  Step 3: LLM Generation     │  ← Context = top-5 docs
│  Final Answer               │
└─────────────────────────────┘
```

### Why Not Just Retrieve Top-5 Directly?

| Problem | Why It Happens | Re-ranking Fix |
|---------|---------------|----------------|
| False positives | Embedding similarities are approximate | Cross-encoder scores actual relevance |
| Keyword mismatch | Semantic search misses exact terms | BM25 or hybrid re-ranking |
| Redundant results | Top-5 may all say the same thing | MMR diversity re-ranking |
| LLM attention bias | LLM ignores middle context | Lost-in-Middle re-ordering |

---

## ⚡ Quick Start

```bash
pip install openai cohere sentence-transformers rank-bm25 flashrank

export OPENAI_API_KEY="sk-xxxx..."
export COHERE_API_KEY="xxxx..."    # only for Cohere method

python reranking_techniques.py
```

```python
from reranking_techniques import rerank

docs = [
    {"id": "1", "text": "Transformers use self-attention for long-range dependencies."},
    {"id": "2", "text": "Python is a programming language."},
    {"id": "3", "text": "BERT is a bidirectional transformer model."},
]

# Universal function — swap method name to change technique
results = rerank(
    query="How do transformers work?",
    docs=docs,
    method="cross_encoder",   # or: bm25, cohere, llm_pointwise, mmr, rrf, ...
    top_k=3
)

for r in results:
    print(f"[{r['score']:.4f}] {r['text']}")
```

---

## 📋 All Techniques at a Glance

| # | Method | Speed | Quality | Cost | Needs GPU? | Best For |
|---|--------|-------|---------|------|-----------|---------|
| 1 | **Cross-Encoder** | ⭐⭐ Slow | ⭐⭐⭐⭐⭐ Best | Free (local) | Optional | Highest accuracy |
| 2 | **Cohere Rerank** | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Best | Paid API | No | Production, no GPU |
| 3 | **LLM Pointwise** | ⭐ Slowest | ⭐⭐⭐⭐ Great | LLM API cost | No | GPT-4 quality scores |
| 4 | **LLM Pairwise** | ⭐ Slowest | ⭐⭐⭐⭐⭐ Best | High cost | No | Highest LLM quality |
| 5 | **LLM Listwise** | ⭐⭐ Slow | ⭐⭐⭐⭐ Great | Medium cost | No | Efficient LLM ranking |
| 6 | **BM25** | ⭐⭐⭐⭐⭐ Fastest | ⭐⭐ OK | Free | No | Keywords, hybrid |
| 7 | **RRF** | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐ Great | Free | No | Combining rankers |
| 8 | **MMR** | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ Great | Free | No | Diverse results |
| 9 | **MaxSim/ColBERT** | ⭐⭐ Slow | ⭐⭐⭐⭐⭐ Best | Free (local) | Optional | Token-level matching |
| 10 | **FlashRank** | ⭐⭐⭐⭐⭐ Fastest | ⭐⭐⭐⭐ Great | Free | No | CPU production |
| 11 | **Weighted Fusion** | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐ Great | Free | No | Tunable hybrid |

---

## 1️⃣ Cross-Encoder Re-Ranking

**The gold standard for quality.** A cross-encoder takes the query AND document together as a single input, so it can model fine-grained token-level interactions between them.

### How It Works

```
Bi-encoder (retrieval):         Cross-encoder (re-ranking):
  Query → [model] → q_vec          [query + doc] → [model] → score
  Doc   → [model] → d_vec
  score = cosine(q_vec, d_vec)   ← Joint encoding = more accurate
```

### Install
```bash
pip install sentence-transformers
```

### Code
```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

# pairs = [(query, doc_text), (query, doc_text), ...]
pairs  = [(query, doc["text"]) for doc in docs]
scores = model.predict(pairs)   # numpy array of floats

# Sort by score (higher = more relevant)
scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
```

### Best Models

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast | Very good | Best speed/quality balance |
| `cross-encoder/ms-marco-electra-base` | Medium | Excellent | Higher accuracy |
| `BAAI/bge-reranker-large` | Slow | State-of-art | Best open-source quality |
| `BAAI/bge-reranker-v2-m3` | Medium | Excellent | Multilingual support |

### When to Use
- When you can afford 100-500ms extra latency
- When retrieval quality is critical
- Production RAG with a GPU server

---

## 2️⃣ Cohere Rerank API

**Best managed re-ranking API.** Send query + documents, get back relevance scores. No model hosting required.

### Install
```bash
pip install cohere
export COHERE_API_KEY="xxxx..."
```

### Code
```python
import cohere
co = cohere.Client("YOUR_COHERE_API_KEY")

response = co.rerank(
    query="what is RAG?",
    documents=["doc text 1", "doc text 2", "doc text 3"],
    top_n=5,
    model="rerank-english-v3.0",   # or rerank-multilingual-v3.0
    return_documents=True
)

for result in response.results:
    print(f"[{result.relevance_score:.4f}] rank={result.index}: {result.document.text[:60]}")
```

### Models
| Model | Languages | Notes |
|-------|-----------|-------|
| `rerank-english-v3.0` | English | Best English quality |
| `rerank-multilingual-v3.0` | 100+ langs | Multi-language support |

### Pricing
~$1 per 1,000 search units (as of 2024). See https://cohere.com/pricing

---

## 3️⃣ LLM-Based Re-Ranking

Three variants: **Pointwise**, **Pairwise**, **Listwise**

### Pointwise (Independent Scoring)
Ask the LLM: *"Score this document 0-10 for relevance to this query"*

```python
PROMPT = """Rate relevance 0-10. Respond with ONLY the integer.

Query: {query}
Document: {document}
Score:"""

response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": PROMPT.format(query=q, document=doc)}],
    temperature=0, max_tokens=3
)
score = int(response.choices[0].message.content.strip())
```

**Cost:** 1 API call per document. 10 docs = 10 API calls.

### Pairwise (Compare Two at a Time)
Ask the LLM: *"Which document A or B is more relevant?"* — Tournament style.

```python
PROMPT = """Which is more relevant? Reply A or B only.
Query: {query}
A: {doc_a}
B: {doc_b}"""

# O(n²) API calls — expensive but most accurate
```

**Cost:** n*(n-1)/2 API calls. 10 docs = 45 API calls.

### Listwise (Rank All at Once)
Ask the LLM to output a ranked list in one shot.

```python
PROMPT = """Rank these documents by relevance. Return IDs comma-separated.
Query: {query}
Documents: {numbered_list}
Ranked (best first):"""

# 1 API call total — most efficient
```

**Cost:** 1 API call regardless of doc count. Use for 5-15 docs.

### When to Use LLM Re-ranking
- Access to GPT-4 and willing to pay for quality
- No GPU available for cross-encoder
- Need to explain why a document is relevant

---

## 4️⃣ BM25 Re-Ranking

**Classic keyword-based scoring.** Fast, free, no embeddings needed. Best as part of a hybrid system.

### Install
```bash
pip install rank-bm25
```

### Code
```python
from rank_bm25 import BM25Okapi

# Tokenize corpus
corpus   = [doc["text"].lower().split() for doc in docs]
bm25     = BM25Okapi(corpus)

# Score query against all docs
query_tokens = "how does attention work".split()
scores       = bm25.get_scores(query_tokens)   # numpy array

# Higher score = more keyword overlap with query
```

### BM25 Formula
```
BM25(q,d) = Σ IDF(qi) × freq(qi,d) × (k1+1)
                        ─────────────────────────────
                        freq(qi,d) + k1 × (1-b + b×|d|/avgdl)

k1 = 1.5  (term frequency saturation)
b  = 0.75 (document length normalization)
```

### When to Use
- Queries with specific technical terms / keywords
- Hybrid combination with vector search (via RRF or weighted fusion)
- When you need zero latency and no GPU

---

## 5️⃣ Reciprocal Rank Fusion (RRF)

**Best way to combine multiple rankers.** Uses rank positions instead of raw scores — so you don't need to normalize incompatible score scales.

### Formula
```
RRF(document) = Σ  1 / (k + rank(document, ranker_i))
               rankers

k = 60  (empirically optimal, prevents top-1 from dominating)
```

### Code
```python
def rrf(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1/(k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Combine BM25 ranks + vector search ranks
fused = rrf([bm25_results, vector_results])
```

### Example
```
Doc A: BM25 rank=1, Vector rank=3  →  RRF = 1/61 + 1/63 = 0.0321
Doc B: BM25 rank=5, Vector rank=1  →  RRF = 1/65 + 1/61 = 0.0318
Doc C: BM25 rank=2, Vector rank=2  →  RRF = 1/62 + 1/62 = 0.0323  ← WINS
```

### When to Use
- **Always use RRF** when combining 2+ retrieval systems
- Hybrid search (keyword + semantic)
- Multi-source retrieval (different vector DBs or document types)

---

## 6️⃣ Maximal Marginal Relevance (MMR)

**Best for diverse results.** Avoids returning 5 near-identical documents by penalizing redundancy.

### Formula
```
MMR(d) = argmax [ λ × sim(d, query) - (1-λ) × max sim(d, d') ]
         d ∈ R\S                                    d'∈ S

R = remaining candidates
S = already selected documents
λ = trade-off between relevance (1.0) and diversity (0.0)
```

### Code
```python
def mmr(query_vec, doc_vecs, docs, lambda_=0.7, top_k=5):
    relevance  = [cosine_sim(query_vec, dv) for dv in doc_vecs]
    selected   = []
    remaining  = list(range(len(docs)))
    
    while len(selected) < top_k:
        best = max(remaining, key=lambda i: 
            lambda_ * relevance[i] - (1-lambda_) * max(
                (cosine_sim(doc_vecs[i], doc_vecs[s]) for s in selected), default=0
            )
        )
        selected.append(best)
        remaining.remove(best)
    
    return [docs[i] for i in selected]
```

### Lambda Guide
| λ value | Behavior | Use Case |
|---------|----------|----------|
| `1.0` | Pure relevance, no diversity | Single specific question |
| `0.7` | 70% relevance, 30% diversity | **Default — good balance** |
| `0.5` | Equal balance | Broad topic exploration |
| `0.0` | Pure diversity | Document summarization |

---

## 7️⃣ ColBERT / MaxSim

**Token-level re-ranking.** Instead of a single embedding per document, ColBERT creates one embedding per token. Relevance = sum of maximum token-level similarities.

### MaxSim Formula
```
score(query, doc) = Σ   max  sim(q_token_i, d_token_j)
                  q_tokens  d_tokens

For each query token → find the best matching doc token → sum all maxes
```

### Why It's Better
```
Standard bi-encoder:
  "attention mechanism" → [0.2, 0.8, -0.1, ...]   (one vector for full phrase)

ColBERT:
  "attention" → [0.3, 0.7, ...]   ←  matches "attention" token in doc
  "mechanism" → [0.1, 0.9, ...]   ←  matches "mechanism" token in doc
  score = max_sim(attention) + max_sim(mechanism)
  (much more precise matching)
```

### Install
```bash
pip install sentence-transformers   # for lightweight MaxSim
pip install colbert-ai              # for full ColBERT
```

---

## 8️⃣ FlashRank (Ultra-Fast CPU)

**Best for production without GPU.** Uses ONNX-quantized cross-encoder models that run at 1-5ms per pair on CPU.

### Install
```bash
pip install flashrank
```

### Code
```python
from flashrank import Ranker, RerankRequest

ranker   = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
passages = [{"id": "1", "text": "doc text here"}, ...]
request  = RerankRequest(query="your query", passages=passages)
results  = ranker.rerank(request)   # sorted by score
```

### Available Models
| Model | Speed | Quality |
|-------|-------|---------|
| `ms-marco-TinyBERT-L-2-v2` | Fastest | Good |
| `ms-marco-MiniLM-L-12-v2` | Fast | Very good |
| `rank-T5-flan` | Medium | Excellent |

---

## 9️⃣ Lost-in-the-Middle Re-ordering

**LLM attention optimization.** Research proves LLMs pay most attention to the **beginning and end** of long contexts. This re-orders your already-ranked docs to put the best ones at the edges.

### The Problem
```
LLM context: [doc1, doc2, doc3, doc4, doc5]
LLM attention:  HIGH  LOW   LOW   LOW  HIGH
                (doc2, doc3, doc4 are "lost in the middle")
```

### The Fix
```python
# Original ranking by relevance: [most_rel, 2nd, 3rd, 4th, 5th]
# Re-ordered for LLM:            [most_rel, 3rd, 5th, 4th, 2nd]
#                                  ^ start         end ^
# Most relevant at START and END of context
```

### Code
```python
def lost_in_middle_reorder(ranked_docs):
    n = len(ranked_docs)
    reordered = [None] * n
    left, right = 0, n-1
    for i, doc in enumerate(ranked_docs):
        if i % 2 == 0:
            reordered[left]  = doc; left  += 1
        else:
            reordered[right] = doc; right -= 1
    return reordered

# Use AFTER any other re-ranking step, just before passing to LLM
top_docs   = cross_encoder_rerank(query, docs, top_k=10)
reordered  = lost_in_middle_reorder(top_docs)
context    = "\n\n".join([d["text"] for d in reordered])
```

### When to Use
- Always apply this as the **final step** before LLM
- Only matters when passing 5+ documents to the LLM

---

## 🔟 Weighted Score Fusion

**Tunable hybrid scoring.** Normalize scores from different systems to [0,1] then combine with configurable weights.

### Code
```python
# Normalize BM25 and vector scores to [0,1]
bm25_norm   = min_max_normalize(bm25_scores)    # [0,1]
vector_norm = min_max_normalize(vector_scores)  # [0,1]

# Weighted combination
combined = 0.3 * bm25_norm + 0.7 * vector_norm

# Tune weights based on your query type:
# - Keyword queries → increase BM25 weight
# - Semantic queries → increase vector weight
```

---

## 🔁 Universal `rerank()` Function

The file provides a single function that dispatches to any method:

```python
from reranking_techniques import rerank

# All methods use the same interface
results = rerank(query, docs, method="cross_encoder",   top_k=5)
results = rerank(query, docs, method="cohere",          top_k=5)
results = rerank(query, docs, method="bm25",            top_k=5)
results = rerank(query, docs, method="rrf",             top_k=5)
results = rerank(query, docs, method="mmr",             top_k=5)
results = rerank(query, docs, method="llm_pointwise",   top_k=5)
results = rerank(query, docs, method="llm_pairwise",    top_k=5)
results = rerank(query, docs, method="llm_listwise",    top_k=5)
results = rerank(query, docs, method="maxsim",          top_k=5)
results = rerank(query, docs, method="flashrank",       top_k=5)
results = rerank(query, docs, method="weighted_fusion", top_k=5)
results = rerank(query, docs, method="biencoder",       top_k=5)

# Each result dict has:
# result["score"]   → float relevance score
# result["method"]  → which method was used
# result["text"]    → original document text
# result["id"]      → original document id
```

---

## 🏗️ Full RAG Pipeline with Re-ranking

```python
from reranking_techniques import rag_with_reranking

answer = rag_with_reranking(
    question="How do transformers handle long-range dependencies?",
    initial_docs=your_vector_db_results,   # e.g. top-20 from Pinecone
    rerank_method="cross_encoder",         # apply cross-encoder re-ranking
    initial_k=20,                          # retrieve 20 initially
    final_k=5                              # pass top-5 to LLM
)
print(answer)
```

### Production Integration Pattern
```python
def production_rag(question: str) -> str:
    # Step 1: Fast retrieval (vector DB)
    candidates = vector_db.search(question, top_k=20)   # your vector DB

    # Step 2: Re-rank (choose based on your needs)
    reranked = rerank(question, candidates, method="cross_encoder", top_k=5)

    # Step 3: Lost-in-middle re-ordering (free quality boost)
    final_docs = lost_in_middle_reorder(reranked)

    # Step 4: Build context and generate
    context = "\n\n".join([d["text"] for d in final_docs])
    return llm_generate(question, context)
```

---

## 📊 Decision Guide

### Which Re-ranker Should I Use?

```
Do you need multilingual support?
    YES → Cohere rerank-multilingual-v3.0 or BAAI/bge-reranker-v2-m3
    NO  ↓

Do you have a GPU available?
    YES → Cross-encoder (BAAI/bge-reranker-large for best quality)
    NO  ↓

Do you have an API budget?
    YES → Cohere Rerank API (easiest, no infra)
    NO  ↓

Do you need results in <10ms?
    YES → BM25 only or FlashRank
    NO  ↓

Do you need diverse (non-redundant) results?
    YES → MMR (λ=0.7)
    NO  ↓

Are you combining multiple retrieval sources?
    YES → Reciprocal Rank Fusion (RRF)
    NO  ↓

Default best choice on CPU:
    → FlashRank (ms-marco-MiniLM-L-12-v2)
```

### Combining Techniques (Recommended Stack)

```python
# BEST PRODUCTION STACK:
# 1. Hybrid retrieval  → vector DB + BM25
# 2. Fuse with RRF    → combine rankings
# 3. Cross-encoder     → re-score top-20
# 4. Lost-in-Middle    → reorder before LLM

candidates  = vector_db.search(query, top_k=50)
bm25_hits   = bm25_search(query, top_k=50)
fused       = reciprocal_rank_fusion([candidates, bm25_hits], top_k=20)
reranked    = cross_encoder_rerank(query, fused, top_k=5)
final       = lost_in_middle_reorder(reranked)
answer      = llm_generate(question, final)
```

---

## 📦 All Dependencies

```bash
# Core
pip install openai

# Cross-encoder + Bi-encoder + MaxSim
pip install sentence-transformers

# Cohere Rerank API
pip install cohere

# BM25
pip install rank-bm25

# FlashRank (fast CPU re-ranking)
pip install flashrank

# PyTorch (needed by sentence-transformers)
pip install torch

# LangChain (optional, for pipeline integration)
pip install langchain-community
```

---

## 🧪 Benchmark: Speed vs Quality

Approximate numbers for re-ranking **20 documents** on CPU:

| Method | Latency | Relative Quality |
|--------|---------|-----------------|
| BM25 | ~1ms | Baseline |
| FlashRank | ~5ms | +15% over BM25 |
| Bi-encoder | ~50ms | +20% over BM25 |
| Cross-encoder (MiniLM) | ~200ms | +30% over BM25 |
| Cross-encoder (ELECTRA) | ~400ms | +35% over BM25 |
| Cohere API | ~300ms (network) | +35% over BM25 |
| LLM Listwise | ~500ms | +30% over BM25 |

> ⚡ **Key insight:** Re-ranking only runs on your top-K candidates (e.g., 20 docs), not the full corpus. Even "slow" cross-encoders add only 200-400ms — usually worth it.
