"""
============================================================
  RE-RANKING TECHNIQUES — Complete RAG Reference
  All major re-ranking methods with full implementations
  
  Re-ranking = taking an initial set of retrieved documents
  and re-scoring/re-ordering them to improve relevance
  before passing to the LLM.

  WHY RE-RANK?
  Vector search finds "semantically similar" vectors — but
  similarity ≠ relevance. A document can be close in vector
  space but not actually answer the question. Re-rankers
  apply a more powerful (slower) model to the TOP-K results
  to find the truly best ones.

  PIPELINE:
  Query → Vector Search (fast, returns top-100)
        → Re-ranker (slow, but only on 100 docs)
        → Top-5 passed to LLM
        → Answer

INSTALL ALL:
    pip install openai cohere sentence-transformers \
                transformers torch rank-bm25 \
                flashrank langchain-community

ENV VARS:
    OPENAI_API_KEY  = "sk-xxxx..."
    COHERE_API_KEY  = "xxxx..."
============================================================
"""

import os
import math
import time
from typing import List, Dict, Any, Tuple, Optional, Callable
from openai import OpenAI

oai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ══════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Generate OpenAI embedding for a text string."""
    text = text.replace("\n", " ")
    resp = oai_client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def display_results(title: str, results: List[Dict], score_key: str = "score"):
    """Pretty-print re-ranking results."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        score = r.get(score_key, r.get("score", "N/A"))
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(f"  #{i} [{score_str}] {r['text'][:80]}...")
    print()


# ══════════════════════════════════════════════════════════════
# SAMPLE DATA — used across all re-rankers
# ══════════════════════════════════════════════════════════════

SAMPLE_QUERY = "How do transformer models handle long-range dependencies?"

SAMPLE_DOCS = [
    {"id": "d1",  "text": "Transformers use self-attention to capture relationships between all token pairs, enabling long-range dependency modeling without recurrence."},
    {"id": "d2",  "text": "LSTM networks use gated recurrent units to maintain state across sequences, partially addressing the vanishing gradient problem."},
    {"id": "d3",  "text": "The attention mechanism computes a weighted sum of values, where weights are based on query-key similarity scores."},
    {"id": "d4",  "text": "CNNs apply local filters across input data and struggle with very long-range dependencies due to limited receptive field."},
    {"id": "d5",  "text": "BERT uses bidirectional attention allowing each token to attend to all other tokens in both directions simultaneously."},
    {"id": "d6",  "text": "Python is a popular programming language used in data science and machine learning projects."},
    {"id": "d7",  "text": "Positional encodings are added to token embeddings in transformers to preserve sequence order information."},
    {"id": "d8",  "text": "The vanishing gradient problem affects deep networks when gradients become too small during backpropagation."},
    {"id": "d9",  "text": "GPT models use causal (unidirectional) self-attention where each token only attends to previous tokens."},
    {"id": "d10", "text": "Recurrent neural networks process sequences step by step and struggle to retain information from early timesteps."},
]


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 1: CROSS-ENCODER RE-RANKING
# ══════════════════════════════════════════════════════════════
"""
WHAT: A cross-encoder takes (query, document) together as input
      and outputs a single relevance score. Unlike bi-encoders
      (used for retrieval), it jointly encodes both texts so it
      can model fine-grained interactions between them.

WHY BETTER: Bi-encoder embeds query and doc independently → loses
            cross-token interactions. Cross-encoder sees both
            simultaneously → much higher quality scores.

BEST MODELS:
  - cross-encoder/ms-marco-MiniLM-L-6-v2   (fast, good)
  - cross-encoder/ms-marco-electra-base     (better accuracy)
  - BAAI/bge-reranker-large                 (state-of-art)
  - BAAI/bge-reranker-v2-m3                 (multilingual)

TRADEOFF: Slow (O(n) forward passes), but only run on top-K docs.
"""

def cross_encoder_rerank(
    query: str,
    docs: List[Dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5
) -> List[Dict]:
    """
    Re-rank documents using a cross-encoder model.
    Input: query + list of docs with 'text' field
    Output: top_k docs sorted by cross-encoder relevance score
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, max_length=512)

    # Build pairs: [(query, doc_text), (query, doc_text), ...]
    pairs = [(query, doc["text"]) for doc in docs]

    # Single batch forward pass → scores for all pairs
    scores = model.predict(pairs)  # numpy array of floats

    # Attach scores and sort descending
    scored = []
    for doc, score in zip(docs, scores):
        scored.append({**doc, "score": float(score), "method": "cross_encoder"})

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 2: COHERE RE-RANK API
# ══════════════════════════════════════════════════════════════
"""
WHAT: Cohere's hosted re-ranking API. Send query + documents,
      get back relevance scores. Uses Cohere's proprietary
      re-ranking model (rerank-english-v3.0).

WHY USE: No GPU/model needed — just an API call. Very high quality.
         rerank-english-v3.0 outperforms most open-source models.

PRICING: Pay-per-use, ~$1 per 1000 searches (as of 2024)
SIGNUP:  https://dashboard.cohere.com/api-keys
"""

def cohere_rerank(
    query: str,
    docs: List[Dict],
    top_k: int = 5,
    model: str = "rerank-english-v3.0"
) -> List[Dict]:
    """
    Re-rank using Cohere's Rerank API.
    Returns top_k docs with relevance_score (0-1).
    """
    import cohere

    co = cohere.Client(api_key=os.environ.get("COHERE_API_KEY", ""))

    # Cohere expects a list of strings
    texts = [doc["text"] for doc in docs]

    response = co.rerank(
        query=query,
        documents=texts,
        top_n=top_k,
        model=model,
        return_documents=True
    )

    results = []
    for result in response.results:
        original_doc = docs[result.index]
        results.append({
            **original_doc,
            "score":            result.relevance_score,  # float 0-1
            "original_rank":    result.index,            # position before re-ranking
            "method":           "cohere_rerank"
        })

    # Already sorted by Cohere, but sort again to be safe
    return sorted(results, key=lambda x: x["score"], reverse=True)


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 3: LLM-BASED RE-RANKING (Pointwise)
# ══════════════════════════════════════════════════════════════
"""
WHAT: Ask an LLM to score each (query, document) pair independently
      on a scale of 0-10. Then sort by score. Called "pointwise"
      because each document is scored independently.

WHY USE: Works with any LLM, no extra model needed. GPT-4 scores
         are very high quality but expensive for large doc sets.

TYPES:
  - Pointwise:  score each doc independently (this implementation)
  - Pairwise:   compare two docs at a time and pick better one
  - Listwise:   show all docs and ask LLM to rank the full list
"""

def llm_pointwise_rerank(
    query: str,
    docs: List[Dict],
    top_k: int = 5,
    llm_model: str = "gpt-4o-mini"
) -> List[Dict]:
    """
    Score each document independently with an LLM.
    Prompt asks: "On a scale 0-10, how relevant is this document?"
    """

    SCORE_PROMPT = """You are a relevance scoring assistant.
Rate how relevant the document is for answering the query.
Respond with ONLY a single integer from 0 to 10.
0 = completely irrelevant
10 = perfectly answers the query

Query: {query}
Document: {document}
Score (0-10):"""

    scored = []
    for doc in docs:
        prompt = SCORE_PROMPT.format(query=query, document=doc["text"])
        response = oai_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        raw = response.choices[0].message.content.strip()

        # Parse the integer score, fallback to 0
        try:
            score = int("".join(filter(str.isdigit, raw[:3])))
            score = max(0, min(10, score))  # clamp to 0-10
        except ValueError:
            score = 0

        scored.append({**doc, "score": score / 10.0, "method": "llm_pointwise"})

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def llm_pairwise_rerank(
    query: str,
    docs: List[Dict],
    top_k: int = 5,
    llm_model: str = "gpt-4o-mini"
) -> List[Dict]:
    """
    Compare documents in pairs — ask LLM which is more relevant.
    Uses a tournament-style approach (each doc vs every other).
    More accurate than pointwise but O(n²) API calls.
    """

    COMPARE_PROMPT = """Which document better answers the query? Reply with ONLY 'A' or 'B'.

Query: {query}

Document A: {doc_a}

Document B: {doc_b}

Better document (A or B):"""

    # Win counter for each doc
    wins = {doc["id"]: 0 for doc in docs}

    # Round-robin tournament
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            prompt = COMPARE_PROMPT.format(
                query=query,
                doc_a=docs[i]["text"],
                doc_b=docs[j]["text"]
            )
            resp = oai_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=3
            )
            winner = resp.choices[0].message.content.strip().upper()
            if "A" in winner:
                wins[docs[i]["id"]] += 1
            else:
                wins[docs[j]["id"]] += 1

    # Convert win count to score
    max_wins = max(wins.values()) if wins.values() else 1
    scored = [
        {**doc, "score": wins[doc["id"]] / max(max_wins, 1), "wins": wins[doc["id"]], "method": "llm_pairwise"}
        for doc in docs
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def llm_listwise_rerank(
    query: str,
    docs: List[Dict],
    top_k: int = 5,
    llm_model: str = "gpt-4o-mini"
) -> List[Dict]:
    """
    Show all documents to the LLM at once and ask it to produce
    a ranked list. Most efficient (one API call) but limited by
    context window and LLM's ability to handle many items.
    """

    doc_list = "\n".join([f"[{doc['id']}] {doc['text']}" for doc in docs])

    LISTWISE_PROMPT = f"""Rank the following documents by relevance to the query.
Return ONLY the document IDs in order from most to least relevant.
Format: ID1, ID2, ID3, ... (comma-separated, most relevant first)

Query: {query}

Documents:
{doc_list}

Ranked IDs (most relevant first):"""

    resp = oai_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": LISTWISE_PROMPT}],
        temperature=0,
        max_tokens=100
    )

    # Parse the ranked list
    raw = resp.choices[0].message.content.strip()
    ranked_ids = [x.strip() for x in raw.split(",")]

    # Build lookup
    doc_lookup = {doc["id"]: doc for doc in docs}

    results = []
    for rank, doc_id in enumerate(ranked_ids):
        if doc_id in doc_lookup:
            score = 1.0 - (rank / len(ranked_ids))  # 1.0 for rank 0, decreasing
            results.append({
                **doc_lookup[doc_id],
                "score":  round(score, 4),
                "rank":   rank + 1,
                "method": "llm_listwise"
            })

    return results[:top_k]


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 4: BM25 RE-RANKING (Sparse / Keyword)
# ══════════════════════════════════════════════════════════════
"""
WHAT: BM25 (Best Match 25) is a classic TF-IDF variant that scores
      documents based on term frequency and inverse document frequency.
      It's a keyword-based method — no embeddings needed.

WHY USE: Extremely fast, no GPU, great for keyword-heavy queries.
         Often used in HYBRID search: BM25 score + vector score.

FORMULA (simplified):
  BM25(q, d) = Σ IDF(qi) * (freq(qi,d) * (k1+1)) / (freq(qi,d) + k1*(1-b+b*|d|/avgdl))
  k1=1.5, b=0.75 are standard parameters

INSTALL:   pip install rank-bm25
"""

def bm25_rerank(
    query: str,
    docs: List[Dict],
    top_k: int = 5
) -> List[Dict]:
    """
    Re-rank documents using BM25 keyword matching.
    Best for: keyword-heavy queries, complementing vector search.
    """
    from rank_bm25 import BM25Okapi

    # Tokenize: lowercase + split (use a real tokenizer for production)
    tokenized_corpus = [doc["text"].lower().split() for doc in docs]
    tokenized_query  = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)   # numpy array

    scored = []
    for doc, score in zip(docs, scores):
        scored.append({**doc, "score": float(score), "method": "bm25"})

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 5: RECIPROCAL RANK FUSION (RRF)
# ══════════════════════════════════════════════════════════════
"""
WHAT: RRF combines rankings from MULTIPLE retrieval systems without
      needing score normalization. Each document gets a score based
      on its rank position across all systems.

FORMULA:
  RRF(d) = Σ_ranker 1 / (k + rank(d, ranker))
  k = 60 (empirically found optimal constant)

WHY: Scores from different systems (BM25 vs cosine) are on different
     scales and incompatible. RRF uses ranks instead — universally
     comparable. Outperforms individual systems in most benchmarks.

USE CASE: Combine vector search + BM25 → better than either alone.
          Classic "hybrid search" implementation.
"""

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    k: int = 60,
    top_k: int = 5
) -> List[Dict]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked doc lists (each sorted best-first)
                      Each doc must have an 'id' field.
        k:            Constant to prevent division by zero (default 60)
        top_k:        How many results to return

    Returns:
        Merged and re-ranked list with RRF scores
    """
    rrf_scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Dict]  = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_lookup[doc_id] = doc

    # Sort by RRF score descending
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in ranked[:top_k]:
        results.append({
            **doc_lookup[doc_id],
            "score":  round(score, 6),
            "method": "rrf"
        })
    return results


def hybrid_rrf_example(query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Classic hybrid search: combine BM25 + vector similarity via RRF.
    This is what most production RAG systems use for retrieval.
    """
    # Ranking 1: BM25 (keyword)
    bm25_ranked = bm25_rerank(query, docs, top_k=len(docs))

    # Ranking 2: Vector similarity (semantic)
    query_vec = get_embedding(query)
    doc_vecs  = [get_embedding(doc["text"]) for doc in docs]
    vector_scored = sorted(
        [{**doc, "score": cosine_similarity(query_vec, vec)} for doc, vec in zip(docs, doc_vecs)],
        key=lambda x: x["score"], reverse=True
    )

    # Fuse both rankings with RRF
    return reciprocal_rank_fusion([bm25_ranked, vector_scored], top_k=top_k)


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 6: COLBERT / MAXSIM RE-RANKING
# ══════════════════════════════════════════════════════════════
"""
WHAT: ColBERT (Contextualized Late Interaction over BERT) encodes
      query and document into token-level embeddings (not a single
      vector). Relevance = sum of MaxSim scores: for each query
      token, find the most similar document token.

WHY: Better than single-vector bi-encoders; cheaper than full
     cross-encoders. The "late interaction" finds the best
     token-level alignment between query and document.

MaxSim formula:
  score(q, d) = Σ_{qi ∈ q} max_{dj ∈ d} (qi · dj)

INSTALL: pip install colbert-ai  (or use lightweight version below)
         pip install sentence-transformers  (for token embeddings)
"""

def maxsim_rerank(
    query: str,
    docs: List[Dict],
    top_k: int = 5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[Dict]:
    """
    Lightweight MaxSim re-ranking using token-level embeddings.
    (Full ColBERT requires colbert-ai package; this approximates the idea.)
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    # Encode query and all documents
    query_emb = model.encode(query,           convert_to_tensor=True)
    doc_embs  = [model.encode(doc["text"],    convert_to_tensor=True) for doc in docs]

    import torch

    # query_emb shape: (dim,) → for token-level, we use word-by-word approach
    # Approximate MaxSim: each query word vs all doc words
    query_words = query.split()
    scored = []

    for doc, doc_emb in zip(docs, doc_embs):
        doc_words   = doc["text"].split()
        query_vecs  = model.encode(query_words, convert_to_tensor=True)   # (q_len, dim)
        doc_vecs    = model.encode(doc_words,   convert_to_tensor=True)   # (d_len, dim)

        # MaxSim: for each query token, max similarity to any doc token
        sim_matrix = torch.mm(query_vecs, doc_vecs.T)                     # (q_len, d_len)
        max_sims   = sim_matrix.max(dim=1).values                         # (q_len,)
        score      = max_sims.mean().item()                               # average MaxSim

        scored.append({**doc, "score": round(score, 4), "method": "maxsim_colbert"})

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 7: FLASHRANK (Lightweight, No GPU Required)
# ══════════════════════════════════════════════════════════════
"""
WHAT: FlashRank is a tiny, ultra-fast re-ranking library that uses
      small cross-encoder models optimized for speed. Runs on CPU.
      Perfect for production without GPU.

WHY: Full cross-encoders need 100-500ms per doc pair on CPU.
     FlashRank uses ONNX-quantized models: ~1-5ms per pair.

INSTALL: pip install flashrank
"""

def flashrank_rerank(
    query: str,
    docs: List[Dict],
    top_k: int = 5,
    model: str = "ms-marco-MiniLM-L-12-v2"
) -> List[Dict]:
    """
    Ultra-fast CPU re-ranking using FlashRank.
    """
    from flashrank import Ranker, RerankRequest

    ranker = Ranker(model_name=model, cache_dir="/tmp/flashrank")

    passages = [{"id": doc["id"], "text": doc["text"]} for doc in docs]
    request  = RerankRequest(query=query, passages=passages)
    results  = ranker.rerank(request)

    doc_lookup = {doc["id"]: doc for doc in docs}
    ranked = []
    for result in results[:top_k]:
        doc_id = result.get("id")
        ranked.append({
            **doc_lookup.get(doc_id, {"id": doc_id, "text": result.get("text", "")}),
            "score":  result.get("score", 0.0),
            "method": "flashrank"
        })
    return ranked


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 8: DIVERSITY RE-RANKING (MMR)
# ══════════════════════════════════════════════════════════════
"""
WHAT: Maximal Marginal Relevance (MMR) balances relevance AND
      diversity. It iteratively picks the next document that
      maximizes relevance to the query WHILE minimizing similarity
      to already-selected documents.

WHY: Pure relevance-based retrieval can return 5 near-identical docs.
     MMR ensures each doc adds NEW information.

FORMULA:
  MMR(d) = λ * sim(d, query) - (1-λ) * max_{d' ∈ selected} sim(d, d')
  λ = 0.7 → 70% relevance, 30% diversity (tune as needed)
"""

def mmr_rerank(
    query: str,
    docs: List[Dict],
    top_k: int = 5,
    lambda_param: float = 0.7
) -> List[Dict]:
    """
    Maximal Marginal Relevance re-ranking.
    
    Args:
        lambda_param: 1.0 = pure relevance, 0.0 = pure diversity
                      0.5-0.7 is typical for RAG
    """
    query_vec = get_embedding(query)
    doc_vecs  = [get_embedding(doc["text"]) for doc in docs]

    # Compute relevance scores (query-doc similarity)
    relevance = [cosine_similarity(query_vec, vec) for vec in doc_vecs]

    selected_indices = []
    remaining_indices = list(range(len(docs)))

    for _ in range(min(top_k, len(docs))):
        best_idx   = None
        best_score = float("-inf")

        for idx in remaining_indices:
            rel_score = relevance[idx]

            # Redundancy: max similarity to any already-selected doc
            if selected_indices:
                redundancy = max(
                    cosine_similarity(doc_vecs[idx], doc_vecs[sel])
                    for sel in selected_indices
                )
            else:
                redundancy = 0.0

            mmr_score = lambda_param * rel_score - (1 - lambda_param) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    results = []
    for rank, idx in enumerate(selected_indices):
        results.append({
            **docs[idx],
            "score":      round(relevance[idx], 4),
            "mmr_rank":   rank + 1,
            "method":     "mmr"
        })
    return results


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 9: SCORE NORMALIZATION + FUSION
# ══════════════════════════════════════════════════════════════
"""
WHAT: When combining scores from different rankers (which produce
      scores on different scales), you need to normalize first.
      Common normalizations:
        - Min-Max:   scale to [0, 1]
        - Z-score:   standardize to mean=0, std=1
        - Softmax:   probabilities that sum to 1

After normalization, you can directly sum or weight the scores
from multiple rankers.
"""

def min_max_normalize(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range."""
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

def z_score_normalize(scores: List[float]) -> List[float]:
    """Standardize scores to mean=0, std=1."""
    mean = sum(scores) / len(scores)
    var  = sum((s - mean) ** 2 for s in scores) / len(scores)
    std  = math.sqrt(var) if var > 0 else 1.0
    return [(s - mean) / std for s in scores]

def softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    """Convert scores to probabilities using softmax."""
    scaled = [s / temperature for s in scores]
    max_s  = max(scaled)  # for numerical stability
    exps   = [math.exp(s - max_s) for s in scaled]
    total  = sum(exps)
    return [e / total for e in exps]

def weighted_score_fusion(
    query: str,
    docs: List[Dict],
    top_k: int = 5,
    weights: Dict[str, float] = None
) -> List[Dict]:
    """
    Combine BM25 + vector similarity with weighted normalized scores.
    weights example: {"bm25": 0.3, "vector": 0.7}
    """
    if weights is None:
        weights = {"bm25": 0.3, "vector": 0.7}

    # Get BM25 scores
    bm25_results = bm25_rerank(query, docs, top_k=len(docs))
    bm25_map     = {r["id"]: r["score"] for r in bm25_results}

    # Get vector scores
    query_vec  = get_embedding(query)
    doc_vecs   = [get_embedding(doc["text"]) for doc in docs]
    vec_scores = [cosine_similarity(query_vec, dv) for dv in doc_vecs]

    # Normalize both to [0, 1]
    bm25_raw    = [bm25_map.get(doc["id"], 0.0) for doc in docs]
    bm25_norm   = min_max_normalize(bm25_raw)
    vector_norm = min_max_normalize(vec_scores)

    # Weighted combination
    scored = []
    for i, doc in enumerate(docs):
        combined = (weights["bm25"]   * bm25_norm[i] +
                    weights["vector"] * vector_norm[i])
        scored.append({
            **doc,
            "score":        round(combined, 4),
            "bm25_score":   round(bm25_norm[i], 4),
            "vector_score": round(vector_norm[i], 4),
            "method":       "weighted_fusion"
        })

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 10: LOST IN THE MIDDLE RE-ORDERING
# ══════════════════════════════════════════════════════════════
"""
WHAT: Research shows LLMs perform worst on information in the
      "middle" of a long context. They pay most attention to the
      beginning and end. "Lost in the Middle" re-ordering places
      the MOST RELEVANT documents at the start and end, and
      LESS RELEVANT ones in the middle.

PAPER: "Lost in the Middle: How Language Models Use Long Contexts"
       (Liu et al., 2023)

BEST FOR: When passing many (5-20) chunks to the LLM context.
"""

def lost_in_middle_reorder(ranked_docs: List[Dict]) -> List[Dict]:
    """
    Re-order already-ranked documents to place the best ones at
    the beginning and end of the context window.

    Input:  [rank1, rank2, rank3, rank4, rank5]  (sorted by relevance)
    Output: [rank1, rank3, rank5, rank4, rank2]  (best at edges)

    The LLM sees rank1 first, rank2 last — both get high attention.
    """
    n = len(ranked_docs)
    reordered = [None] * n

    # Place top docs at start and end alternately
    left  = 0
    right = n - 1

    for i, doc in enumerate(ranked_docs):
        if i % 2 == 0:
            reordered[left] = {**doc, "reorder_position": left, "method": "lost_in_middle"}
            left += 1
        else:
            reordered[right] = {**doc, "reorder_position": right, "method": "lost_in_middle"}
            right -= 1

    return reordered


# ══════════════════════════════════════════════════════════════
# TECHNIQUE 11: SEMANTIC SIMILARITY RE-RANKING (Bi-Encoder)
# ══════════════════════════════════════════════════════════════
"""
WHAT: Use a sentence transformer (bi-encoder) to re-rank.
      Faster than cross-encoders (encode query+docs independently)
      but more accurate than raw cosine similarity on chunk embeddings
      because the re-ranking model is specialized for similarity.

WHY: Your retrieval embeddings might use text-embedding-3-small
     optimized for search recall. A re-ranking bi-encoder like
     all-mpnet-base-v2 is fine-tuned for similarity scoring.

MODELS:
  - sentence-transformers/all-mpnet-base-v2         (best quality CPU)
  - sentence-transformers/all-MiniLM-L6-v2          (fast, good)
  - BAAI/bge-large-en-v1.5                          (state-of-art)
"""

def biencoder_rerank(
    query: str,
    docs: List[Dict],
    top_k: int = 5,
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
) -> List[Dict]:
    """
    Re-rank using a sentence transformer bi-encoder.
    Encodes query and docs separately, scores by cosine similarity.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    doc_texts = [doc["text"] for doc in docs]
    doc_embs  = model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)

    import numpy as np
    # Cosine similarity = dot product when normalized
    scores = doc_embs @ query_emb  # (n_docs,)

    scored = []
    for doc, score in zip(docs, scores):
        scored.append({**doc, "score": float(round(score, 4)), "method": "biencoder"})

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


# ══════════════════════════════════════════════════════════════
# UNIVERSAL RE-RANKING FUNCTION
# Automatically selects and applies the right re-ranker
# ══════════════════════════════════════════════════════════════

def rerank(
    query: str,
    docs: List[Dict],
    method: str = "cross_encoder",
    top_k: int = 5,
    **kwargs
) -> List[Dict]:
    """
    Universal re-ranking function. Dispatches to the chosen method.

    Args:
        query:   The search query string
        docs:    List of dicts, each must have 'id' and 'text' fields
        method:  One of:
                   "cross_encoder"      → sentence-transformers CrossEncoder
                   "cohere"             → Cohere Rerank API
                   "llm_pointwise"      → LLM scores each doc (0-10)
                   "llm_pairwise"       → LLM compares doc pairs
                   "llm_listwise"       → LLM ranks all at once
                   "bm25"               → BM25 keyword scoring
                   "rrf"                → Reciprocal Rank Fusion (hybrid)
                   "mmr"                → Max Marginal Relevance (diverse)
                   "maxsim"             → ColBERT-style MaxSim
                   "flashrank"          → Ultra-fast CPU re-ranking
                   "weighted_fusion"    → Normalized BM25 + vector scores
                   "biencoder"          → Bi-encoder sentence transformer
        top_k:   Number of results to return
        **kwargs: Method-specific parameters

    Returns:
        List of dicts sorted by score (best first), with added
        'score' and 'method' fields.
    """

    method_map = {
        "cross_encoder":   cross_encoder_rerank,
        "cohere":          cohere_rerank,
        "llm_pointwise":   llm_pointwise_rerank,
        "llm_pairwise":    llm_pairwise_rerank,
        "llm_listwise":    llm_listwise_rerank,
        "bm25":            bm25_rerank,
        "rrf":             lambda q, d, **kw: hybrid_rrf_example(q, d, top_k=top_k),
        "mmr":             mmr_rerank,
        "maxsim":          maxsim_rerank,
        "flashrank":       flashrank_rerank,
        "weighted_fusion": weighted_score_fusion,
        "biencoder":       biencoder_rerank,
    }

    if method not in method_map:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(method_map.keys())}")

    fn = method_map[method]
    return fn(query, docs, top_k=top_k, **kwargs)


# ══════════════════════════════════════════════════════════════
# FULL RAG PIPELINE WITH RE-RANKING
# ══════════════════════════════════════════════════════════════

def rag_with_reranking(
    question: str,
    initial_docs: List[Dict],
    rerank_method: str = "cross_encoder",
    initial_k: int = 20,
    final_k: int = 5
) -> str:
    """
    Complete RAG pipeline with re-ranking:

    1. Start with initial_docs (simulates vector DB retrieval of top-20)
    2. Re-rank using chosen method → top-5
    3. Build context from top-5
    4. Generate answer with OpenAI

    In production, step 1 would be: vector_db.search(query, top_k=initial_k)
    """

    print(f"\n[RAG] Question: {question}")
    print(f"[RAG] Initial docs: {len(initial_docs)} | Re-rank method: {rerank_method} | Final k: {final_k}")

    # Step 1: Re-rank
    t0 = time.time()
    reranked = rerank(question, initial_docs, method=rerank_method, top_k=final_k)
    t1 = time.time()
    print(f"[RAG] Re-ranking took {(t1-t0)*1000:.0f}ms")

    # Step 2: Build context
    context_parts = []
    for i, doc in enumerate(reranked, 1):
        context_parts.append(f"[Source {i} | score={doc.get('score', 'N/A')}]\n{doc['text']}")
    context = "\n\n".join(context_parts)

    # Step 3: Generate
    prompt = f"""You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

    response = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content
    print(f"[RAG] Answer generated.\n")
    return answer


# ══════════════════════════════════════════════════════════════
# BENCHMARK: Compare all re-rankers on same query
# ══════════════════════════════════════════════════════════════

def benchmark_rerankers(
    query: str,
    docs: List[Dict],
    methods: List[str] = None,
    top_k: int = 3
) -> Dict[str, List[Dict]]:
    """
    Run multiple re-rankers on the same query+docs and compare.
    Shows which docs each method selects as most relevant.
    Returns dict of method → ranked results.
    """
    if methods is None:
        methods = ["bm25", "biencoder", "mmr", "weighted_fusion", "rrf"]

    all_results = {}
    for method in methods:
        print(f"\n--- Running: {method} ---")
        try:
            t0 = time.time()
            results = rerank(query, docs, method=method, top_k=top_k)
            elapsed = (time.time() - t0) * 1000
            all_results[method] = results
            print(f"    Time: {elapsed:.0f}ms")
            for r in results:
                print(f"    [{r.get('score', '?'):.4f}] {r['text'][:60]}...")
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results[method] = []

    return all_results


# ══════════════════════════════════════════════════════════════
# DEMO RUN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  RE-RANKING TECHNIQUES DEMO")
    print("="*60)
    print(f"\nQuery: {SAMPLE_QUERY}")
    print(f"Docs:  {len(SAMPLE_DOCS)} documents\n")

    # ── Demo 1: BM25 (no API needed)
    print("\n[1] BM25 Keyword Re-ranking")
    for r in bm25_rerank(SAMPLE_QUERY, SAMPLE_DOCS, top_k=3):
        print(f"    [{r['score']:.4f}] {r['text'][:70]}")

    # ── Demo 2: MMR (diverse results)
    print("\n[2] MMR Diversity Re-ranking (lambda=0.7)")
    for r in mmr_rerank(SAMPLE_QUERY, SAMPLE_DOCS, top_k=3, lambda_param=0.7):
        print(f"    [{r['score']:.4f}] {r['text'][:70]}")

    # ── Demo 3: RRF hybrid fusion
    print("\n[3] Reciprocal Rank Fusion (BM25 + Vector)")
    for r in hybrid_rrf_example(SAMPLE_QUERY, SAMPLE_DOCS, top_k=3):
        print(f"    [{r['score']:.6f}] {r['text'][:70]}")

    # ── Demo 4: Weighted Score Fusion
    print("\n[4] Weighted Score Fusion (BM25 30% + Vector 70%)")
    for r in weighted_score_fusion(SAMPLE_QUERY, SAMPLE_DOCS, top_k=3):
        print(f"    [combined={r['score']:.4f} bm25={r['bm25_score']:.4f} vec={r['vector_score']:.4f}] {r['text'][:50]}")

    # ── Demo 5: Lost in Middle re-ordering
    print("\n[5] Lost-in-Middle Re-ordering")
    bm25_results = bm25_rerank(SAMPLE_QUERY, SAMPLE_DOCS, top_k=5)
    reordered    = lost_in_middle_reorder(bm25_results)
    for r in reordered:
        print(f"    [pos={r['reorder_position']}] {r['text'][:70]}")

    # ── Demo 6: Full RAG Pipeline
    print("\n[6] Full RAG Pipeline with re-ranking")
    answer = rag_with_reranking(
        question=SAMPLE_QUERY,
        initial_docs=SAMPLE_DOCS,
        rerank_method="bm25",   # swap to "cross_encoder" for best quality
        final_k=4
    )
    print(f"Answer: {answer}")

    # ── Demo 7: Universal function
    print("\n[7] Universal rerank() function")
    results = rerank(SAMPLE_QUERY, SAMPLE_DOCS, method="bm25", top_k=3)
    for r in results:
        print(f"    [{r['score']:.4f}] {r['text'][:70]}")
