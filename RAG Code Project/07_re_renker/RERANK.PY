"""
Docstring for re_renker.RERANK



❌ Before Retrieval?

NO — re-ranking needs retrieved candidates

✅ After Retrieval?

YES — always after vector DB retrieval

docs = vector_db.similarity_search(query, k=20)
reranked_docs = reranker.rerank(query, docs, top_n=5)
answer = llm.generate(query, reranked_docs)

B. Cross-Encoder (Most Accurate ⭐)
Model	                                  Use
cross-encoder/ms-marco-MiniLM-L-6-v2	  Search QA
bge-reranker-large	                     RAG production
Cohere Rerank	                         API-based
Jina Reranker	                         Large context




E. Hybrid Re-Ranking (Production Favorite)
BM25 score + Vector similarity + Cross-Encoder score

Embedding Model  → BGE / OpenAI
Vector DB        → FAISS / Chroma / Pinecone
Re-Ranker        → Cross-Encoder / BGE Reranker
LLM              → GPT / LLaMA / Mixtral

pip install sentence-transformers
pip install faiss-cpu
pip install rank-bm25

"""


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Vegan diet improves heart health",
    "Python is used in AI",
    "RAG improves LLM accuracy",
    "Vector databases store embeddings"
]

doc_embeddings = embed_model.encode(documents)
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# 7️⃣ Retrieval Function (Before Re-Ranking)
def retrieve(query, k=10):
    query_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(query_emb), k)
    return [documents[i] for i in indices[0]]


# 8️⃣ Cross-Encoder Re-Ranking (BEST METHOD)
# Load Re-Ranker
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Re-Ranking Function


def rerank(query, docs, top_n=5):
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_n]


# End-to-End RAG Flow (Production-Style)


query = "How does RAG work?"

retrieved_docs = retrieve(query, k=15)

reranked_docs = rerank(query, retrieved_docs, top_n=5)

for doc, score in reranked_docs:
    print(score, doc)


# Hybrid Re-Ranking (BM25 + Cross-Encoder)


from rank_bm25 import BM25Okapi

tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

def hybrid_rerank(query, docs):
    bm25_scores = bm25.get_scores(query.split())
    ce_scores = reranker.predict([(query, d) for d in docs])

    final_scores = [
        0.4 * bm25_scores[i] + 0.6 * ce_scores[i]
        for i in range(len(docs))
    ]

    return sorted(zip(docs, final_scores), key=lambda x: x[1], reverse=True)

# How to Check Re-Ranking Result (Debugging)

for doc, score in reranked_docs:
    print(f"Score: {score:.4f} | Doc: {doc}")

    print("Before Re-Ranking:")
for d in retrieved_docs[:5]:
    print(d)

print("\nAfter Re-Ranking:")
for d, s in reranked_docs:
    print(d)

# Connect Re-Ranked Docs to LLM (Generation)

context = "\n".join([doc for doc, _ in reranked_docs])

prompt = f"""
Answer the question using the context below:

Context:
{context}

Question:
{query}
"""

response = llm.generate(prompt)
print(response)

