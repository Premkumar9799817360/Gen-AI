# pip install chromadb
# pip install sentence-transformers 



# ===============================
# 1. Imports
# ===============================

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ===============================
# 2. Embedding Model
# ===============================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def get_embedding(text: str):
    return embedding_model.encode(text).tolist()


# ===============================
# 3. ChromaDB Setup
# ===============================

chroma_client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",
        anonymized_telemetry=False
    )
)

collection = chroma_client.get_or_create_collection(
    name="rag_docs"
)
# ===============================
# 5. Chunking
# ===============================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = []
for doc in documents:
    chunks.extend(text_splitter.split_text(doc))


# ===============================
# 6. Indexing & Storing
# ===============================

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        metadatas=[{"source": "manual"}],
        ids=[f"doc_{i}"],
        embeddings=[get_embedding(chunk)]
    )

chroma_client.persist()



# ===============================
# 7. Retrieval
# ===============================

def retrieve(query, top_k=5):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]



# ===============================
# 8. Re-ranking
# ===============================

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs):
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked]



# ===============================
# 9. LLM Setup
# ===============================

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

def generate_answer(context, question):
    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ===============================
# 10. Full RAG Pipeline
# ===============================

def rag_pipeline(query):
    # Step 1: Retrieve
    retrieved_docs = retrieve(query, top_k=5)

    # Step 2: Re-rank
    ranked_docs = rerank(query, retrieved_docs)

    # Step 3: Select top context
    context = "\n".join(ranked_docs[:2])

    # Step 4: Generate
    answer = generate_answer(context, query)

    return answer


# ===============================
# 11. Run
# ===============================

query = "What is the role of re-ranking in RAG?"
response = rag_pipeline(query)

print(response)
