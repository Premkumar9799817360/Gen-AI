# pip install pinecone-client
# pip install sentence-transformers
# pip install langchain
# pip install groq
# pip install numpy
# export PINECONE_API_KEY="YOUR_PINECONE_KEY"
# export GROQ_API_KEY="YOUR_GROQ_KEY"

# ============================================================
# import os
# import numpy as np
# import pinecone

# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from groq import Groq


# ===============================
# Embedding Model
# ===============================

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def embed(text: str):
    return embedder.encode(text).tolist()


# ===============================
# Pinecone Setup
# ===============================

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment="gcp-starter"
)

INDEX_NAME = "rag-pinecone-index"

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine"
    )

index = pinecone.Index(INDEX_NAME)


# ===============================
# Chunking
# ===============================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = []
for doc in documents:
    chunks.extend(splitter.split_text(doc))


# ===============================
# Indexing
# ===============================

vectors = []

for i, chunk in enumerate(chunks):
    vectors.append(
        (
            f"doc-{i}",
            embed(chunk),
            {"text": chunk}
        )
    )

index.upsert(vectors)


# ===============================
# Retrieval
# ===============================

def retrieve(query, top_k=8):
    query_vec = embed(query)

    res = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True
    )

    return [match["metadata"]["text"] for match in res["matches"]]




# ===============================
# GROQ LLM
# ===============================

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_answer(context, question):
    prompt = f"""
You are an expert AI assistant.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content




# ===============================
# RAG Pipeline
# ===============================

def rag_pipeline(query):
    # Step 1: Retrieve
    retrieved = retrieve(query)

    # Step 2: MMR Re-rank
    reranked = mmr_rerank(query, retrieved)

    # Step 3: Context
    context = "\n".join(reranked)

    # Step 4: Generate
    return generate_answer(context, query)



if __name__ == "__main__":
    query = "Why is MMR useful in RAG systems?"
    answer = rag_pipeline(query)
    print(answer)
















































































