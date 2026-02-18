"""
============================================================
OPEN-SOURCE EMBEDDING MODELS (TOP 10)
============================================================

This file contains:
- ONLY open-source embedding models
- NO paid APIs
- Each embedding model has:
    - clear explanation
    - separate function
    - reusable & production-ready code

Use cases:
- RAG (Retrieval Augmented Generation)
- Semantic Search
- Similarity Search
- Vector Databases (FAISS, Chroma, Qdrant)
- NLP pipelines

COMMON INSTALLATION (run once):
--------------------------------
pip install -U sentence-transformers transformers torch gensim fasttext

============================================================
"""

# ============================================================
# COMMON IMPORTS
# ============================================================

from sentence_transformers import SentenceTransformer
from typing import List


# ============================================================
# 1. all-MiniLM-L6-v2 (FASTEST & MOST POPULAR)
# ============================================================

def embed_minilm(texts: List[str]):
    """
    MODEL:
        sentence-transformers/all-MiniLM-L6-v2

    DIMENSION:
        384

    WHY IMPORTANT:
        - Very fast
        - Low memory usage
        - Most widely used for RAG & semantic search

    HOW IT WORKS:
        - Transformer-based
        - Compresses sentence meaning into 384 numbers
        - Optimized for speed

    BEST USE CASE:
        - Chatbots
        - Real-time semantic search
        - Small to medium datasets
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings


# ============================================================
# 2. all-mpnet-base-v2 (HIGH QUALITY EMBEDDINGS)
# ============================================================

def embed_mpnet(texts: List[str]):
    """
    MODEL:
        sentence-transformers/all-mpnet-base-v2

    DIMENSION:
        768

    WHY IMPORTANT:
        - Better semantic understanding than MiniLM
        - Higher accuracy

    HOW IT WORKS:
        - MPNet transformer
        - Learns masked + permuted tokens

    BEST USE CASE:
        - High-quality RAG
        - Document similarity
    """
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return model.encode(texts, normalize_embeddings=True)


# ============================================================
# 3. BGE (BEST OPEN-SOURCE RAG MODEL)
# ============================================================

def embed_bge(texts: List[str]):
    """
    MODEL:
        BAAI/bge-base-en-v1.5

    DIMENSION:
        768

    WHY IMPORTANT:
        - Designed specifically for retrieval
        - Top-performing open-source RAG embedding

    HOW IT WORKS:
        - Dual-encoder architecture
        - Uses instruction-style embeddings

    IMPORTANT NOTE:
        Prefix text with retrieval instruction
    """
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    texts = ["Represent this sentence for retrieval: " + t for t in texts]
    return model.encode(texts, normalize_embeddings=True)


# ============================================================
# 4. E5 (QUERY / DOCUMENT AWARE)
# ============================================================

def embed_e5(texts: List[str], mode: str = "doc"):
    """
    MODEL:
        intfloat/e5-base-v2

    DIMENSION:
        768

    WHY IMPORTANT:
        - Explicit separation between query and document
        - Very strong search accuracy

    HOW IT WORKS:
        - Prefix-based embeddings
        - Same model, different intent

    MODE:
        - mode="query"
        - mode="doc"
    """
    model = SentenceTransformer("intfloat/e5-base-v2")

    if mode == "query":
        texts = ["query: " + t for t in texts]
    else:
        texts = ["passage: " + t for t in texts]

    return model.encode(texts, normalize_embeddings=True)


# ============================================================
# 5. NOMIC EMBEDDINGS (LONG DOCUMENTS)
# ============================================================

def embed_nomic(texts: List[str]):
    """
    MODEL:
        nomic-ai/nomic-embed-text-v1

    DIMENSION:
        768

    WHY IMPORTANT:
        - Handles long text well
        - Excellent for PDFs and reports

    HOW IT WORKS:
        - GPT-style embedding transformer
        - Trained on document-level data
    """
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1")
    return model.encode(texts, normalize_embeddings=True)


# ============================================================
# 6. SENTENCE-T5 (PARAPHRASE & SIMILARITY)
# ============================================================

def embed_sentence_t5(texts: List[str]):
    """
    MODEL:
        sentence-transformers/sentence-t5-base

    DIMENSION:
        768

    WHY IMPORTANT:
        - Best for paraphrase detection
        - Strong semantic equivalence

    HOW IT WORKS:
        - Encoder-decoder (T5)
        - Focused on sentence meaning
    """
    model = SentenceTransformer("sentence-transformers/sentence-t5-base")
    return model.encode(texts, normalize_embeddings=True)


# ============================================================
# 7. LaBSE (MULTI-LANGUAGE)
# ============================================================

def embed_labse(texts: List[str]):
    """
    MODEL:
        sentence-transformers/LaBSE

    DIMENSION:
        768

    WHY IMPORTANT:
        - Supports 100+ languages
        - Same meaning → same vector across languages

    HOW IT WORKS:
        - Language-agnostic encoder
    """
    model = SentenceTransformer("sentence-transformers/LaBSE")
    return model.encode(texts, normalize_embeddings=True)


# ============================================================
# 8. DISTILUSE MULTILINGUAL (LIGHTWEIGHT)
# ============================================================

def embed_distiluse(texts: List[str]):
    """
    MODEL:
        distiluse-base-multilingual-cased-v2

    DIMENSION:
        512

    WHY IMPORTANT:
        - Faster than LaBSE
        - Multilingual + lightweight

    HOW IT WORKS:
        - Distilled Universal Sentence Encoder
    """
    model = SentenceTransformer(
        "sentence-transformers/distiluse-base-multilingual-cased-v2"
    )
    return model.encode(texts, normalize_embeddings=True)


# ============================================================
# 9. FASTTEXT (WORD-LEVEL, VERY FAST)
# ============================================================

def embed_fasttext(texts: List[str]):
    """
    MODEL:
        fastText (cc.en.300.bin)

    DIMENSION:
        300

    WHY IMPORTANT:
        - Extremely fast
        - Handles misspellings & rare words

    HOW IT WORKS:
        - Character n-grams
        - Static embeddings (word-level)

    NOTE:
        Download model manually from fastText website
    """
    import fasttext
    model = fasttext.load_model("cc.en.300.bin")
    return [model.get_sentence_vector(t) for t in texts]


# ============================================================
# 10. WORD2VEC (CLASSIC BASELINE)
# ============================================================

def train_word2vec(sentences: List[List[str]]):
    """
    MODEL:
        Word2Vec (Gensim)

    DIMENSION:
        100–300 (configurable)

    WHY IMPORTANT:
        - Lightweight
        - Educational baseline

    HOW IT WORKS:
        - Predicts surrounding words
        - Static embeddings
    """
    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=5,
        min_count=1,
        workers=4
    )
    return model


# ============================================================
# END OF FILE
# ============================================================


# other from CLoud ai 


"""
⚡ QUICK REFERENCE CHEAT SHEET
Copy-paste these code snippets directly into your projects
"""

# ============================================================================
# 📦 ONE-TIME INSTALLATION
# ============================================================================

"""
pip install -U sentence-transformers transformers torch numpy
pip install fasttext gensim  # Optional: for FastText & Word2Vec
"""

# ============================================================================
# 🚀 QUICK START - COPY & PASTE THESE
# ============================================================================

# ----------------------------------------------------------------------------
# Example 1: Simple Semantic Search (MiniLM - Fastest)
# ----------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer
import numpy as np

def simple_semantic_search():
    # Initialize
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Your documents
    docs = [
        "Python is a programming language",
        "Machine learning uses neural networks",
        "Data science involves statistics"
    ]
    
    # Embed documents
    doc_embeddings = model.encode(docs, normalize_embeddings=True)
    
    # Search query
    query = "What is ML?"
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    
    # Find best match
    similarities = np.dot(doc_embeddings, query_embedding)
    best_idx = np.argmax(similarities)
    
    print(f"Best match: {docs[best_idx]} (score: {similarities[best_idx]:.4f})")


# ----------------------------------------------------------------------------
# Example 2: RAG with Chunking (BGE - Best for RAG)
# ----------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer
import numpy as np

def rag_with_chunking():
    # Initialize
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    # Long document
    long_doc = """Your very long document here...""" * 10
    
    # Simple chunking (split by tokens)
    def chunk_text(text, max_chars=1600, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
    
    # Create chunks
    chunks = chunk_text(long_doc)
    
    # Add retrieval prefix
    chunks_with_prefix = [f"Represent this sentence for retrieval: {c}" for c in chunks]
    
    # Embed chunks
    chunk_embeddings = model.encode(chunks_with_prefix, normalize_embeddings=True)
    
    # Query
    query = "Your question here?"
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    
    # Find top 3 relevant chunks
    similarities = np.dot(chunk_embeddings, query_embedding)
    top_3_idx = np.argsort(similarities)[::-1][:3]
    
    # Get context for LLM
    context = "\n\n".join([chunks[i] for i in top_3_idx])
    
    print(f"Context ready ({len(context)} chars)")
    return context


# ----------------------------------------------------------------------------
# Example 3: Query vs Document (E5 - Explicit Separation)
# ----------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer
import numpy as np

def query_document_search():
    # Initialize
    model = SentenceTransformer("intfloat/e5-base-v2")
    
    # Documents (with 'passage:' prefix)
    docs = [
        "passage: Neural networks process information in layers",
        "passage: Backpropagation trains neural networks",
        "passage: CNNs are best for image data"
    ]
    doc_embeddings = model.encode(docs, normalize_embeddings=True)
    
    # Query (with 'query:' prefix)
    query = "query: How to train neural nets?"
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    
    # Search
    similarities = np.dot(doc_embeddings, query_embedding)
    best_idx = np.argmax(similarities)
    
    # Remove prefix for display
    result = docs[best_idx].replace("passage: ", "")
    print(f"Best match: {result}")


# ----------------------------------------------------------------------------
# Example 4: Long Documents (Nomic - 8K context)
# ----------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer

def embed_long_document():
    # Initialize (supports very long texts)
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    
    # Long document (PDF, research paper, etc.)
    long_text = """Your PDF content here...""" * 100
    
    # Chunk (Nomic can handle 8K tokens, so use bigger chunks)
    def chunk_large(text, max_chars=8000, overlap=800):
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
    
    chunks = chunk_large(long_text)
    
    # Embed
    embeddings = model.encode(chunks, normalize_embeddings=True)
    
    print(f"Embedded {len(chunks)} large chunks")
    return embeddings


# ----------------------------------------------------------------------------
# Example 5: Multilingual Search (LaBSE)
# ----------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer
import numpy as np

def multilingual_search():
    # Initialize
    model = SentenceTransformer("sentence-transformers/LaBSE")
    
    # Mixed language documents
    docs = [
        "Python is a programming language",  # English
        "Python est un langage de programmation",  # French
        "Python ist eine Programmiersprache",  # German
    ]
    
    # Embed (works for all languages)
    doc_embeddings = model.encode(docs, normalize_embeddings=True)
    
    # English query
    query = "What is Python?"
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    
    # Find best match (works across languages!)
    similarities = np.dot(doc_embeddings, query_embedding)
    
    for doc, sim in zip(docs, similarities):
        print(f"[{sim:.4f}] {doc}")


# ----------------------------------------------------------------------------
# Example 6: Duplicate Detection (T5 - Paraphrase)
# ----------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer
import numpy as np

def find_duplicates():
    # Initialize
    model = SentenceTransformer("sentence-transformers/sentence-t5-base")
    
    # Texts to check
    texts = [
        "The cat sat on the mat",
        "A feline rested on the rug",  # Paraphrase
        "Python is a language",
        "The cat was sitting on the mat"  # Similar
    ]
    
    # Embed
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    # Calculate pairwise similarity
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    # Find duplicates (similarity > 0.7)
    threshold = 0.7
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if similarity_matrix[i][j] > threshold:
                print(f"\n🔄 Duplicate found (score: {similarity_matrix[i][j]:.4f}):")
                print(f"   • {texts[i]}")
                print(f"   • {texts[j]}")


# ============================================================================
# 🎯 CHUNKING STRATEGIES - COPY THESE FUNCTIONS
# ============================================================================

# Strategy 1: Character-based chunking
def chunk_by_chars(text, chunk_size=1000, overlap=100):
    """Simple character-based chunking"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return [c for c in chunks if c.strip()]


# Strategy 2: Sentence-based chunking
def chunk_by_sentences(text, sentences_per_chunk=5):
    """Split by sentences (simple version)"""
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i+sentences_per_chunk])
        chunks.append(chunk)
    return chunks


# Strategy 3: Paragraph-based chunking
def chunk_by_paragraphs(text, paragraphs_per_chunk=2):
    """Split by paragraphs"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    for i in range(0, len(paragraphs), paragraphs_per_chunk):
        chunk = '\n\n'.join(paragraphs[i:i+paragraphs_per_chunk])
        chunks.append(chunk)
    return chunks


# Strategy 4: Token-based chunking (approximate)
def chunk_by_tokens(text, max_tokens=512, overlap=50):
    """Chunk by approximate token count (1 token ≈ 4 chars)"""
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        
        # Try to break at sentence
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > max_chars * 0.5:
                chunk = chunk[:last_period + 1]
                end = start + len(chunk)
        
        chunks.append(chunk.strip())
        start = end - overlap_chars
    
    return [c for c in chunks if c]


# ============================================================================
# 💡 MODEL SELECTION GUIDE
# ============================================================================

"""
USE CASE                  → MODEL
─────────────────────────────────────────────────────────────
Fast semantic search      → MiniLM (all-MiniLM-L6-v2)
High accuracy search      → MPNet (all-mpnet-base-v2)
RAG / Retrieval          → BGE (BAAI/bge-base-en-v1.5)
Query vs Document        → E5 (intfloat/e5-base-v2)
Long documents (PDFs)    → Nomic (nomic-embed-text-v1)
Paraphrase detection     → T5 (sentence-t5-base)
Multilingual (100+ lang) → LaBSE
Fast multilingual        → DistilUSE
Low resource / Fast      → FastText
Baseline / Educational   → Word2Vec

DIMENSIONS:
───────────
MiniLM       → 384
MPNet        → 768
BGE          → 768
E5           → 768
Nomic        → 768
T5           → 768
LaBSE        → 768
DistilUSE    → 512
FastText     → 300
Word2Vec     → 100-300
"""


# ============================================================================
# 🔧 UTILITY FUNCTIONS
# ============================================================================

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def batch_cosine_similarity(embeddings1, embeddings2):
    """Calculate pairwise cosine similarities"""
    # Normalize
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    # Dot product
    return np.dot(embeddings1, embeddings2.T)


def find_top_k(query_embedding, doc_embeddings, k=5):
    """Find top K most similar documents"""
    similarities = np.dot(doc_embeddings, query_embedding)
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return top_k_indices, similarities[top_k_indices]


# ============================================================================
# 📊 COMPLETE EXAMPLE: END-TO-END RAG SYSTEM
# ============================================================================

def complete_rag_example():
    """
    Complete RAG system example - copy this for your project!
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    print("🤖 Complete RAG System Example\n")
    
    # Step 1: Initialize model
    print("1️⃣ Loading model...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    # Step 2: Your knowledge base
    documents = [
        """Machine learning is a subset of artificial intelligence that enables 
        computers to learn from data without being explicitly programmed.""",
        
        """Neural networks are computational models inspired by biological neural 
        networks. They consist of layers of interconnected nodes.""",
        
        """Deep learning uses neural networks with multiple layers to learn 
        hierarchical representations of data.""",
        
        """Supervised learning requires labeled data where the correct output 
        is provided for each input during training.""",
        
        """Unsupervised learning finds patterns in data without labeled outputs, 
        such as clustering similar items together."""
    ]
    
    # Step 3: Chunk if needed (these are already small)
    print("2️⃣ Processing documents...")
    chunks = documents  # Already chunked
    
    # Step 4: Embed with retrieval prefix
    chunks_prefixed = [f"Represent this sentence for retrieval: {c}" for c in chunks]
    doc_embeddings = model.encode(chunks_prefixed, normalize_embeddings=True)
    print(f"   ✓ Embedded {len(chunks)} chunks")
    
    # Step 5: User query
    query = "How do neural networks work?"
    print(f"\n3️⃣ Query: '{query}'")
    
    # Step 6: Embed query (no prefix for query)
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    
    # Step 7: Find relevant chunks
    similarities = np.dot(doc_embeddings, query_embedding)
    top_3_idx = np.argsort(similarities)[::-1][:3]
    
    print(f"\n4️⃣ Top 3 relevant chunks:")
    for i, idx in enumerate(top_3_idx, 1):
        print(f"\n   Rank {i} (score: {similarities[idx]:.4f}):")
        print(f"   {chunks[idx][:100]}...")
    
    # Step 8: Build context for LLM
    context = "\n\n".join([chunks[idx] for idx in top_3_idx])
    
    # Step 9: Create prompt for LLM
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""
    
    print(f"\n5️⃣ Final prompt ready ({len(prompt)} chars)")
    print(f"\n✅ Send this prompt to your LLM!")
    
    return prompt


# ============================================================================
# RUN EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("⚡ QUICK REFERENCE - READY TO USE CODE")
    print("="*80 + "\n")
    
    # Uncomment to run examples:
    # simple_semantic_search()
    # rag_with_chunking()
    # query_document_search()
    # multilingual_search()
    # find_duplicates()
    complete_rag_example()
    
    print("\n" + "="*80)
    print("✅ Copy any function above into your project!")
    print("="*80)