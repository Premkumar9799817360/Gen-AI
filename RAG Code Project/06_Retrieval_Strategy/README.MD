# Retrieval Strategies After Vector Database Storage

This document explains common production-grade retrieval strategies used
after documents are stored in a Vector Database.

---

## 1. Top-K Similarity Search
Retrieves the top K most similar chunks using vector embeddings.
Fast and simple, but may miss exact keyword matches.

Use when:
- Small to medium datasets
- Pure semantic search

---

## 2. Hybrid Search (Vector + Keyword)
Combines semantic similarity with keyword matching.
Improves accuracy for technical and exact queries.

Use when:
- Queries contain exact terms (IDs, names)
- High precision is required

---

## 3. Metadata-Filtered Search
Retrieves documents based on metadata conditions before vector search.

Use when:
- Documents have categories, dates, or sources
- Need controlled retrieval

Example filters:
- file_type = "pdf"
- year >= 2024

---

## 4. Multi-Vector Retrieval
Uses multiple embeddings per document (summary + chunks).
Improves recall for long documents.

Use when:
- Large documents
- Multiple semantic meanings per document

---

## 5. Parent-Child Retrieval
Search is performed on small child chunks,
but the full parent document is returned.

Use when:
- Need full context
- PDFs, reports, manuals

---

## 6. Re-Ranking Models
Initial retrieval fetches many results,
then a re-ranking model selects the best ones.

Use when:
- High-stakes answers
- Production RAG systems

---

## Recommended Production Pipeline

Query
 → Hybrid Search
 → Metadata Filter
 → Top-K Retrieval
 → Re-Ranking
 → LLM Answer

---

This repository contains a simple Python example for each strategy.