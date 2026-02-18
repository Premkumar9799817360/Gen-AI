"""
Minimal CSV cleaning for RAG ingestion.

Purpose:
- Essential cleaning steps for structured CSV data before chunking.
- Simple because CSV is already structured (unlike nested JSON/PDFs).

How it works:
- `load_csv(path)` reads CSV into list of dicts.
- `clean_text(text)` normalizes unicode, removes control chars, collapses whitespace.
- `csv_to_documents(path, ...)` loads, cleans text columns, filters empty rows,
  returns `langchain.schema.Document` objects ready for chunking.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document


def load_csv(path: str) -> List[dict]:
    """Load CSV file into list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                rows.append(row)
    return rows


def clean_text(text: str) -> str:
    """Simple text cleaning: unicode normalize, remove control chars, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def csv_to_documents(
    path: str,
    text_columns: Optional[List[str]] = None,
    min_chars: int = 30,
) -> List[Document]:
    """Convert CSV rows to cleaned Document objects.

    - Loads CSV, cleans specified text columns (or all if None).
    - Skips rows where all cleaned text is empty.
    - Each row becomes one Document.
    """
    rows = load_csv(path)
    docs: List[Document] = []

    for i, row in enumerate(rows):
        if not row:
            continue
        
        # Determine which columns to use for content
        if text_columns:
            parts = [clean_text(str(row.get(col, ""))) for col in text_columns]
        else:
            # Use all columns
            parts = [clean_text(str(v)) for v in row.values()]
        
        # Combine into single text
        text = "\n".join(p for p in parts if p)
        
        # Skip if too short or empty
        if not text or len(text) < min_chars:
            continue
        
        # Create metadata from first few columns
        metadata = {
            "source": Path(path).name,
            "row": str(i),
        }
        # Add ID if present
        if "id" in row:
            metadata["id"] = str(row["id"])
        
        docs.append(Document(page_content=text, metadata=metadata))
    
    return docs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Simple CSV cleaning for RAG")
    p.add_argument("input", help="Path to CSV file")
    p.add_argument("--columns", nargs="*", help="Text columns to extract (optional, uses all if not specified)")
    p.add_argument("--min-chars", type=int, default=30, help="Minimum chars to keep a row")
    args = p.parse_args()

    docs = csv_to_documents(args.input, text_columns=args.columns, min_chars=args.min_chars)
    print(f"Produced {len(docs)} cleaned documents from CSV")
    if docs:
        print("--- Example content (first row) ---")
        print(docs[0].page_content[:500])
