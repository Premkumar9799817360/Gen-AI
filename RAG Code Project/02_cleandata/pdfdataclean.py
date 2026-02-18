"""
Universal PDF cleaning utilities for LangChain ingestion.

Features:
- Robust text extraction with `pdfplumber` and PyMuPDF (`fitz`) fallback
- Optional OCR via `pytesseract` when extraction is poor
- Dehyphenation, header/footer removal, watermark/page-number removal
- Unicode normalization and whitespace cleanup
- Export as `langchain.schema.Document` objects with metadata

Usage: call `pdf_to_documents(path)` to get cleaned Documents ready
for chunking/embedding in a RAG pipeline.
"""

from __future__ import annotations

import io
import logging
import os
import re
import unicodedata
from collections import Counter
from typing import List, Dict, Optional

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:
    import pytesseract
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None
    Image = None

from langchain.schema import Document

LOG = logging.getLogger(__name__)


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text


def _dehyphenate(text: str) -> str:
    # Join words split across line endings like 'exam-\nple' => 'example'
    text = re.sub(r"-\n\s*", "", text)
    # Remove hyphenation artifacts at spaces too
    text = re.sub(r"(\w)-\s+\n\s*(\w)", r"\1\2", text)
    return text


def _remove_redundant_whitespace(text: str) -> str:
    # Normalize newlines, collapse repeating whitespace
    text = re.sub(r"\r\n?", "\n", text)
    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = text.strip()
    return text


def _remove_page_numbers_and_common_patterns(text: str) -> str:
    # Remove common page numbering lines like '1', 'Page 1 of 10', '(1)'
    text = re.sub(r"^\s*page\s+\d+\b.*$", "", text, flags=re.I | re.M)
    text = re.sub(r"^\s*\(?\d+\)?\s*$", "", text, flags=re.M)
    text = re.sub(r"\bpage\s+\d+\b", "", text, flags=re.I)
    return text


def _detect_repeating_lines(pages: List[str], min_freq: float = 0.6) -> set:
    # Identify short lines that repeat across many pages (headers/footers)
    lines_per_page = [set(l for l in p.splitlines() if len(l.strip()) and len(l.strip()) < 100) for p in pages]
    counter = Counter()
    for s in lines_per_page:
        counter.update(s)
    n_pages = max(1, len(pages))
    repeating = {line for line, cnt in counter.items() if cnt / n_pages >= min_freq}
    return repeating


def clean_extracted_text(pages: List[str]) -> List[str]:
    """Clean a list of page texts and return cleaned page texts.

    This function is extraction-agnostic: pass in raw text extracted per page.
    """
    # Basic normalizations per page
    pages = [(_normalize_unicode(p)) for p in pages]
    pages = [(_dehyphenate(p)) for p in pages]
    pages = [(_remove_redundant_whitespace(p)) for p in pages]

    # Remove headers/footers that repeat across pages
    repeating = _detect_repeating_lines(pages)
    cleaned_pages = []
    for p in pages:
        if repeating:
            lines = [ln for ln in p.splitlines() if ln.strip() not in repeating]
            p = "\n".join(lines)
        p = _remove_page_numbers_and_common_patterns(p)
        # Remove watermark-like phrases
        p = re.sub(r"\b(CONFIDENTIAL|DRAFT|INTERNAL USE ONLY|DO NOT DISTRIBUTE)\b", "", p, flags=re.I)
        p = _remove_redundant_whitespace(p)
        cleaned_pages.append(p)
    return cleaned_pages


def _ocr_page_via_fitz(page, dpi: int = 200) -> str:
    # Render page to image bytes via PyMuPDF and run pytesseract (if available)
    if fitz is None or pytesseract is None or Image is None:
        return ""
    try:
        mat = page.get_pixmap(dpi=dpi)
        img_bytes = mat.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        LOG.debug("OCR via fitz failed: %s", e)
        return ""


def extract_text_from_pdf(path: str, ocr: bool = True, ocr_min_chars: int = 80) -> List[str]:
    """Extract text per page from a PDF using pdfplumber with fitz fallback.

    If a page's extracted text is short and `ocr` is True, attempt OCR.
    Returns list of page strings.
    """
    pages_text: List[str] = []

    # Try pdfplumber first
    if pdfplumber is not None:
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        text = ""
                    pages_text.append(text)
        except Exception as e:
            LOG.debug("pdfplumber failed to open %s: %s", path, e)

    # If we got nothing and have fitz, use it to extract or to OCR
    if (not pages_text or all(not t.strip() for t in pages_text)) and fitz is not None:
        try:
            doc = fitz.open(path)
            pages_text = []
            for p in doc:
                try:
                    text = p.get_text("text") or ""
                except Exception:
                    text = ""
                pages_text.append(text)
        except Exception as e:
            LOG.debug("fitz extraction failed for %s: %s", path, e)

    # If still empty, create empty page list with single empty string
    if not pages_text:
        pages_text = [""]

    # Optionally run OCR per page when text is too small
    if ocr and pytesseract is not None and fitz is not None:
        try:
            doc = fitz.open(path)
            ocr_pages_text = []
            for idx, page in enumerate(doc):
                text = pages_text[idx] if idx < len(pages_text) else ""
                if len(text.strip()) < ocr_min_chars:
                    ocr_text = _ocr_page_via_fitz(page)
                    # prefer OCR if it's longer
                    text = ocr_text if len(ocr_text.strip()) > len(text.strip()) else text
                ocr_pages_text.append(text)
            pages_text = ocr_pages_text
        except Exception as e:
            LOG.debug("OCR fallback failed for %s: %s", path, e)

    return pages_text


def pdf_to_documents(path: str, ocr: bool = True, min_page_chars: int = 50) -> List[Document]:
    """Convert PDF at `path` to cleaned `langchain.schema.Document` objects.

    Each returned Document corresponds to one PDF page (use splitting later
    for smaller chunks). Metadata includes source path and page number.
    """
    pages = extract_text_from_pdf(path, ocr=ocr)
    cleaned = clean_extracted_text(pages)

    docs: List[Document] = []
    for i, text in enumerate(cleaned, start=1):
        if not text or len(text.strip()) < min_page_chars:
            continue
        metadata: Dict[str, str] = {
            "source": os.path.basename(path),
            "source_path": os.path.abspath(path),
            "page": str(i),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Clean PDF and output sample text")
    p.add_argument("pdf", help="Path to PDF file")
    p.add_argument("--ocr", action="store_true", help="Enable OCR fallback")
    args = p.parse_args()

    LOG.info("Extracting and cleaning %s (ocr=%s)", args.pdf, args.ocr)
    docs = pdf_to_documents(args.pdf, ocr=args.ocr)
    LOG.info("Produced %d cleaned page documents", len(docs))
    if docs:
        print("--- Example (first doc) ---")
        print(docs[0].page_content[:1000])