"""
Minimal universal JSON cleaning utilities for RAG pipelines.

Purpose:
- Provide the essential JSON cleaning steps required before chunking/embedding.
- Keep it intentionally small: normalize text, strip HTML/control chars,
  de-duplicate empty records, and turn JSON records into `langchain` Documents.

How it works (summary):
- `load_json(path)` reads JSON or NDJSON files into Python objects.
- `extract_text_from_record(record, fields=None)` gathers all string values
  (or only specified fields) recursively.
- `clean_text(text)` normalizes unicode, removes HTML tags/control chars,
  collapses whitespace, and strips.
- `json_to_documents(data, ...)` runs extraction+cleaning, filters short/empty
  results, and returns a list of `langchain.schema.Document` objects with
  minimal metadata (source, index).

Only must-have cleaning steps are included so output is ready for chunking.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

from langchain.schema import Document


def load_json(path: str) -> Any:
	"""Load JSON from a file.

	Supports regular JSON arrays/objects and newline-delimited JSON (NDJSON).
	"""
	path = os.path.abspath(path)
	with open(path, "r", encoding="utf-8") as f:
		text = f.read().strip()
		if not text:
			return []
		# Heuristic: NDJSON if many lines and each line is a JSON object
		if "\n" in text and text.lstrip().startswith("{"):
			items = []
			for line in text.splitlines():
				line = line.strip()
				if not line:
					continue
				try:
					items.append(json.loads(line))
				except Exception:
					# fall back to ignoring malformed lines
					continue
			return items
		try:
			return json.loads(text)
		except Exception:
			# If not parseable, return raw text in a list
			return [text]


def _gather_strings(obj: Any) -> Iterable[str]:
	"""Recursively yield all string values from a JSON-like structure."""
	if obj is None:
		return
	if isinstance(obj, str):
		yield obj
	elif isinstance(obj, dict):
		for v in obj.values():
			yield from _gather_strings(v)
	elif isinstance(obj, list) or isinstance(obj, tuple):
		for item in obj:
			yield from _gather_strings(item)


def extract_text_from_record(record: Dict[str, Any], fields: Optional[List[str]] = None) -> str:
	"""Extract and concatenate text from `record`.

	If `fields` is provided, only those keys (top-level) are considered.
	Otherwise all string values found recursively are concatenated.
	"""
	parts: List[str] = []
	if fields:
		for f in fields:
			if f in record and record[f] is not None:
				parts.extend(list(_gather_strings(record[f])))
	else:
		parts.extend(list(_gather_strings(record)))
	return "\n".join(p for p in parts if isinstance(p, str))


def clean_text(text: str) -> str:
	"""Perform minimal, safe cleaning of text.

	Steps:
	- Normalize Unicode (NFKC)
	- Remove HTML tags
	- Remove control characters
	- Collapse whitespace and strip
	"""
	if not text:
		return ""
	text = unicodedata.normalize("NFKC", text)
	# Remove simple HTML tags
	text = re.sub(r"<[^>]+>", " ", text)
	# Remove control chars (except newline/tab)
	text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]+", " ", text)
	# Collapse whitespace
	text = re.sub(r"[ \t\f\v]+", " ", text)
	text = re.sub(r" *\n+ *", "\n", text)
	text = text.strip()
	return text


def json_to_documents(
	data: Any,
	source: Optional[str] = None,
	fields: Optional[List[str]] = None,
	min_chars: int = 50,
) -> List[Document]:
	"""Convert loaded JSON data to cleaned `Document` objects.

	- `data` may be a list of records, a single dict, or raw text.
	- `fields` restricts which top-level keys to extract from each record.
	- Records yielding cleaned text shorter than `min_chars` are skipped.
	"""
	docs: List[Document] = []
	records: List[Any]
	if isinstance(data, list):
		records = data
	elif isinstance(data, dict):
		# treat single dict as one record
		records = [data]
	else:
		# raw text or unknown type
		cleaned = clean_text(str(data))
		if len(cleaned) >= min_chars:
			meta = {"source": source or "", "index": 0}
			docs.append(Document(page_content=cleaned, metadata=meta))
		return docs

	for i, rec in enumerate(records):
		if not isinstance(rec, dict):
			# convert primitives to string
			text = clean_text(str(rec))
		else:
			text = extract_text_from_record(rec, fields=fields)
			text = clean_text(text)
		if not text or len(text) < min_chars:
			continue
		meta: Dict[str, str] = {"source": source or "", "index": str(i)}
		# Optionally include some record-level metadata (id/key) if present
		if isinstance(rec, dict):
			if "id" in rec:
				meta["id"] = str(rec.get("id"))
			if "name" in rec:
				meta["name"] = str(rec.get("name"))
		docs.append(Document(page_content=text, metadata=meta))
	return docs


if __name__ == "__main__":
	import argparse
	from pathlib import Path

	p = argparse.ArgumentParser(description="Simple JSON cleaning for RAG ingestion")
	p.add_argument("input", help="Path to JSON or NDJSON file")
	p.add_argument("--fields", nargs="*", help="Top-level fields to extract (optional)")
	p.add_argument("--min-chars", type=int, default=50, help="Minimum chars to keep a record")
	args = p.parse_args()

	path = Path(args.input)
	if not path.exists():
		raise SystemExit(f"File not found: {path}")

	data = load_json(str(path))
	docs = json_to_documents(data, source=str(path.name), fields=args.fields, min_chars=args.min_chars)
	print(f"Produced {len(docs)} cleaned documents")
	if docs:
		print("--- Example content (first doc) ---")
		print(docs[0].page_content[:1000])

