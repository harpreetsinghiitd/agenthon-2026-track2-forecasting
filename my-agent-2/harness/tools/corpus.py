"""Frozen corpus access — list, search, and retrieve documents.

Role in the pipeline
--------------------
The text corpus is a directory of ``.txt`` files (and optionally a
``corpus_index.json`` metadata file).  This module provides three tools the
agentic loop (Phase 6) and the Phase 3 batch extractor use:

* ``list_documents(corpus_dir)`` — return all document IDs in deterministic
  order (sorted by filename).
* ``search_corpus(corpus_dir, query, asof, k=10)`` — BM25-style keyword search
  returning top-k doc_ids.  Phase 3 uses simple keyword overlap; a full BM25
  implementation is a drop-in replacement.
* ``get_document(corpus_dir, doc_id)`` — return ``{"doc_id", "timestamp",
  "content"}`` for a single document.  Filters by ``timestamp <= asof``.

All functions enforce the leakage cutoff: no document with ``timestamp > asof``
is ever returned.  This is defence-in-depth — the organiser's staging gate
(g2) already validates corpus timestamps, but the agent re-filters before any
LLM call.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CorpusError",
    "Document",
    "list_documents",
    "search_corpus",
    "get_document",
    "load_corpus_index",
]

_LOG = logging.getLogger(__name__)


class CorpusError(RuntimeError):
    """Raised when a corpus operation fails (missing file, bad index, etc.)."""


@dataclass(frozen=True, slots=True)
class Document:
    """One corpus document, validated and leakage-filtered."""
    doc_id: str
    timestamp: str        # ISO date (YYYY-MM-DD)
    content: str


# -----------------------------------------------------------------------------
# Index loading
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _IndexEntry:
    """Entry from corpus_index.json (if present)."""
    doc_id: str
    timestamp: str
    doc_type: str | None = None
    source: str | None = None


def load_corpus_index(corpus_dir: Path) -> dict[str, _IndexEntry] | None:
    """Load ``corpus_index.json`` if present; return ``None`` if absent or invalid."""
    idx_path = corpus_dir / "corpus_index.json"
    if not idx_path.is_file():
        return None
    try:
        raw = json.loads(idx_path.read_text(encoding="utf-8"))
        entries = {}
        for item in raw.get("documents", []):
            e = _IndexEntry(
                doc_id=item["doc_id"],
                timestamp=item["timestamp"],
                doc_type=item.get("doc_type"),
                source=item.get("source"),
            )
            entries[e.doc_id] = e
        return entries
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        _LOG.warning("Failed to load corpus_index.json: %s", exc)
        return None


# -----------------------------------------------------------------------------
# Timestamp parsing & filtering
# -----------------------------------------------------------------------------


def _parse_iso_date(s: str) -> str | None:
    """Validate ISO date string; return it or ``None`` if invalid."""
    try:
        # Light validation: YYYY-MM-DD
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return None
        return s
    except Exception:
        return None


def _doc_passes_asof(timestamp: str | None, asof: str) -> bool:
    """Return True iff timestamp is present and <= asof (lexicographic works for ISO)."""
    if timestamp is None:
        return False
    return timestamp <= asof


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def list_documents(corpus_dir: Path, asof: str) -> list[str]:
    """Return all document IDs in ``corpus_dir`` with timestamp <= asof.

    Looks for ``corpus_index.json`` first; if absent, falls back to scanning
    ``*.txt`` files and inferring timestamps from filenames (expecting
    ``YYYY-MM-DD`` prefix or embedded date).

    Returns:
        Sorted list of doc_ids that pass the asof filter.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise CorpusError(f"corpus_dir not a directory: {corpus_dir}")

    index = load_corpus_index(corpus_dir)
    if index is not None:
        return sorted(
            doc_id for doc_id, entry in index.items()
            if _doc_passes_asof(entry.timestamp, asof)
        )

    # Fallback: scan *.txt files
    doc_ids = []
    for txt in sorted(corpus_dir.glob("*.txt")):
        # Try to extract timestamp from filename: YYYY-MM-DD_* or *YYYY-MM-DD*
        ts = None
        m = re.match(r"(\d{4}-\d{2}-\d{2})", txt.stem)
        if m:
            ts = m.group(1)
        else:
            # Could also read first line if it's a date header
            try:
                first = txt.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
                if re.match(r"^\d{4}-\d{2}-\d{2}$", first):
                    ts = first
            except Exception:
                pass
        if _doc_passes_asof(ts, asof):
            doc_ids.append(txt.stem)
    return doc_ids


def get_document(corpus_dir: Path, doc_id: str, asof: str) -> Document:
    """Load a single document by ID, enforcing the asof cutoff.

    Looks up ``doc_id`` in the index (if present) or matches ``*.txt`` stem.
    Reads the full text content.  Raises if the document is missing, its
    timestamp is missing/invalid, or timestamp > asof.
    """
    corpus_dir = Path(corpus_dir)
    index = load_corpus_index(corpus_dir)

    if index and doc_id in index:
        entry = index[doc_id]
        if not _doc_passes_asof(entry.timestamp, asof):
            raise CorpusError(f"Document {doc_id} timestamp {entry.timestamp} > asof {asof}")
        txt_path = corpus_dir / f"{doc_id}.txt"
        if not txt_path.is_file():
            # Some corpora use doc_id as full filename
            for ext in (".txt", ".md"):
                cand = corpus_dir / f"{doc_id}{ext}"
                if cand.is_file():
                    txt_path = cand
                    break
            else:
                raise CorpusError(f"Text file for {doc_id} not found in {corpus_dir}")
    else:
        # No index or not in index: find by stem
        candidates = list(corpus_dir.glob(f"{doc_id}.*"))
        if not candidates:
            candidates = list(corpus_dir.glob(f"*{doc_id}*.txt"))
        if not candidates:
            raise CorpusError(f"Document {doc_id} not found in {corpus_dir}")
        txt_path = candidates[0]

    try:
        content = txt_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        raise CorpusError(f"Failed to read {txt_path}: {exc}") from exc

    # Extract timestamp: from index if available, else from filename/content
    timestamp = None
    if index and doc_id in index:
        timestamp = index[doc_id].timestamp
    else:
        m = re.match(r"(\d{4}-\d{2}-\d{2})", txt_path.stem)
        if m:
            timestamp = m.group(1)
        else:
            # Try first line
            first = content.splitlines()[0].strip() if content else ""
            if re.match(r"^\d{4}-\d{2}-\d{2}$", first):
                timestamp = first

    timestamp = _parse_iso_date(timestamp or "")
    if not timestamp or not _doc_passes_asof(timestamp, asof):
        raise CorpusError(
            f"Document {doc_id} has no valid timestamp <= asof {asof}"
        )

    return Document(doc_id=doc_id, timestamp=timestamp, content=content)


def search_corpus(
    corpus_dir: Path,
    query: str,
    asof: str,
    k: int = 10,
) -> list[str]:
    """Simple keyword-overlap search (BM25-lite) returning top-k doc_ids.

    This is a minimal implementation: split query into lowercase tokens,
    score each document by token overlap count, return top-k.  A proper BM25
    implementation can replace this without changing the call sites.

    Args:
        corpus_dir: Directory containing the corpus.
        query: Search string.
        asof: Leakage cutoff (documents with timestamp > asof are excluded).
        k: Max results to return.

    Returns:
        List of up to ``k`` doc_ids, highest score first.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return []

    doc_ids = list_documents(corpus_dir, asof)
    if not doc_ids:
        return []

    scored = []
    for doc_id in doc_ids:
        try:
            doc = get_document(corpus_dir, doc_id, asof)
        except CorpusError:
            continue
        # Simple token overlap score
        doc_tokens = set(re.findall(r"\w+", doc.content.lower()))
        score = len(query_tokens & doc_tokens)
        if score > 0:
            scored.append((score, doc_id))

    scored.sort(key=lambda x: (-x[0], x[1]))  # highest score first, tie-break by id
    return [doc_id for _, doc_id in scored[:k]]