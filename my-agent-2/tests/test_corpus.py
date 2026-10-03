"""Tests for `harness.tools.corpus` — list, search, get document."""

from __future__ import annotations

import pathlib

import pytest

from harness.tools.corpus import (
    CorpusError,
    Document,
    get_document,
    list_documents,
    search_corpus,
)


def _make_corpus(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal corpus with index and a few txt files."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "corpus_index.json").write_text(
        """{
  "documents": [
    {"doc_id": "doc_a", "timestamp": "2024-01-01", "doc_type": "statement"},
    {"doc_id": "doc_b", "timestamp": "2024-06-15", "doc_type": "minutes"},
    {"doc_id": "doc_c", "timestamp": "2025-01-01", "doc_type": "statement"}
  ]
}""",
        encoding="utf-8",
    )
    (c / "doc_a.txt").write_text("2024-01-01\nFed holds rates steady.", encoding="utf-8")
    (c / "doc_b.txt").write_text("2024-06-15\nInflation easing, cuts coming.", encoding="utf-8")
    (c / "doc_c.txt").write_text("2025-01-01\nNew policy framework.", encoding="utf-8")
    return c


def test_list_documents_filters_by_asof(tmp_path: pathlib.Path) -> None:
    c = _make_corpus(tmp_path)
    # asof before doc_c
    ids = list_documents(c, "2024-12-31")
    assert set(ids) == {"doc_a", "doc_b"}
    # asof after all
    ids = list_documents(c, "2025-12-31")
    assert set(ids) == {"doc_a", "doc_b", "doc_c"}
    # asof before all
    ids = list_documents(c, "2023-12-31")
    assert ids == []


def test_get_document_returns_full_object(tmp_path: pathlib.Path) -> None:
    c = _make_corpus(tmp_path)
    doc = get_document(c, "doc_a", "2024-12-31")
    assert isinstance(doc, Document)
    assert doc.doc_id == "doc_a"
    assert doc.timestamp == "2024-01-01"
    assert "Fed holds rates" in doc.content


def test_get_document_rejects_future_asof(tmp_path: pathlib.Path) -> None:
    c = _make_corpus(tmp_path)
    with pytest.raises(CorpusError, match="timestamp .* > asof"):
        get_document(c, "doc_c", "2024-12-31")


def test_get_document_missing_raises(tmp_path: pathlib.Path) -> None:
    c = _make_corpus(tmp_path)
    with pytest.raises(CorpusError, match="not found"):
        get_document(c, "nonexistent", "2024-12-31")


def test_search_corpus_keyword_overlap(tmp_path: pathlib.Path) -> None:
    c = _make_corpus(tmp_path)
    # Query matching doc_a
    hits = search_corpus(c, "Fed holds", "2024-12-31", k=5)
    assert "doc_a" in hits
    # Query matching doc_b
    hits = search_corpus(c, "inflation easing", "2024-12-31", k=5)
    assert "doc_b" in hits
    # Empty query
    assert search_corpus(c, "", "2024-12-31") == []


def test_search_corpus_respects_asof(tmp_path: pathlib.Path) -> None:
    c = _make_corpus(tmp_path)
    # doc_c is post-asof, should not appear even if it matches
    hits = search_corpus(c, "policy", "2024-12-31", k=10)
    assert "doc_c" not in hits


def test_corpus_without_index_fallbacks_to_filenames(tmp_path: pathlib.Path) -> None:
    c = tmp_path / "corpus_no_index"
    c.mkdir()
    (c / "2024-01-01_fomc.txt").write_text("2024-01-01\nStatement.", encoding="utf-8")
    (c / "2024-06-15_minutes.txt").write_text("2024-06-15\nMinutes.", encoding="utf-8")
    ids = list_documents(c, "2024-12-31")
    assert set(ids) == {"2024-01-01_fomc", "2024-06-15_minutes"}
    doc = get_document(c, "2024-01-01_fomc", "2024-12-31")
    assert doc.timestamp == "2024-01-01"