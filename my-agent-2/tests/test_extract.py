"""Tests for `harness.llm.extract` — vector and scenario extraction (with mocked client)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from harness.bridge.vector_to_adjustment import aggregate_vectors
from harness.llm.cache import LLMCache
from harness.llm.extract import (
    ExtractError,
    Scenario,
    Vector,
    _validate_scenarios,
    _validate_vector,
    extract_scenarios,
    extract_vector,
)
from harness.llm.client import LLMClientError


def _mock_client(response: str) -> MagicMock:
    """Create a mock LLMClient that returns `response`."""
    client = MagicMock(spec=["complete"])
    client.complete.return_value = response
    return client


def _mock_cache(hit: bool = False, response: str | None = None) -> MagicMock:
    """Create a mock LLMCache."""
    cache = MagicMock(spec=["get", "set", "has"])
    cache.has.return_value = hit
    cache.get.return_value = response
    return cache


class TestValidateVector:
    def test_valid_vector(self) -> None:
        raw = {
            "drift": 0.3,
            "width": 1.2,
            "skew": -0.1,
            "urgency": 0.7,
            "established": True,
            "evidence": ["Fed holds rates", "Inflation cooling"],
            "contradicts": ["doc_123"],
        }
        vec = _validate_vector(raw)
        assert isinstance(vec, Vector)
        assert vec.drift == 0.3
        assert vec.evidence == ("Fed holds rates", "Inflation cooling")

    def test_missing_field_raises(self) -> None:
        raw = {"drift": 0.0, "width": 1.0, "skew": 0.0, "urgency": 0.5, "evidence": ["e"]}
        with pytest.raises(ExtractError, match="missing fields"):
            _validate_vector(raw)

    def test_out_of_range_raises(self) -> None:
        for f, bad in [("drift", 1.5), ("width", 2.5), ("urgency", 1.1), ("skew", -2.0)]:
            raw = {"drift": 0.0, "width": 1.0, "skew": 0.0, "urgency": 0.5, "established": True, "evidence": ["e"]}
            raw[f] = bad
            with pytest.raises(ExtractError, match=f"out of range"):
                _validate_vector(raw)

    def test_established_must_be_bool(self) -> None:
        raw = {"drift": 0.0, "width": 1.0, "skew": 0.0, "urgency": 0.5, "established": "yes", "evidence": ["e"]}
        with pytest.raises(ExtractError, match="bool"):
            _validate_vector(raw)

    def test_evidence_nonempty(self) -> None:
        raw = {"drift": 0.0, "width": 1.0, "skew": 0.0, "urgency": 0.5, "established": True, "evidence": []}
        with pytest.raises(ExtractError, match="non-empty"):
            _validate_vector(raw)

    def test_contradicts_optional(self) -> None:
        raw = {"drift": 0.0, "width": 1.0, "skew": 0.0, "urgency": 0.5, "established": True, "evidence": ["e"]}
        vec = _validate_vector(raw)
        assert vec.contradicts == ()


class TestValidateScenarios:
    def test_valid_scenarios(self) -> None:
        raw = {
            "scenarios": [
                {"label": "early_cut", "weight": 0.6, "doc_ids": ["a", "b"], "drift": -0.2, "width": 1.1, "skew": 0.0},
                {"label": "higher_longer", "weight": 0.4, "doc_ids": ["c"], "drift": 0.1, "width": 0.9, "skew": 0.1},
            ]
        }
        scens = _validate_scenarios(raw)
        assert len(scens) == 2
        assert abs(sum(s.weight for s in scens) - 1.0) < 1e-9
        assert scens[0].label == "early_cut"
        assert scens[0].doc_ids == ("a", "b")

    def test_too_few_scenarios(self) -> None:
        raw = {"scenarios": [{"label": "x", "weight": 1.0, "doc_ids": ["a"], "drift": 0.0, "width": 1.0, "skew": 0.0}]}
        with pytest.raises(ExtractError, match="2.4"):  # en-dash in message
            _validate_scenarios(raw)

    def test_too_many_scenarios(self) -> None:
        raw = {"scenarios": [{"label": f"s{i}", "weight": 0.2, "doc_ids": ["a"], "drift": 0.0, "width": 1.0, "skew": 0.0} for i in range(5)]}
        with pytest.raises(ExtractError, match="2.4"):
            _validate_scenarios(raw)

    def test_missing_field(self) -> None:
        raw = {"scenarios": [
            {"label": "x", "weight": 1.0, "doc_ids": ["a"]},  # missing drift/width/skew
            {"label": "y", "weight": 1.0, "doc_ids": ["b"], "drift": 0.0, "width": 1.0, "skew": 0.0}
        ]}
        with pytest.raises(ExtractError, match="missing field"):
            _validate_scenarios(raw)

    def test_doc_ids_nonempty(self) -> None:
        raw = {"scenarios": [
            {"label": "x", "weight": 1.0, "doc_ids": [], "drift": 0.0, "width": 1.0, "skew": 0.0},
            {"label": "y", "weight": 1.0, "doc_ids": ["b"], "drift": 0.0, "width": 1.0, "skew": 0.0}
        ]}
        with pytest.raises(ExtractError, match="doc_ids"):
            _validate_scenarios(raw)

    def test_weights_renormalised(self) -> None:
        # Use weights that are valid individually but don't sum to 1
        raw = {"scenarios": [
            {"label": "a", "weight": 0.8, "doc_ids": ["a"], "drift": 0.0, "width": 1.0, "skew": 0.0},
            {"label": "b", "weight": 0.4, "doc_ids": ["b"], "drift": 0.0, "width": 1.0, "skew": 0.0},
        ]}
        scens = _validate_scenarios(raw)
        assert abs(scens[0].weight - 2/3) < 1e-9
        assert abs(scens[1].weight - 1/3) < 1e-9

    def test_zero_total_weight_raises(self) -> None:
        raw = {"scenarios": [
            {"label": "a", "weight": 0.0, "doc_ids": ["a"], "drift": 0.0, "width": 1.0, "skew": 0.0},
            {"label": "b", "weight": 0.0, "doc_ids": ["b"], "drift": 0.0, "width": 1.0, "skew": 0.0},
        ]}
        with pytest.raises(ExtractError, match="sum to zero"):
            _validate_scenarios(raw)


class TestExtractVector:
    def test_cache_hit_returns_cached(self, tmp_path) -> None:
        cache = _mock_cache(hit=True, response=json.dumps({"drift": 0.1, "width": 1.0, "skew": 0.0, "urgency": 0.5, "established": False, "evidence": ["cached"]}))
        client = _mock_client("should not be called")
        vec = extract_vector(client, cache, doc_id="doc1", timestamp="2024-01-01", content="text", asof="2024-06-28", family="T2-F1", target_type="level")
        assert vec.drift == 0.1
        client.complete.assert_not_called()

    def test_cache_miss_calls_model(self, tmp_path) -> None:
        cache = _mock_cache(hit=False)
        client = _mock_client(json.dumps({"drift": 0.2, "width": 1.0, "skew": 0.0, "urgency": 0.5, "established": True, "evidence": ["model"]}))
        vec = extract_vector(client, cache, doc_id="doc2", timestamp="2024-01-01", content="text", asof="2024-06-28", family="T2-F1", target_type="level")
        assert vec.drift == 0.2
        client.complete.assert_called_once()

    def test_model_error_raises_extract_error(self) -> None:
        cache = _mock_cache(hit=False)
        client = _mock_client("{}")
        client.complete.side_effect = LLMClientError("model down")
        with pytest.raises(ExtractError, match="Model call failed"):
            extract_vector(client, cache, doc_id="x", timestamp="2024-01-01", content="t", asof="2024-06-28", family="T2-F1", target_type="level")

    def test_invalid_json_raises(self) -> None:
        cache = _mock_cache(hit=False)
        client = _mock_client("not json")
        with pytest.raises(ExtractError, match="invalid JSON"):
            extract_vector(client, cache, doc_id="x", timestamp="2024-01-01", content="t", asof="2024-06-28", family="T2-F1", target_type="level")

    def test_schema_violation_raises(self) -> None:
        cache = _mock_cache(hit=False)
        client = _mock_client(json.dumps({"drift": 2.0, "width": 1.0, "skew": 0.0, "urgency": 0.5, "established": True, "evidence": ["e"]}))  # drift out of range
        with pytest.raises(ExtractError, match="out of range"):
            extract_vector(client, cache, doc_id="x", timestamp="2024-01-01", content="t", asof="2024-06-28", family="T2-F1", target_type="level")


class TestExtractScenarios:
    def test_cache_hit(self) -> None:
        cache = _mock_cache(hit=True, response=json.dumps({"scenarios": [
            {"label": "a", "weight": 0.6, "doc_ids": ["a"], "drift": 0.0, "width": 1.0, "skew": 0.0},
            {"label": "b", "weight": 0.4, "doc_ids": ["b"], "drift": 0.1, "width": 1.0, "skew": 0.1},
        ]}))
        client = _mock_client("should not be called")
        scens = extract_scenarios(client, cache, documents=[{"doc_id": "a", "timestamp": "2024-01-01", "content": "t"}], asof="2024-06-28", family="T2-F1", target_type="level")
        assert len(scens) == 2
        client.complete.assert_not_called()

    def test_model_error_raises(self) -> None:
        cache = _mock_cache(hit=False)
        client = _mock_client("{}")
        client.complete.side_effect = LLMClientError("down")
        with pytest.raises(ExtractError, match="Model call failed"):
            extract_scenarios(client, cache, documents=[], asof="2024-06-28", family="T2-F1", target_type="level")