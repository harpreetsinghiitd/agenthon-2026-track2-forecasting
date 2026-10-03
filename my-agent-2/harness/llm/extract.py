"""LLM feature extraction — vectors and scenarios from the frozen corpus.

Role in the pipeline
--------------------
Phase 3's text layer turns the frozen corpus into structured signals that the
deterministic bridge can consume.  Two extraction modes:

* ``extract_vector`` — one vector per document, capturing the document's
  stance on policy, inflation, growth, risk, etc.  The vector schema is
  defined in §9.1.1 of the design (drift, width, skew, urgency, established,
  evidence doc_ids, contradicts).

* ``extract_scenarios`` — 2–4 global scenarios with weights and supporting
  doc_ids.  Each scenario is a narrative (e.g. "Fed cuts early", "Sticky inflation")
  that the bridge later mixes into per-draw parameters.

Both functions are cached via ``LLMCache`` so a second run with the same
corpus hits the disk cache and produces byte-identical output.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.llm.cache import LLMCache
from harness.llm.client import LLMClient, LLMClientError

__all__ = [
    "ExtractError",
    "Vector",
    "Scenario",
    "extract_vector",
    "extract_scenarios",
]

_LOG = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# JSON Schemas (passed to the model via response_format)
# -----------------------------------------------------------------------------

_VECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "drift": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "width": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        "skew": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "urgency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "established": {"type": "boolean"},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "contradicts": {"type": "array", "items": {"type": "string"}, "default": []},
    },
    "required": ["drift", "width", "skew", "urgency", "established", "evidence"],
    "additionalProperties": False,
}

_SCENARIOS_SCHEMA = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "minLength": 1},
                    "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "doc_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "drift": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                    "width": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                    "skew": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                },
                "required": ["label", "weight", "doc_ids", "drift", "width", "skew"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scenarios"],
    "additionalProperties": False,
}

# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

_SYSTEM_VECTOR = """You are a central-bank text analyst.  Read the provided document and
return a structured vector capturing its implications for financial forecasting.

Output ONLY valid JSON matching the schema.  Do not include commentary.

Fields:
- drift: policy direction implied by the text.  -1 = strongly dovish, 0 = neutral,
  +1 = strongly hawkish.  This is a *direction*, not a magnitude.
- width: uncertainty implied.  0 = certain, 2 = maximum uncertainty.  Higher when
  the text is vague, conditional, or highlights unknowns.
- skew: tail asymmetry.  -1 = left-tail risk (crash / easing), +1 = right-tail risk
  (spike / tightening), 0 = symmetric.
- urgency: how soon the implied move matters.  0 = long-run, 1 = immediate.
- established: true iff the text states a *done deal* (decision taken, policy
  enacted, data released).  false for forward guidance, speculation, leaks.
- evidence: list of short quotes (max 160 chars each) from the document that
  support your assessment.  Must be verbatim substrings.
- contradicts: list of doc_ids (if known) whose conclusions this document
  opposes.  Empty array if no contradiction.

All numeric fields are floats in the indicated ranges.  Be conservative: when
uncertain, shrink toward 0.
"""

_SYSTEM_SCENARIOS = """You are a scenario designer for probabilistic forecasting.
Given a set of document summaries, propose 2–4 mutually exclusive macro scenarios
that cover the plausible range of outcomes.  Each scenario gets a weight (summing
to 1) and a quantitative imprint (drift/width/skew).

Output ONLY valid JSON matching the schema.  Do not include commentary.

Fields per scenario:
- label: short human-readable name (e.g. "early_cut", "higher_for_longer").
- weight: probability mass (0–1).  Weights will be renormalised to sum to 1.
- doc_ids: list of document IDs that support this scenario.
- drift: scenario-level policy drift (-1 to 1).
- width: scenario-level uncertainty (0 to 2).
- skew: scenario-level tail asymmetry (-1 to 1).

The scenarios should be *mutually exclusive* narratives, not minor variations.
"""

_USER_VECTOR_TEMPLATE = """Document ID: {doc_id}
Timestamp: {timestamp}
Content:
{content}

AS-OF date: {asof}
Card family: {family}
Target type: {target_type}

Extract the vector for this document only.  Do not use outside knowledge.
"""

_USER_SCENARIOS_TEMPLATE = """AS-OF date: {asof}
Card family: {family}
Target type: {target_type}

Document summaries:
{summaries}

Propose 2–4 scenarios covering the plausible macro narratives.  Each scenario
must cite at least one doc_id from the list above.
"""

# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Vector:
    """One document's extracted signal vector."""
    drift: float      # -1..1
    width: float      # 0..2
    skew: float       # -1..1
    urgency: float    # 0..1
    established: bool
    evidence: tuple[str, ...]
    contradicts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A global macro scenario with weight and quantitative imprint."""
    label: str
    weight: float      # 0..1, renormalised later
    doc_ids: tuple[str, ...]
    drift: float       # -1..1
    width: float       # 0..2
    skew: float        # -1..1


class ExtractError(RuntimeError):
    """Raised when extraction fails validation or the model returns garbage."""


# -----------------------------------------------------------------------------
# Core extraction helpers
# -----------------------------------------------------------------------------


def _validate_vector(obj: dict[str, Any]) -> Vector:
    """Validate against the vector schema and return a Vector."""
    required = {"drift", "width", "skew", "urgency", "established", "evidence"}
    missing = required - obj.keys()
    if missing:
        raise ExtractError(f"Vector missing fields: {missing}")
    for f in ("drift", "skew", "urgency"):
        v = obj[f]
        if not isinstance(v, (int, float)) or not (-1.0 <= v <= 1.0):
            raise ExtractError(f"Vector.{f} out of range [-1,1]: {v}")
    if not (0.0 <= obj["width"] <= 2.0):
        raise ExtractError(f"Vector.width out of range [0,2]: {obj['width']}")
    if not isinstance(obj["established"], bool):
        raise ExtractError(f"Vector.established must be bool: {obj['established']!r}")
    if not isinstance(obj["evidence"], list) or not obj["evidence"]:
        raise ExtractError("Vector.evidence must be non-empty list")
    contradicts = obj.get("contradicts", [])
    if not isinstance(contradicts, list):
        raise ExtractError("Vector.contradicts must be list")
    return Vector(
        drift=float(obj["drift"]),
        width=float(obj["width"]),
        skew=float(obj["skew"]),
        urgency=float(obj["urgency"]),
        established=obj["established"],
        evidence=tuple(str(e)[:160] for e in obj["evidence"]),
        contradicts=tuple(str(c) for c in contradicts),
    )


def _validate_scenarios(obj: dict[str, Any]) -> list[Scenario]:
    """Validate against the scenarios schema and return list[Scenario]."""
    scens = obj.get("scenarios")
    if not isinstance(scens, list) or not (2 <= len(scens) <= 4):
        raise ExtractError("scenarios must be list of 2–4 items")
    out: list[Scenario] = []
    for s in scens:
        for f in ("label", "weight", "doc_ids", "drift", "width", "skew"):
            if f not in s:
                raise ExtractError(f"Scenario missing field: {f}")
        if not isinstance(s["doc_ids"], list) or not s["doc_ids"]:
            raise ExtractError("Scenario.doc_ids must be non-empty list")
        for f in ("drift", "skew"):
            v = s[f]
            if not isinstance(v, (int, float)) or not (-1.0 <= v <= 1.0):
                raise ExtractError(f"Scenario.{f} out of range [-1,1]: {v}")
        if not (0.0 <= s["width"] <= 2.0):
            raise ExtractError(f"Scenario.width out of range [0,2]: {s['width']}")
        w = float(s["weight"])
        if not (0.0 <= w <= 1.0):
            raise ExtractError(f"Scenario.weight out of range [0,1]: {w}")
        out.append(
            Scenario(
                label=str(s["label"]),
                weight=w,
                doc_ids=tuple(str(d) for d in s["doc_ids"]),
                drift=float(s["drift"]),
                width=float(s["width"]),
                skew=float(s["skew"]),
            )
        )
    # Renormalise weights to sum to 1
    total = sum(s.weight for s in out)
    if total <= 0.0:
        raise ExtractError("Scenario weights sum to zero")
    out = [
        Scenario(
            label=s.label,
            weight=s.weight / total,
            doc_ids=s.doc_ids,
            drift=s.drift,
            width=s.width,
            skew=s.skew,
        )
        for s in out
    ]
    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def extract_vector(
    client: LLMClient,
    cache: LLMCache,
    *,
    doc_id: str,
    timestamp: str,
    content: str,
    asof: str,
    family: str,
    target_type: str,
) -> Vector:
    """Extract a signal vector from one document, with caching.

    The cache key incorporates model name, the full prompt (system + user),
    and the JSON schema.  If a cached entry exists, it is returned without
    calling the model.

    Args:
        client: Configured House model client.
        cache: Content-hash cache instance.
        doc_id: Document identifier (used in prompt + evidence).
        timestamp: Document timestamp (ISO date).
        content: Full document text.
        asof: Card's data cutoff (YYYY-MM-DD).
        family: Card family (T2-F1..T2-F4).
        target_type: level | log_return | spread.

    Returns:
        Validated ``Vector``.

    Raises:
        ExtractError: model error, schema violation, or cache I/O failure.
    """
    prompt = _USER_VECTOR_TEMPLATE.format(
        doc_id=doc_id,
        timestamp=timestamp,
        content=content[:8000],  # truncate to avoid token overflow
        asof=asof,
        family=family,
        target_type=target_type,
    )
    if cache.has(MODEL_NAME, _SYSTEM_VECTOR + prompt, _VECTOR_SCHEMA):
        cached = cache.get(MODEL_NAME, _SYSTEM_VECTOR + prompt, _VECTOR_SCHEMA)
        if cached:
            _LOG.debug("Cache hit for vector %s", doc_id)
            return _validate_vector(json.loads(cached))

    _LOG.debug("Calling model for vector %s", doc_id)
    try:
        raw = client.complete(
            system_prompt=_SYSTEM_VECTOR,
            user_prompt=prompt,
            schema=_VECTOR_SCHEMA,
            temperature=0.0,
        )
    except LLMClientError as exc:
        raise ExtractError(f"Model call failed for {doc_id}: {exc}") from exc

    try:
        vec = _validate_vector(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise ExtractError(f"Model returned invalid JSON for {doc_id}: {exc}") from exc

    cache.set(MODEL_NAME, _SYSTEM_VECTOR + prompt, raw, _VECTOR_SCHEMA)
    return vec


def extract_scenarios(
    client: LLMClient,
    cache: LLMCache,
    *,
    documents: list[dict[str, str]],  # each: {"doc_id", "timestamp", "content"}
    asof: str,
    family: str,
    target_type: str,
) -> list[Scenario]:
    """Extract 2–4 global scenarios from the corpus, with caching.

    Args:
        client: House model client.
        cache: Content-hash cache.
        documents: List of {"doc_id", "timestamp", "content"} dicts.
        asof, family, target_type: Card context.

    Returns:
        List of validated ``Scenario`` (weights renormalised to 1).
    """
    summaries = "\n\n".join(
        f"DOC_ID: {d['doc_id']}\nTIMESTAMP: {d['timestamp']}\nCONTENT: {d['content'][:2000]}"
        for d in documents
    )
    prompt = _USER_SCENARIOS_TEMPLATE.format(
        summaries=summaries,
        asof=asof,
        family=family,
        target_type=target_type,
    )
    cache_key_prompt = _SYSTEM_SCENARIOS + prompt
    if cache.has(MODEL_NAME, cache_key_prompt, _SCENARIOS_SCHEMA):
        cached = cache.get(MODEL_NAME, cache_key_prompt, _SCENARIOS_SCHEMA)
        if cached:
            _LOG.debug("Cache hit for scenarios")
            return _validate_scenarios(json.loads(cached))

    _LOG.debug("Calling model for scenarios")
    try:
        raw = client.complete(
            system_prompt=_SYSTEM_SCENARIOS,
            user_prompt=prompt,
            schema=_SCENARIOS_SCHEMA,
            temperature=0.0,
        )
    except LLMClientError as exc:
        raise ExtractError(f"Model call failed for scenarios: {exc}") from exc

    try:
        scens = _validate_scenarios(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise ExtractError(f"Model returned invalid JSON for scenarios: {exc}") from exc

    cache.set(MODEL_NAME, cache_key_prompt, raw, _SCENARIOS_SCHEMA)
    return scens


# Re-export for downstream modules
from harness.llm.client import _MODEL_NAME_ENV as MODEL_NAME  # noqa: E402