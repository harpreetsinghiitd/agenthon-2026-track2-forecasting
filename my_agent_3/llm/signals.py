"""Executive summary: validate, hard-bound, and expose the text-signal contract.

Every numeric field is clamped to a fixed range no model response can exceed,
categoricals are normalised, and the playbook's narrowing rule is enforced
here (the single choke point): only a corpus-*stated* (``established``)
adjustment may scale variance below 1.0. Any endpoint or validation failure
degrades to ``NEUTRAL`` so the forecast always proceeds without text.
"""

from __future__ import annotations

import json

from . import client, prompts

__all__ = ["NEUTRAL", "BOUNDS", "parse_signals", "extract_signals"]

NEUTRAL = {
    "mean_shift_sigma": 0.0,
    "variance_scale": 1.0,
    "tail_direction": "none",
    "tail_strength": 0.0,
    "correlation_scale": 1.0,
    "shock_probability": 0.0,
    "basis": "inferred",
    "evidence": "No text adjustment was applied: the LLM layer ran in neutral mode.",
}

#: Hard clamps; the witness cannot move a knob outside these bounds.
BOUNDS = {
    "mean_shift_sigma": (-0.5, 0.5),
    "variance_scale": (0.75, 1.75),
    "tail_strength": (0.0, 1.0),
    "correlation_scale": (1.0, 1.5),
    "shock_probability": (0.0, 1.0),
}

_TAIL_DIRECTIONS = ("left", "right", "none")
_BASES = ("established", "inferred")
_MAX_EVIDENCE_CHARS = 2000


def _number(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(name: str, value: float) -> float:
    low, high = BOUNDS[name]
    return min(high, max(low, value))


def parse_signals(raw: str) -> dict:
    """Validate and hard-bound one model response, or raise ``ValueError``."""
    text = str(raw).strip()
    if text.startswith("```"):  # tolerate fenced JSON despite the prompt
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response contains no JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("signal payload must be a JSON object")

    signal = dict(NEUTRAL)
    signal["mean_shift_sigma"] = _clamp(
        "mean_shift_sigma", _number(payload.get("mean_shift_sigma", 0.0), 0.0)
    )
    basis = str(payload.get("basis", "inferred")).strip().lower()
    basis = basis if basis in _BASES else "inferred"
    scale = _clamp("variance_scale", _number(payload.get("variance_scale", 1.0), 1.0))
    # The narrowing rule lives here so no caller can bypass it: inferred
    # tone may move the centre but must not tighten the spread.
    if basis != "established":
        scale = max(scale, 1.0)
    signal["variance_scale"] = scale
    signal["basis"] = basis
    direction = str(payload.get("tail_direction", "none")).strip().lower()
    signal["tail_direction"] = direction if direction in _TAIL_DIRECTIONS else "none"
    signal["tail_strength"] = _clamp(
        "tail_strength", _number(payload.get("tail_strength", 0.0), 0.0)
    )
    signal["correlation_scale"] = _clamp(
        "correlation_scale", _number(payload.get("correlation_scale", 1.0), 1.0)
    )
    signal["shock_probability"] = _clamp(
        "shock_probability", _number(payload.get("shock_probability", 0.0), 0.0)
    )
    evidence = str(payload.get("evidence", "")).strip()
    signal["evidence"] = evidence[:_MAX_EVIDENCE_CHARS] or NEUTRAL["evidence"]
    return signal


def extract_signals(documents: list[dict], asof: str, card: dict) -> dict:
    """Elicit, validate, and bound corpus signals; ``NEUTRAL`` on any failure.

    This is the only entry point the pipeline calls; conditioning functions can
    therefore assume a fully bounded signal dictionary.
    """
    if not documents:
        return dict(NEUTRAL)
    try:
        raw = client.chat_completion(prompts.build_messages(documents, asof, card))
        return parse_signals(raw)
    except Exception as exc:  # endpoint down, garbage response, bad JSON: pass through
        fallback = dict(NEUTRAL)
        fallback["evidence"] = f"No text adjustment was applied ({exc})."
        return fallback
