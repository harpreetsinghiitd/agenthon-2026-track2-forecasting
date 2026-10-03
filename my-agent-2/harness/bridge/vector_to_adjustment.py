"""Deterministic bridge: aggregated vector → per-asset forecast parameters.

Role in the pipeline
--------------------
The LLM extracts per-document vectors (drift, width, skew, urgency, established,
evidence).  The agent aggregates them into a single vector per asset (e.g. by
averaging drift/width/skew weighted by urgency × established).  This module
converts that aggregated vector into the three parameters that
``simulate_mixture`` consumes:

* ``drift`` — per-step centre shift (can be positive or negative).
* ``width_mult`` — multiplier on the bootstrap innovation spread.
* ``skew`` — passed through to Phase 5+ (GARCH/regimes); unused in Phase 3.

Hard constraints (from the build prompt, §10):
* Drift is capped at ``0.15 × horizon_sd`` (the text may nudge, not dominate).
* Only ``established=true`` adjustments may *reduce* ``width_mult`` below 1.0.
  ``established=false`` (inferred) adjustments may move the centre but must not
  narrow the distribution.
* All maths is pure and deterministic — no RNG, no external state.

The function signature matches §10.1 of the design doc.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.llm.extract import Vector

__all__ = ["BridgeError", "Adjustment", "vector_to_adjustment"]


class BridgeError(ValueError):
    """Raised when the vector or horizon_sd produces invalid parameters."""


@dataclass(frozen=True, slots=True)
class Adjustment:
    """Deterministic forecast adjustments for one asset."""
    drift: float      # per-step centre shift (capped at 0.15 * horizon_sd)
    width_mult: float # multiplier on bootstrap innovation spread (≥ 0)
    skew: float       # tail asymmetry hint (passed to Phase 5+; unused in P3)


#: Maximum absolute drift as a fraction of the horizon spread.
_DRIFT_CAP_FRAC: float = 0.15


def vector_to_adjustment(
    vector: Vector,
    horizon_sd: float,
) -> Adjustment:
    """Convert an aggregated signal vector into forecast adjustments.

    The vector's ``drift`` is a unitless direction in [-1, 1].  We scale it by
    the empirical horizon spread and the urgency weight, then cap at
    ``0.15 × horizon_sd`` per the build prompt.

    The vector's ``width`` is a unitless uncertainty in [0, 2].  We map it to a
    multiplicative spread factor: 1.0 = no change, >1 = widen, <1 = narrow.
    CRITICAL: only ``established=true`` vectors may narrow (width_mult < 1.0).
    Inferred vectors (established=false) are floored at 1.0.

    ``skew`` is passed through unchanged for future phases.

    Args:
        vector: Aggregated signal vector for one asset.
        horizon_sd: Baseline spread from ``compute_horizon_sd`` (positive).

    Returns:
        ``Adjustment`` with drift, width_mult, skew.

    Raises:
        BridgeError: if ``horizon_sd`` is not positive or vector fields are
            out of expected ranges (defensive; vectors should already be valid).
    """
    if horizon_sd <= 0.0:
        raise BridgeError(f"horizon_sd must be positive, got {horizon_sd}")

    # --- Drift: scale direction by urgency, cap at 0.15 × horizon_sd ---
    raw_drift = vector.drift * vector.urgency
    max_drift = _DRIFT_CAP_FRAC * horizon_sd
    drift = max(-max_drift, min(max_drift, raw_drift))

    # --- Width multiplier: map [0,2] → multiplicative factor ---
    # vector.width = 1.0 means "no opinion" → width_mult = 1.0
    # width < 1.0 = more certain → narrow (only if established)
    # width > 1.0 = less certain → widen (always allowed)
    if vector.width < 1.0:
        # Candidate narrowing factor
        candidate = vector.width
        if vector.established:
            width_mult = candidate
        else:
            # Inferred certainty cannot narrow the distribution
            width_mult = 1.0
    else:
        # Widening is always allowed
        width_mult = vector.width

    if width_mult < 0.0:
        # Defensive; validated vectors have width >= 0
        raise BridgeError(f"width_mult negative after mapping: {width_mult}")

    skew = vector.skew  # pass-through for Phase 5+

    return Adjustment(drift=drift, width_mult=width_mult, skew=skew)


def aggregate_vectors(
    vectors: list[Vector],
    *,
    weights: list[float] | None = None,
) -> Vector:
    """Aggregate multiple per-document vectors into one asset-level vector.

    Weighting scheme (per §10.1 design):
    * Each vector's weight = urgency × (2.0 if established else 1.0).
    * If explicit ``weights`` are provided (e.g. from scenario mixing), they
      override the default.

    The aggregation is a weighted mean for numeric fields; ``established`` is
    ``True`` iff *all* contributing vectors are established (conservative).
    ``evidence`` and ``contradicts`` are concatenated (deduplicated).

    Args:
        vectors: Non-empty list of per-document vectors.
        weights: Optional explicit weights (same length as ``vectors``).

    Returns:
        Single aggregated ``Vector``.

    Raises:
        BridgeError: empty input or length mismatch.
    """
    if not vectors:
        raise BridgeError("aggregate_vectors: empty vector list")
    n = len(vectors)
    if weights is not None and len(weights) != n:
        raise BridgeError(f"weights length {len(weights)} != vectors length {n}")

    # Default weights: urgency * (2 if established else 1)
    if weights is None:
        weights = [v.urgency * (2.0 if v.established else 1.0) for v in vectors]

    total_w = sum(weights)
    if total_w <= 0.0:
        # Fallback to uniform if all weights zero (e.g. all urgency=0)
        weights = [1.0] * n
        total_w = float(n)

    def wmean(field: str) -> float:
        return sum(w * getattr(v, field) for v, w in zip(vectors, weights)) / total_w

    established_all = all(v.established for v in vectors)

    # Concatenate evidence/contradicts, deduplicate preserving order
    def concat(field: str) -> tuple[str, ...]:
        seen = set()
        out = []
        for v in vectors:
            for item in getattr(v, field):
                if item not in seen:
                    seen.add(item)
                    out.append(item)
        return tuple(out)

    return Vector(
        drift=wmean("drift"),
        width=wmean("width"),
        skew=wmean("skew"),
        urgency=wmean("urgency"),
        established=established_all,
        evidence=concat("evidence"),
        contradicts=concat("contradicts"),
    )