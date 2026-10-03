"""Deterministic scenario mixing — scenarios + seed → per-draw parameters.

Role in the pipeline
--------------------
``extract_scenarios`` returns 2–4 macro scenarios, each with a weight, drift,
width, and skew.  This module turns those into per-draw parameters for
``simulate_mixture``:

* Each draw is assigned a scenario via a multinomial draw using the scenario
  weights.  The draw index is seeded so the assignment is deterministic.
* The draw inherits that scenario's (drift, width, skew).

This is the Phase 3→4 bridge: the LLM proposes narratives; the deterministic
mixer converts them to per-draw numbers that ``simulate_mixture`` consumes.
"""

from __future__ import annotations

from harness.llm.extract import Scenario

__all__ = ["scenario_mix"]


def scenario_mix(
    scenarios: list[tuple[float, float, float]],  # (drift, width, skew) per scenario
    weights: list[float],
    n_draws: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """Assign each draw a scenario and return per-draw parameter arrays.

    The assignment uses a multinomial distribution with the given weights,
    seeded for determinism.  All draws with the same seed and inputs produce
    identical assignments.

    Args:
        scenarios: List of (drift, width, skew) tuples, one per scenario.
        weights: Scenario weights (must sum to ~1.0, length matches scenarios).
        n_draws: Number of draws to generate.
        seed: Integer seed (typically derived from ``seed_from_unit_id``).

    Returns:
        Three lists of length ``n_draws``: (drift_per_draw, width_per_draw, skew_per_draw).

    Raises:
        ValueError: length mismatch, weights don't sum to ~1, or n_draws < 1.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")
    if len(scenarios) != len(weights):
        raise ValueError(f"scenarios ({len(scenarios)}) != weights ({len(weights)})")
    if not scenarios:
        raise ValueError("at least one scenario required")
    total_w = sum(weights)
    if abs(total_w - 1.0) > 1e-9:
        # Renormalise defensively
        weights = [w / total_w for w in weights]

    import numpy as np

    rng = np.random.default_rng(seed)
    # Multinomial: one trial per draw, outcome = scenario index
    assign = rng.multinomial(1, weights, size=n_draws).argmax(axis=1)

    drift_out = [scenarios[i][0] for i in assign]
    width_out = [scenarios[i][1] for i in assign]
    skew_out = [scenarios[i][2] for i in assign]

    return drift_out, width_out, skew_out