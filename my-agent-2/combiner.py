"""Combine raw draws with LLM signals into adjusted draws.

Currently a passthrough when signals is None.

When implemented, apply() should modify the draws according to the signals,
e.g. shift the mean, scale the spread, fatten a tail, adjust correlations.
"""
import numpy as np


def apply(draws: np.ndarray, signals: dict | None) -> np.ndarray:
    if signals is None:
        return draws

    adjusted = draws.copy()

    # Example rules (uncomment and tune when signals exist):
    # if signals.get("policy_stance") == "hawkish":
    #     adjusted += 0.005
    # if signals.get("uncertainty") == "high":
    #     adjusted *= 1.5
    # if signals.get("tail_direction") == "left":
    #     adjusted = adjusted - 0.5 * np.maximum(0.0, -adjusted) ** 2

    return adjusted