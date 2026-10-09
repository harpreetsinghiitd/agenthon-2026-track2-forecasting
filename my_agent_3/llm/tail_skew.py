"""Executive summary: stretch the text-indicated tail decile away from the centre.

Widening is applied by extending the existing outer 10% of draws in the
stated direction — the empirical tail gets fatter while ranks, draw count and
the opposite tail are preserved. Deterministic: no resampling, so ``seed`` is
kept for interface compatibility only.
"""

from __future__ import annotations

import numpy as np

__all__ = ["apply"]


def apply(draws, signals: dict, seed: int = 71):
    """Stretch the left or right tail decile by ``(1 + tail_strength)``."""
    direction = str(signals.get("tail_direction", "none")).strip().lower()
    strength = float(signals.get("tail_strength", 0.0))
    if direction not in ("left", "right") or strength <= 0.0:
        return draws
    out = np.asarray(draws, dtype=float).copy()
    centre = np.median(out, axis=0, keepdims=True)
    level = 0.1 if direction == "left" else 0.9
    threshold = np.quantile(out, level, axis=0, keepdims=True)
    in_tail = out <= threshold if direction == "left" else out >= threshold
    stretched = centre + (out - centre) * (1.0 + strength)
    return np.where(in_tail, stretched, out)
