"""Executive summary: blend a common demeaned shock into all assets' draws.

Adding the cross-asset average deviation to every asset raises co-movement
without moving any asset's centre, so the 30% variogram component gains
text-conditioned dependence while marginals keep their calibrated shape.
Only tightening above 1.0 is supported; the bounds cap the blend at half the
common shock. Single-asset cards pass through unchanged.
"""

from __future__ import annotations

import numpy as np

__all__ = ["apply"]


def apply(draws, panel, card: dict, signals: dict, seed: int = 72):
    """Blend ``(correlation_scale - 1)`` of the common demeaned path into draws."""
    scale = float(signals.get("correlation_scale", 1.0))
    out = np.asarray(draws, dtype=float)
    if scale <= 1.0 or out.ndim != 3 or out.shape[1] < 2:
        return out
    centre = np.median(out, axis=0, keepdims=True)
    common = np.mean(out - centre, axis=1, keepdims=True)
    weight = min(scale - 1.0, 0.5)
    return out + weight * common
