"""Executive summary: scale each cell's dispersion by the bounded variance signal.

Scaling is centred on the per-cell median so a few extreme draws cannot drag
the centre. The established-vs-inferred narrowing rule is enforced once, in
``signals.parse_signals``; this function trusts the bounded dictionary.
"""

from __future__ import annotations

import numpy as np

__all__ = ["apply"]


def apply(draws, signals: dict):
    """Widen or tighten each asset x horizon cell around its median."""
    scale = float(signals.get("variance_scale", 1.0))
    if scale == 1.0:
        return draws
    out = np.asarray(draws, dtype=float).copy()
    centre = np.median(out, axis=0, keepdims=True)
    return centre + (out - centre) * scale
