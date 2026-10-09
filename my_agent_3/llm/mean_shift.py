"""Executive summary: shift each cell's centre by a bounded, unitless amount.

The signal's sigma is measured against the draws' own per-cell standard
deviation, so the witness stays panel-blind: it reports direction in
standard-deviation units and never sees a price level.
"""

from __future__ import annotations

import numpy as np

__all__ = ["apply"]


def apply(draws, signals: dict):
    """Shift every asset x horizon cell by ``mean_shift_sigma`` times its sd."""
    sigma = float(signals.get("mean_shift_sigma", 0.0))
    if sigma == 0.0:
        return draws
    out = np.asarray(draws, dtype=float).copy()
    sd = out.std(axis=0, keepdims=True)
    # A degenerate (zero-sd) cell has no scale to interpret; leave it alone.
    out = out + sigma * np.where(sd > 0, sd, 0.0)
    return out
