"""Executive summary: add simple high/low volatility regime scaling to GARCH draws."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import garch


def transform(draws: np.ndarray, panel, card: dict, asof: str,
              seed: int = 53) -> np.ndarray:
    """Apply GARCH scaling, then emphasize recent high-volatility regimes."""
    out = garch.transform(draws, panel, card, asof, seed=seed)
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.loc[data["date"] <= pd.Timestamp(asof)]
    targets = card["targets"]
    for ai, asset in enumerate(targets["asset_ids"]):
        values = data.loc[data["asset"].astype(str).str.lower() == str(asset).lower(), "value"].to_numpy(float)
        changes = np.diff(values)
        if len(changes) < 30:
            continue
        short = float(np.std(changes[-20:]))
        long = float(np.std(changes[:-20])) if len(changes) > 25 else short
        if long > 1e-12 and short > long:
            factor = min(1.5, short / long)
            center = out[:, ai, :].mean(axis=0, keepdims=True)
            out[:, ai, :] = center + (out[:, ai, :] - center) * factor
    return out
