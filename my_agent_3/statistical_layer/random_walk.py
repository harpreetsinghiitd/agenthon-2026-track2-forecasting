"""Executive summary: robust random-walk final fallback forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .arima import _find_column, _horizon_steps


def fit_and_sample(panel: pd.DataFrame, card: dict, asof: str, n_draws: int,
                   seed: int = 44) -> np.ndarray:
    """Sample random-walk paths from historical increments."""
    targets = card["targets"]
    assets = [str(x) for x in targets["asset_ids"]]
    horizons = [int(x) for x in targets["horizons"]]
    target_type = targets["target_type"]
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.loc[data["date"] <= pd.Timestamp(asof)]
    steps = _horizon_steps(data, card, asof, assets, horizons)
    max_steps = int(steps.max())
    wide = data.pivot_table(index="date", columns="asset", values="value").sort_index()
    draws = np.empty((n_draws, len(assets), len(horizons)), dtype=float)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(assets))]
    for ai, asset in enumerate(assets):
        series = wide[_find_column(wide, asset)].dropna().astype(float)
        if len(series) < 2:
            raise ValueError(f"asset {asset!r} needs at least two observations for random walk")
        values = series.to_numpy()
        if target_type == "log_return":
            if np.any(values <= -1):
                raise ValueError(f"asset {asset!r} contains returns <= -100%")
            increments = np.log1p(values)
            mu = float(np.mean(increments))
            residuals = increments - mu
            paths = rngs[ai].choice(residuals if len(residuals) else np.array([0.0]),
                                     size=(n_draws, max_steps), replace=True) + mu
            for hi, step in enumerate(steps[ai]):
                draws[:, ai, hi] = paths[:, :int(step)].sum(axis=1)
        elif target_type == "level":
            last = float(values[-1])
            increments = np.diff(values)
            mu = float(np.mean(increments)) if len(increments) else 0.0
            residuals = increments - mu
            sampled = rngs[ai].choice(residuals if len(residuals) else np.array([0.0]),
                                      size=(n_draws, max_steps), replace=True) + mu
            paths = last + np.cumsum(sampled, axis=1)
            for hi, step in enumerate(steps[ai]):
                draws[:, ai, hi] = paths[:, int(step) - 1]
        else:
            raise ValueError(f"unsupported target_type {target_type!r}")
    return draws
