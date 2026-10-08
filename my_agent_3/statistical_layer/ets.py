"""Executive summary: exponential-smoothing fallback forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .arima import MIN_HISTORY, _find_column, _horizon_steps


def fit_and_sample(panel: pd.DataFrame, card: dict, asof: str, n_draws: int,
                   seed: int = 43) -> np.ndarray:
    """Fit ETS per asset and return draws shaped [draw, asset, horizon]."""
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
        if len(series) < MIN_HISTORY:
            raise ValueError(f"asset {asset!r} has too little history for ETS")
        y = series.to_numpy()
        if target_type == "log_return":
            if np.any(y <= -1):
                raise ValueError(f"asset {asset!r} contains returns <= -100%")
            y = np.log1p(y)
        elif target_type != "level":
            raise ValueError(f"unsupported target_type {target_type!r}")
        model = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                     seasonal=None, initialization_method="estimated").fit(optimized=True)
        mean = np.asarray(model.forecast(max_steps), dtype=float)
        residual = y - np.asarray(model.fittedvalues, dtype=float)
        residual = residual[np.isfinite(residual)]
        if len(residual) < 2:
            residual = y - float(np.mean(y))
        if len(residual) < 1:
            residual = np.array([0.0])

        # Resample short residual blocks so each horizon belongs to one
        # simulated path instead of receiving an unrelated endpoint error.
        block_length = min(5, len(residual))
        n_blocks = int(np.ceil(max_steps / block_length))
        max_start = len(residual) - block_length + 1
        starts = rngs[ai].integers(0, max_start, size=(n_draws, n_blocks))
        offsets = np.arange(block_length)
        sampled = residual[(starts[:, :, None] + offsets).reshape(n_draws, -1)]
        sampled = sampled[:, :max_steps]
        paths = mean[None, :] + sampled
        for hi, step in enumerate(steps[ai]):
            step = int(step)
            draws[:, ai, hi] = (
                paths[:, :step].sum(axis=1)
                if target_type == "log_return"
                else paths[:, step - 1]
            )
    return draws
