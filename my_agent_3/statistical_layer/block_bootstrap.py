"""Executive summary: sample multi-step forecasts from historical change blocks.

This model preserves short runs of volatility and return dependence by resampling
contiguous blocks of transformed observations. It does not read text or outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .arima import _find_column, _horizon_steps


def fit_and_sample(
    panel: pd.DataFrame,
    card: dict,
    asof: str,
    n_draws: int,
    seed: int = 47,
    block_length: int = 10,
) -> np.ndarray:
    """Resample overlapping historical change blocks into forecast paths.

    Log-return targets use ``log1p`` and are summed. Positive FX levels (identified
    by a ``*_per_*`` value unit) use log changes and are exponentiated back. Other
    levels use additive changes. Horizon keys remain unchanged in the output grid;
    explicit monthly mappings are resolved through the shared horizon helper.
    """
    if block_length < 1:
        raise ValueError("block_length must be at least one")
    targets = card["targets"]
    assets = [str(x) for x in targets["asset_ids"]]
    horizons = [int(x) for x in targets["horizons"]]
    target_type = targets["target_type"]
    value_unit = str(targets.get("value_unit", "")).lower()
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date", "asset", "value"])
    data = data.loc[data["date"] <= pd.Timestamp(asof)]
    steps = _horizon_steps(data, card, asof, assets, horizons)
    if steps.shape != (len(assets), len(horizons)) or np.any(steps <= 0):
        raise ValueError(f"invalid resolved forecast steps: {steps}")

    wide = data.pivot_table(index="date", columns="asset", values="value").sort_index()
    draws = np.empty((n_draws, len(assets), len(horizons)), dtype=np.float64)
    asset_rngs = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(len(assets))
    ]

    for ai, asset in enumerate(assets):
        series = wide[_find_column(wide, asset)].dropna().astype(float)
        values = series.to_numpy(dtype=np.float64)
        if len(values) < 2:
            raise ValueError(f"asset {asset!r} needs at least two observations")

        if target_type == "log_return":
            if np.any(values <= -1.0):
                raise ValueError(f"asset {asset!r} contains returns <= -100%")
            increments = np.log1p(values)
            mode = "sum"
            anchor = 0.0
        elif target_type == "level" and "_per_" in value_unit:
            if np.any(values <= 0.0):
                raise ValueError(f"FX level series for {asset!r} must be positive")
            increments = np.diff(np.log(values))
            mode = "log_level"
            anchor = float(np.log(values[-1]))
        elif target_type == "level":
            increments = np.diff(values)
            mode = "level"
            anchor = float(values[-1])
        else:
            raise ValueError(f"unsupported target_type {target_type!r}")

        if not len(increments):
            raise ValueError(f"asset {asset!r} has no usable changes")
        n_steps = int(np.max(steps[ai]))
        effective_block = min(block_length, len(increments))
        n_blocks = int(np.ceil(n_steps / effective_block))
        max_start = len(increments) - effective_block + 1
        starts = asset_rngs[ai].integers(0, max_start, size=(n_draws, n_blocks))
        paths = np.empty((n_draws, n_steps), dtype=np.float64)
        offsets = np.arange(effective_block)
        for draw_index in range(n_draws):
            sampled = increments[(starts[draw_index, :, None] + offsets).ravel()]
            paths[draw_index] = np.cumsum(sampled[:n_steps])

        for hi, step in enumerate(steps[ai]):
            accumulated = paths[:, int(step) - 1]
            if mode == "sum":
                draws[:, ai, hi] = accumulated
            elif mode == "log_level":
                draws[:, ai, hi] = np.exp(anchor + accumulated)
            else:
                draws[:, ai, hi] = anchor + accumulated

    return draws
