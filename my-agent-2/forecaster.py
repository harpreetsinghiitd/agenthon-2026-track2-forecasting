"""Numerical forecaster.

Returns draws of shape [n_draws, n_assets, n_horizons].
"""
import numpy as np
import pandas as pd


def _find_column(wide: pd.DataFrame, asset_id: str) -> str:
    if asset_id in wide.columns:
        return asset_id
    lower = {c.lower(): c for c in wide.columns}
    if asset_id.lower() in lower:
        return lower[asset_id.lower()]
    raise KeyError(f"asset {asset_id!r} not found in panel columns {list(wide.columns)}")


def fit_and_sample(panel: pd.DataFrame, card: dict, asof: str, n_draws: int) -> np.ndarray:
    targets = card["targets"]
    asset_ids = list(targets["asset_ids"])
    horizons = list(targets["horizons"])
    target_type = targets["target_type"]

    # Coerce both sides to pandas Timestamp before comparing.
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    asof_ts = pd.Timestamp(asof)
    panel = panel[panel["date"] <= asof_ts]

    wide = panel.pivot_table(index="date", columns="asset", values="value").sort_index()

    n_assets = len(asset_ids)
    n_horizons = len(horizons)
    draws = np.zeros((n_draws, n_assets, n_horizons), dtype=float)

    rng = np.random.default_rng(seed=42)

    for ai, asset in enumerate(asset_ids):
        col = _find_column(wide, asset)
        series = wide[col].dropna()
        if len(series) < 2:
            draws[:, ai, :] = 0.0
            continue

        if target_type == "level":
            last = float(series.iloc[-1])
            daily_diff = series.diff().dropna()
            sigma_d = float(daily_diff.std()) if len(daily_diff) > 1 else 0.0
            for hi, h in enumerate(horizons):
                draws[:, ai, hi] = last + rng.normal(0.0, sigma_d * np.sqrt(h), n_draws)

        elif target_type == "log_return":
            r = series.astype(float).to_numpy()
            log_r = np.log1p(r)
            mu = float(np.mean(log_r))
            sigma = float(np.std(log_r))
            for hi, h in enumerate(horizons):
                draws[:, ai, hi] = rng.normal(mu * h, sigma * np.sqrt(h), n_draws)

        else:
            raise ValueError(f"unknown target_type {target_type!r}")

    return draws