"""Numerical forecaster — Tier 1 (ARIMA per asset).

Input:  panel DataFrame with columns [date, asset, value, panel_id]
        card dict with [targets] asset_ids, horizons, target_type
        asof date string "YYYY-MM-DD"
        n_draws integer

Output: np.ndarray of shape [n_draws, n_assets, n_horizons]

Never reads the text corpus. Pure numerical function.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")  # statsmodels prints convergence noise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_column(wide: pd.DataFrame, asset_id: str) -> str:
    """Match asset_id to a column name, case-insensitively if needed."""
    if asset_id in wide.columns:
        return asset_id
    lower = {c.lower(): c for c in wide.columns}
    if asset_id.lower() in lower:
        return lower[asset_id.lower()]
    raise KeyError(f"asset {asset_id!r} not in panel columns {list(wide.columns)}")


def _fit_arima(series: np.ndarray) -> ARIMA:
    """Fit ARIMA with a small order search, picking lowest AIC.

    Tries a handful of common orders. If all fail, raises.
    """
    candidates = [
        (1, 0, 1),
        (1, 1, 1),
        (2, 0, 1),
        (2, 1, 1),
        (1, 0, 0),
        (0, 1, 1),
    ]
    best_model = None
    best_aic = np.inf

    for order in candidates:
        try:
            model = ARIMA(series, order=order).fit()
            if np.isfinite(model.aic) and model.aic < best_aic:
                best_aic = model.aic
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("all ARIMA orders failed to fit")
    return best_model


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_and_sample(panel: pd.DataFrame, card: dict, asof: str, n_draws: int) -> np.ndarray:
    targets = card["targets"]
    asset_ids = list(targets["asset_ids"])
    horizons = list(targets["horizons"])
    target_type = targets["target_type"]
    max_h = max(horizons)

    # --- Filter and pivot ---
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    asof_ts = pd.Timestamp(asof)
    panel = panel[panel["date"] <= asof_ts]

    wide = (
        panel.pivot_table(index="date", columns="asset", values="value")
        .sort_index()
    )

    n_assets = len(asset_ids)
    n_horizons = len(horizons)
    draws = np.zeros((n_draws, n_assets, n_horizons), dtype=float)
    rng = np.random.default_rng(seed=42)

    # --- Per-asset loop ---
    for ai, asset in enumerate(asset_ids):
        try:
            col = _find_column(wide, asset)
        except KeyError:
            # Asset missing from panel — fill with zeros
            draws[:, ai, :] = 0.0
            continue

        series = wide[col].dropna()

        if len(series) < 20:
            draws[:, ai, :] = 0.0
            continue

        # --- Prepare the modelling target ---
        if target_type == "log_return":
            # Panel values are decimal daily simple returns.
            # Target is cumulative log return: sum(log(1 + r)).
            r = series.to_numpy(dtype=float)
            y = np.log1p(r)
            cumulate = True
        elif target_type == "level":
            # Target is the level itself at each horizon step.
            y = series.to_numpy(dtype=float)
            cumulate = False
        else:
            raise ValueError(f"unknown target_type {target_type!r}")

        # --- Fit ARIMA ---
        try:
            model = _fit_arima(y)
        except Exception:
            # Fit failed — fall back to historical mean and std
            mu = np.full(max_h, np.mean(y))
            se = np.full(max_h, np.std(y))
            eps = rng.standard_normal((n_draws, max_h))
            paths = mu[None, :] + se[None, :] * eps
            for hi, h in enumerate(horizons):
                if cumulate:
                    draws[:, ai, hi] = paths[:, :h].sum(axis=1)
                else:
                    draws[:, ai, hi] = paths[:, h - 1]
            continue

        # --- Forecast mean and std error at each step ---
        fc = model.get_forecast(steps=max_h)
        mu = np.asarray(fc.predicted_mean, dtype=float)
        se = np.asarray(fc.se_mean, dtype=float)

        # --- Draw paths: [n_draws, max_h] ---
        eps = rng.standard_normal((n_draws, max_h))
        paths = mu[None, :] + se[None, :] * eps

        # --- Extract requested horizons ---
        for hi, h in enumerate(horizons):
            if cumulate:
                # Cumulative log return over h days
                draws[:, ai, hi] = paths[:, :h].sum(axis=1)
            else:
                # Level at step h
                draws[:, ai, hi] = paths[:, h - 1]

    return draws
