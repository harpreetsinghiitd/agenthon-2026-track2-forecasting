"""ARIMA on log changes for positive level series, reconstructed as levels."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .arima import _find_column, _horizon_steps


def fit_and_sample(panel: pd.DataFrame, card: dict, asof: str, n_draws: int,
                   seed: int = 47) -> np.ndarray:
    """Fit ARMA candidates to log-level differences and sample level paths.

    This candidate applies only to positive ``level`` targets. Each simulated
    log-change path is cumulatively added to the last observed log level before
    exponentiation, preserving horizon dependence and positive support.
    """
    targets = card["targets"]
    if targets["target_type"] != "level":
        raise ValueError("log-change ARIMA is only defined for level targets")
    assets = [str(x) for x in targets["asset_ids"]]
    horizons = [int(x) for x in targets["horizons"]]
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date", "asset", "value"])
    data = data.loc[data["date"] <= pd.Timestamp(asof)]
    steps = _horizon_steps(data, card, asof, assets, horizons)
    max_steps = int(np.max(steps))
    wide = data.pivot_table(index="date", columns="asset", values="value").sort_index()
    draws = np.empty((n_draws, len(assets), len(horizons)), dtype=np.float64)
    rngs = [np.random.default_rng(child) for child in
            np.random.SeedSequence(seed).spawn(len(assets))]
    orders = ((1, 0, 1), (1, 0, 0), (0, 0, 1), (0, 0, 0))

    for ai, asset in enumerate(assets):
        series = wide[_find_column(wide, asset)].dropna().astype(float).to_numpy()
        if len(series) < 21:
            raise ValueError(f"asset {asset!r} needs at least 21 level observations")
        if np.any(series <= 0):
            raise ValueError(f"asset {asset!r} contains nonpositive levels")
        log_levels = np.log(series)
        changes = np.diff(log_levels)
        best, best_aic = None, np.inf
        failures = []
        for order in orders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(changes, order=order, trend="c").fit()
                converged = getattr(fitted, "mle_retvals", {}).get("converged", True)
                if converged and np.isfinite(fitted.aic) and fitted.aic < best_aic:
                    best, best_aic = fitted, float(fitted.aic)
            except Exception as exc:
                failures.append(str(exc))
        if best is None:
            raise RuntimeError(f"no log-change ARIMA fit converged for {asset}: {failures[-1:]}")
        simulated = np.asarray(best.simulate(
            nsimulations=max_steps, anchor="end", repetitions=n_draws,
            rng=rngs[ai],
        ), dtype=np.float64)
        if simulated.shape == (max_steps, 1, n_draws):
            change_paths = simulated[:, 0, :].T
        elif simulated.shape == (max_steps, n_draws):
            change_paths = simulated.T
        else:
            raise RuntimeError(f"unexpected simulated log-change path shape {simulated.shape}")
        level_paths = np.exp(log_levels[-1] + np.cumsum(change_paths, axis=1))
        if not np.isfinite(level_paths).all():
            raise RuntimeError(f"nonfinite reconstructed level paths for {asset!r}")
        for hi, step in enumerate(steps[ai]):
            draws[:, ai, hi] = level_paths[:, int(step) - 1]
    return draws
