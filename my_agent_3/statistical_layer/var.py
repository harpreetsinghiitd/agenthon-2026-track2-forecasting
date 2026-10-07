"""Executive summary: VAR model for cross-asset mean dynamics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from .arima import _horizon_steps


def fit_and_sample(panel: pd.DataFrame, card: dict, asof: str, n_draws: int,
                   seed: int = 45) -> np.ndarray:
    """Fit a VAR to aligned target series and sample correlated forecast paths."""
    targets = card["targets"]
    assets = [str(x) for x in targets["asset_ids"]]
    horizons = [int(x) for x in targets["horizons"]]
    target_type = targets["target_type"]
    if len(assets) < 2:
        raise ValueError("VAR requires at least two target assets")
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.loc[data["date"] <= pd.Timestamp(asof)]
    wide = data.pivot_table(index="date", columns="asset", values="value").sort_index()
    cols = {str(c).lower(): c for c in wide.columns}
    try:
        wide = wide[[cols[a.lower()] for a in assets]].dropna()
    except KeyError as exc:
        raise ValueError(f"VAR target asset missing from panel: {exc}") from exc
    if target_type == "log_return":
        if (wide.to_numpy() <= -1).any():
            raise ValueError("VAR log-return panel contains values <= -100%")
        model_data = np.log1p(wide)
        cumulative = True
    elif target_type == "level":
        model_data = wide.diff().dropna()
        cumulative = False
    else:
        raise ValueError(f"unsupported target_type {target_type!r}")
    if len(model_data) < max(25, len(assets) + 5):
        raise ValueError("too little aligned history for VAR")
    fitted = VAR(model_data).fit(maxlags=min(5, max(1, len(model_data) // 10)), ic="aic")
    if fitted.k_ar < 1:
        raise ValueError("VAR selected zero lags")
    # Daily VAR sampling is supported here. Monthly target mappings are resolved by
    # the shared helper, but a monthly multivariate simulator is not yet implemented.
    steps = _horizon_steps(data, card, asof, assets, horizons)
    if not np.all(steps == steps[0:1, :]):
        raise ValueError("VAR currently requires the same resolved horizons for all assets")
    hsteps = steps[0]
    max_steps = int(hsteps.max())
    residuals = np.asarray(fitted.resid, dtype=float)
    cov = np.cov(residuals, rowvar=False)
    cov = np.atleast_2d(cov) + np.eye(len(assets)) * 1e-10
    rng = np.random.default_rng(seed)
    innovations = rng.multivariate_normal(np.zeros(len(assets)), cov,
                                          size=(n_draws, max_steps))
    paths = np.empty((n_draws, max_steps, len(assets)), dtype=float)
    lag_state = np.repeat(model_data.to_numpy()[-fitted.k_ar:][None, :, :], n_draws, axis=0)
    coefs = np.asarray(fitted.coefs, dtype=float)
    intercept = np.asarray(fitted.intercept, dtype=float)
    for t in range(max_steps):
        pred = np.repeat(intercept[None, :], n_draws, axis=0)
        for lag in range(1, fitted.k_ar + 1):
            pred += lag_state[:, -lag, :] @ coefs[lag - 1].T
        pred += innovations[:, t, :]
        paths[:, t, :] = pred
        lag_state = np.concatenate([lag_state[:, 1:, :], pred[:, None, :]], axis=1)
    draws = np.empty((n_draws, len(assets), len(horizons)), dtype=float)
    for hi, step in enumerate(hsteps):
        step = int(step)
        if cumulative:
            selected = paths[:, :step, :].sum(axis=1)
        else:
            selected = wide.to_numpy()[-1][None, :] + paths[:, :step, :].sum(axis=1)
        draws[:, :, hi] = selected
    return draws
