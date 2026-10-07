"""Executive summary: fit a small ARIMA candidate set and sample forecasts.

The model uses only the numerical panel through ``asof``. It returns Monte Carlo
draws in target order and raises when fitting cannot produce a usable forecast,
allowing the caller to try ETS and random-walk fallbacks.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


ARIMA_ORDERS = (
    (1, 0, 1),
    (1, 1, 1),
    (2, 0, 1),
    (2, 1, 1),
    (1, 0, 0),
    (0, 1, 1),
)
MIN_HISTORY = 20


def _find_column(wide: pd.DataFrame, asset_id: str) -> str:
    """Find an asset column, allowing case differences in authored IDs."""
    if asset_id in wide.columns:
        return asset_id
    by_lower = {str(column).lower(): column for column in wide.columns}
    match = by_lower.get(asset_id.lower())
    if match is None:
        raise KeyError(f"asset {asset_id!r} not present in panel columns {list(wide.columns)}")
    return match


def _fit_arima(values: np.ndarray):
    """Fit candidates and return the converged model with lowest finite AIC."""
    best_model = None
    best_aic = np.inf
    failures = []

    for order in ARIMA_ORDERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(values, order=order).fit()
            converged = getattr(fitted, "mle_retvals", {}).get("converged", True)
            if converged and np.isfinite(fitted.aic) and fitted.aic < best_aic:
                best_model = fitted
                best_aic = float(fitted.aic)
        except Exception as exc:  # Candidate failures should not prevent trying others.
            failures.append(f"{order}: {exc}")

    if best_model is None:
        detail = "; ".join(failures[-2:]) or "all fits failed or did not converge"
        raise RuntimeError(f"no ARIMA candidate produced a converged finite-AIC fit ({detail})")
    return best_model


def _horizon_steps(
    panel: pd.DataFrame,
    card: dict,
    asof: str,
    asset_ids: list[str],
    horizons: list[int],
) -> np.ndarray:
    """Resolve model steps while preserving authored horizon keys for output."""
    targets = card["targets"]
    frequency = str(
        targets.get("target_frequency", card.get("metadata", {}).get("target_frequency", "daily"))
    ).lower()
    if frequency not in {"monthly", "month", "monthly_level"}:
        if any(h <= 0 for h in horizons):
            raise ValueError("daily forecast horizons must be positive step counts")
        return np.tile(np.asarray(horizons, dtype=int), (len(asset_ids), 1))

    if targets.get("target_type") != "level":
        raise ValueError("monthly horizon mapping is supported only for level targets")

    histories = {}
    last_observations = {}
    for asset in asset_ids:
        series = panel.loc[panel["asset"].astype(str).str.lower() == asset.lower()]
        dates = pd.DatetimeIndex(pd.to_datetime(series["date"]).drop_duplicates().sort_values())
        if dates.empty:
            raise ValueError(f"no monthly observations available for target asset {asset!r}")
        periods = dates.to_period("M")
        if periods.has_duplicates:
            raise ValueError(f"target asset {asset!r} has more than one observation per month")
        histories[asset] = periods
        last_observations[asset] = dates[-1].date().isoformat()

    try:
        from qfbench2_track_forecasting.horizons import monthly_horizon_steps
    except ImportError as exc:
        raise RuntimeError(
            "monthly target forecasting requires "
            "qfbench2_track_forecasting.horizons.monthly_horizon_steps"
        ) from exc

    return monthly_horizon_steps(
        asset_ids,
        horizons,
        last_observations,
        asof=asof,
        card=card,
        forecast_spec=card.get("_forecast_spec"),
    )


def fit_and_sample(
    panel: pd.DataFrame,
    card: dict,
    asof: str,
    n_draws: int,
    seed: int = 42,
) -> np.ndarray:
    """Fit per-asset ARIMA models and return draws shaped ``[draw, asset, horizon]``.

    For ``log_return`` targets, panel simple returns are transformed with
    ``log1p`` and forecast daily log-return draws are accumulated to each horizon.
    For ``level`` targets, ARIMA is fit directly to the level series. Forecast
    errors are sampled as Gaussian and independent across forecast steps; later
    distribution/joint layers can add richer tails and dependence.
    """
    targets = card["targets"]
    asset_ids = [str(asset) for asset in targets["asset_ids"]]
    horizons = [int(horizon) for horizon in targets["horizons"]]
    target_type = targets["target_type"]
    if target_type not in {"level", "log_return"}:
        raise ValueError(f"unsupported target_type {target_type!r}")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    if not asset_ids or not horizons:
        raise ValueError("card targets must declare assets and horizons")

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date", "asset", "value"])
    data = data.loc[data["date"] <= pd.Timestamp(asof)]
    if data.empty:
        raise ValueError(f"panel contains no observations on or before {asof}")

    steps = _horizon_steps(data, card, asof, asset_ids, horizons)
    if steps.shape != (len(asset_ids), len(horizons)) or np.any(steps <= 0):
        raise ValueError(f"invalid resolved forecast steps: {steps}")
    max_steps = int(np.max(steps))

    wide = data.pivot_table(index="date", columns="asset", values="value").sort_index()
    draws = np.empty((n_draws, len(asset_ids), len(horizons)), dtype=np.float64)
    rngs = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(len(asset_ids))
    ]

    for asset_index, asset in enumerate(asset_ids):
        column = _find_column(wide, asset)
        series = wide[column].dropna().astype(float)
        if len(series) < MIN_HISTORY:
            raise ValueError(
                f"asset {asset!r} has {len(series)} observations; ARIMA requires "
                f"at least {MIN_HISTORY}"
            )

        raw = series.to_numpy(dtype=np.float64)
        if target_type == "log_return":
            if np.any(raw <= -1.0):
                raise ValueError(f"asset {asset!r} contains returns <= -100%, invalid for log1p")
            values = np.log1p(raw)
        else:
            values = raw

        model = _fit_arima(values)
        forecast = model.get_forecast(steps=max_steps)
        mean = np.asarray(forecast.predicted_mean, dtype=np.float64)
        standard_error = np.asarray(forecast.se_mean, dtype=np.float64)
        if mean.shape != (max_steps,) or standard_error.shape != (max_steps,):
            raise RuntimeError(f"ARIMA returned unexpected forecast shape for {asset!r}")
        if not np.isfinite(mean).all() or not np.isfinite(standard_error).all():
            raise RuntimeError(f"ARIMA returned non-finite forecasts for {asset!r}")

        paths = mean[None, :] + rngs[asset_index].standard_normal(
            (n_draws, max_steps)
        ) * standard_error[None, :]
        for horizon_index, step in enumerate(steps[asset_index]):
            step = int(step)
            if target_type == "log_return":
                draws[:, asset_index, horizon_index] = paths[:, :step].sum(axis=1)
            else:
                draws[:, asset_index, horizon_index] = paths[:, step - 1]

    return draws
