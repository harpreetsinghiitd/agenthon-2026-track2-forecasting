"""Executive summary: forecast F1 cards from weighted historical analog paths.

The forecaster compares the latest numerical state with earlier states whose
complete future paths are already observed. It samples those paths as Monte
Carlo draws and never reads text or observations after ``asof``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .arima import _horizon_steps


FEATURE_WINDOWS = (1, 5, 21, 63)
MIN_ANALOGS = 20


def _state_features(values: np.ndarray, target_type: str,
                    value_unit: str) -> np.ndarray:
    """Build scale-neutral trend, change, volatility, and level-state features."""
    n_rows, n_assets = values.shape
    columns = []
    is_fx_level = target_type == "level" and "_per_" in value_unit

    for asset_index in range(n_assets):
        series = values[:, asset_index]
        if target_type == "log_return":
            if np.any(series <= -1.0):
                raise ValueError("analog return history contains values <= -100%")
            transformed = np.log1p(series)
            changes = transformed
        elif is_fx_level:
            if np.any(series <= 0.0):
                raise ValueError("analog FX level history must be positive")
            transformed = np.log(series)
            changes = np.diff(transformed, prepend=np.nan)
        elif target_type == "level":
            transformed = series
            changes = np.diff(transformed, prepend=np.nan)
        else:
            raise ValueError(f"unsupported target_type {target_type!r}")

        change_series = pd.Series(changes)
        for window in FEATURE_WINDOWS:
            columns.append(change_series.rolling(window, min_periods=window).sum().to_numpy())
        for window in (21, 63):
            columns.append(change_series.rolling(window, min_periods=window).std().to_numpy())

        if target_type == "level":
            level_series = pd.Series(transformed)
            rolling_mean = level_series.rolling(63, min_periods=63).mean()
            rolling_sd = level_series.rolling(63, min_periods=63).std().replace(0.0, np.nan)
            columns.append(((level_series - rolling_mean) / rolling_sd).to_numpy())

    return np.column_stack(columns).reshape(n_rows, -1)


def _robust_scale(candidate_features: np.ndarray,
                  current_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robustly scale state features using only states observed by the cutoff."""
    center = np.nanmedian(candidate_features, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(candidate_features - center), axis=0)
    fallback = np.nanstd(candidate_features, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-10), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-10), scale, 1.0)
    return (candidate_features - center) / scale, (current_features - center) / scale


def _analogue_weights(distances: np.ndarray, n_analogs: int) -> tuple[np.ndarray, np.ndarray]:
    """Select nearest donors and temper weights to avoid a single-path forecast."""
    count = min(int(n_analogs), len(distances))
    nearest = np.argsort(distances, kind="stable")[:count]
    local_distances = distances[nearest]
    if not np.any(local_distances > 1e-12):
        return nearest, np.full(count, 1.0 / count)

    temperature = max(float(np.quantile(local_distances, 0.75)), 1e-6)
    for _ in range(8):
        weights = np.exp(-local_distances / temperature)
        weights /= weights.sum()
        effective_n = 1.0 / np.square(weights).sum()
        if effective_n >= min(10, count):
            return nearest, weights
        temperature *= 2.0
    return nearest, np.full(count, 1.0 / count)


def fit_and_sample(
    panel: pd.DataFrame,
    card: dict,
    asof: str,
    n_draws: int,
    seed: int = 48,
    n_analogs: int = 100,
    min_analogs: int = MIN_ANALOGS,
) -> np.ndarray:
    """Return weighted historical-analogue draws shaped ``[draw, asset, horizon]``.

    Analogue donors are spaced apart to limit clusters of overlapping future
    windows. Their future changes are anchored at the current level, while
    log-return targets are returned as cumulative future log returns.
    """
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    targets = card["targets"]
    assets = [str(asset) for asset in targets["asset_ids"]]
    horizons = [int(horizon) for horizon in targets["horizons"]]
    target_type = str(targets["target_type"])
    value_unit = str(targets.get("value_unit", "")).lower()
    if not assets or not horizons:
        raise ValueError("card targets must declare assets and horizons")

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date", "asset", "value"])
    data = data.loc[data["date"] <= pd.Timestamp(asof)]
    target_rows = data.loc[data["asset"].astype(str).str.lower().isin(
        {asset.lower() for asset in assets}
    )]
    if target_rows.empty:
        raise ValueError("panel has no observations for the target assets")

    wide_all = target_rows.pivot_table(
        index="date", columns="asset", values="value", aggfunc="last"
    ).sort_index()
    column_lookup = {str(column).lower(): column for column in wide_all.columns}
    missing = [asset for asset in assets if asset.lower() not in column_lookup]
    if missing:
        raise ValueError(f"target assets missing from panel: {missing}")
    wide_all = wide_all[[column_lookup[asset.lower()] for asset in assets]]
    last_dates = [wide_all[column].last_valid_index() for column in wide_all.columns]
    if len(set(last_dates)) != 1:
        raise ValueError("target assets do not share a common latest observation date")
    wide = wide_all.dropna(axis=0, how="any")
    if wide.empty or wide.index[-1] != last_dates[0]:
        raise ValueError("target assets do not have an aligned latest observation")

    steps = _horizon_steps(data, card, asof, assets, horizons)
    if steps.shape != (len(assets), len(horizons)) or np.any(steps <= 0):
        raise ValueError(f"invalid resolved forecast steps: {steps}")
    max_steps = int(np.max(steps))
    values = wide.to_numpy(dtype=np.float64)
    n_rows = len(values)
    feature_history = max(FEATURE_WINDOWS)
    if n_rows <= feature_history + max_steps:
        raise ValueError("not enough history for analogue features and a complete donor path")

    all_features = _state_features(values, target_type, value_unit)
    stride = max(1, min(21, max_steps // 5))
    candidate_indices = np.arange(feature_history - 1, n_rows - max_steps, stride)
    candidate_features = all_features[candidate_indices]
    current_features = all_features[-1:]
    valid = np.isfinite(candidate_features).all(axis=1)
    candidate_indices = candidate_indices[valid]
    candidate_features = candidate_features[valid]
    if len(candidate_indices) < min_analogs:
        raise ValueError(
            f"only {len(candidate_indices)} complete analogues; need at least {min_analogs}"
        )
    if not np.isfinite(current_features).all():
        raise ValueError("current state lacks enough history for analogue features")

    scaled_candidates, scaled_current = _robust_scale(candidate_features, current_features)
    distances = np.sqrt(np.mean(np.square(scaled_candidates - scaled_current), axis=1))
    nearest, weights = _analogue_weights(distances, n_analogs)
    donors = candidate_indices[nearest]
    rng = np.random.default_rng(seed)
    selected = donors[rng.choice(len(donors), size=n_draws, replace=True, p=weights)]

    draws = np.empty((n_draws, len(assets), len(horizons)), dtype=np.float64)
    if target_type == "log_return":
        if np.any(values <= -1.0):
            raise ValueError("analog return history contains values <= -100%")
        cumulative = np.vstack([
            np.zeros((1, len(assets))), np.cumsum(np.log1p(values), axis=0)
        ])
        for ai in range(len(assets)):
            for hi, step in enumerate(steps[ai]):
                end = selected + int(step)
                draws[:, ai, hi] = cumulative[end + 1, ai] - cumulative[selected + 1, ai]
    elif target_type == "level" and "_per_" in value_unit:
        if np.any(values <= 0.0):
            raise ValueError("positive FX level analogues require strictly positive values")
        log_values = np.log(values)
        for ai in range(len(assets)):
            anchor = values[-1, ai]
            for hi, step in enumerate(steps[ai]):
                end = selected + int(step)
                draws[:, ai, hi] = anchor * np.exp(log_values[end, ai] - log_values[selected, ai])
    elif target_type == "level":
        for ai in range(len(assets)):
            anchor = values[-1, ai]
            for hi, step in enumerate(steps[ai]):
                end = selected + int(step)
                draws[:, ai, hi] = anchor + values[end, ai] - values[selected, ai]
    else:
        raise ValueError(f"unsupported target_type {target_type!r}")

    if not np.isfinite(draws).all():
        raise ValueError("analogue forecaster produced non-finite draws")
    return draws
