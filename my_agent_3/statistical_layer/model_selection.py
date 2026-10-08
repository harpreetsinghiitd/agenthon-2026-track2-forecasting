"""Executive summary: choose an F1 forecaster using past-only rolling validation.

For each card, candidate models are evaluated at recent historical origins whose
complete target windows are already observed by the requested as-of date. The
selected model is then refit on all observations available at the real as-of.
"""

from __future__ import annotations

import math
import copy
from functools import partial

import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps
from qfbench2_track_forecasting.tail import tail_pinball

from . import analog, arima, block_bootstrap, ets, random_walk


DEFAULT_VALIDATION_ORIGINS = 4
MIN_VALIDATION_ORIGINS = 2
MIN_VALIDATION_HISTORY = 100
TAIL_LEVELS = (0.01, 0.05, 0.95, 0.99)

MIN_CHALLENGER_GAIN = 0.10
MIN_CHALLENGER_WIN_RATE = 0.60
MAX_CHALLENGER_WEIGHT = 0.25
LOCAL_DRIFT_ANCHOR_GAIN = 0.05


def _model_candidates(card: dict, local_drift_window: int = 252,
                     local_drift_shrink: float = 0.5):
    """Return a target-aware candidate set with explicit ARIMA alternatives."""
    candidates = {
        "ARIMA": arima.fit_and_sample,
        "ARIMA residual blocks": partial(
            arima.fit_and_sample, innovation="residual_block", block_length=5,
        ),
        "ETS": ets.fit_and_sample,
        "analog": analog.fit_and_sample,
        "local drift RW": partial(
            random_walk.fit_and_sample, drift_window=local_drift_window,
            drift_shrink=local_drift_shrink,
        ),
        "local drift RW (126, full drift)": partial(
            random_walk.fit_and_sample, drift_window=126, drift_shrink=0.0,
        ),
        "local drift RW (63, shrunk)": partial(
            random_walk.fit_and_sample, drift_window=63, drift_shrink=1.0,
        ),
    }
    if card["targets"]["target_type"] == "log_return":
        candidates.update({
            "ARIMA(1,0,0)": partial(arima.fit_and_sample, order=(1, 0, 0)),
            "block bootstrap 5": partial(block_bootstrap.fit_and_sample, block_length=5),
            "block bootstrap 20": partial(block_bootstrap.fit_and_sample, block_length=20),
        })
    else:
        candidates.update({
            "ARIMA(0,1,1) drift": partial(
                arima.fit_and_sample, order=(0, 1, 1), trend="t",
            ),
            "ARIMA(1,1,0) drift": partial(
                arima.fit_and_sample, order=(1, 1, 0), trend="t",
            ),
        })
    return candidates


def _target_at(panel: pd.DataFrame, card: dict, origin: pd.Timestamp,
               steps: np.ndarray) -> np.ndarray:
    """Read card targets from data immediately after an inner validation origin."""
    assets = [str(asset) for asset in card["targets"]["asset_ids"]]
    horizons = [int(h) for h in card["targets"]["horizons"]]
    target_type = str(card["targets"]["target_type"])
    y = np.empty((len(assets), len(horizons)), dtype=np.float64)

    for asset_index, asset in enumerate(assets):
        series = panel.loc[
            panel["asset"].astype(str).str.lower() == asset.lower()
        ].sort_values("date")
        future = series.loc[series["date"] > origin, "value"].to_numpy(dtype=float)
        for horizon_index, _ in enumerate(horizons):
            step = int(steps[asset_index, horizon_index])
            if step <= 0 or len(future) < step:
                raise ValueError(f"not enough future observations for {asset!r}, step {step}")
            if target_type == "log_return":
                path = future[:step]
                if np.any(path <= -1.0):
                    raise ValueError("validation returns contain values <= -100%")
                y[asset_index, horizon_index] = np.log1p(path).sum()
            elif target_type == "level":
                y[asset_index, horizon_index] = future[step - 1]
            else:
                raise ValueError(f"unsupported target_type {target_type!r}")
    return y


def _validation_origins(panel: pd.DataFrame, card: dict, asof: str,
                        n_origins: int) -> list[tuple[pd.Timestamp, np.ndarray, pd.DataFrame, np.ndarray]]:
    """Build recent, non-overlapping pseudo-origins using only data through asof."""
    from .arima import _horizon_steps

    targets = card["targets"]
    assets = [str(asset) for asset in targets["asset_ids"]]
    horizons = [int(h) for h in targets["horizons"]]
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date", "asset", "value"])
    data = data.loc[data["date"] <= pd.Timestamp(asof)].sort_values("date")
    target_data = data.loc[data["asset"].astype(str).str.lower().isin(
        {asset.lower() for asset in assets}
    )]
    dates = pd.DatetimeIndex(target_data["date"].drop_duplicates().sort_values())
    if len(dates) <= MIN_VALIDATION_HISTORY:
        return []

    valid = []
    # Search backward from the latest origin whose full target path is known.
    # This bounds validation work even for long daily histories.
    latest_position = len(dates) - 1 - max(horizons)
    earliest_position = MIN_VALIDATION_HISTORY - 1
    if latest_position < earliest_position:
        return []
    positions = []
    position = latest_position
    spacing = max(1, max(horizons))
    attempts = max(20, n_origins * 8)
    for _ in range(attempts):
        if position < earliest_position:
            break
        positions.append(position)
        position -= spacing

    # Potential origins are constrained to have enough training history and a
    # complete realized target path before the actual as-of cutoff.
    for date_position in positions:
        origin = dates[date_position]
        train = target_data.loc[target_data["date"] <= origin].copy()
        try:
            steps = _horizon_steps(train, card, origin.date().isoformat(), assets, horizons)
            if np.any(steps <= 0):
                continue
            future_counts = [
                int((target_data.loc[
                    (target_data["asset"].astype(str).str.lower() == asset.lower())
                    & (target_data["date"] > origin)
                ].groupby("date").size().shape[0]))
                for asset in assets
            ]
            if any(future_counts[i] < int(steps[i].max()) for i in range(len(assets))):
                continue
            y = _target_at(target_data, card, origin, steps)
            valid.append((origin, steps, train, y))
        except Exception:
            continue

    return sorted(valid, key=lambda item: item[0])[-n_origins:]


def _components(draws: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    samples = np.asarray(draws, dtype=np.float64).reshape(draws.shape[0], -1)
    truth = np.asarray(y, dtype=np.float64).reshape(-1)
    return (
        float(crps.crps_marginal(samples, truth)),
        float(crps.variogram_score(samples, truth, p=0.5)),
        float(tail_pinball(samples, truth, TAIL_LEVELS)),
    )


def _weights(card: dict) -> tuple[float, float, float]:
    scoring = card.get("scoring", {})
    params = scoring.get("params", {}) if isinstance(scoring, dict) else {}
    configured = params.get("weights", {}) if isinstance(params, dict) else {}
    marginal = float(configured.get("marginal", 0.5))
    joint = float(configured.get("joint", 0.3))
    tail = float(configured.get("tail", 0.2))
    n_cells = len(card["targets"]["asset_ids"]) * len(card["targets"]["horizons"])
    if n_cells == 1:
        total = marginal + tail
        return marginal / total, 0.0, tail / total
    total = marginal + joint + tail
    return marginal / total, joint / total, tail / total


def _relative_score(draws: np.ndarray, y: np.ndarray,
                    reference: tuple[float, float, float],
                    weights: tuple[float, float, float]) -> float:
    losses = _components(draws, y)
    normalized = []
    for loss, baseline in zip(losses, reference):
        # A tiny floor prevents division by zero on constant validation targets.
        floor = max(np.finfo(float).eps, abs(float(baseline)) * 1e-8)
        normalized.append(loss / max(abs(float(baseline)), floor))
    return float(sum(w * value for w, value in zip(weights, normalized)))


def select_and_sample(panel: pd.DataFrame, card: dict, asof: str, n_draws: int,
                      seed: int = 42,
                      n_origins: int = DEFAULT_VALIDATION_ORIGINS,
                      candidate_subset: tuple[str, ...] | None = None,
                      return_full_path: bool = False,
                      local_drift_window: int = 252,
                      local_drift_shrink: float = 0.5,
                      ) -> tuple[np.ndarray, str, str]:
    """Build a conservative ARIMA/local-drift-centered mixture from past-only validation."""
    folds = _validation_origins(panel, card, asof, n_origins)
    fit_card = card
    if return_full_path:
        frequency = str(card["targets"].get(
            "target_frequency", card.get("metadata", {}).get("target_frequency", "daily")
        )).lower()
        if frequency not in {"daily", "business_daily", "daily_level", ""}:
            raise ValueError("full path output is currently supported for daily targets")
        fit_card = copy.deepcopy(card)
        fit_card["targets"]["horizons"] = list(
            range(1, max(int(h) for h in card["targets"]["horizons"]) + 1)
        )
    expected_horizons = len(fit_card["targets"]["horizons"])
    weights = _weights(card)
    if len(folds) < MIN_VALIDATION_ORIGINS:
        draws = arima.fit_and_sample(panel, fit_card, asof, n_draws, seed=seed)
        note = f"model selection unavailable ({len(folds)} valid origins); ARIMA default"
        return draws, "ARIMA", note

    candidates = _model_candidates(card, local_drift_window=local_drift_window,
                                   local_drift_shrink=local_drift_shrink)
    if candidate_subset is not None:
        missing = set(candidate_subset).difference(candidates)
        if missing:
            raise ValueError(f"unknown model-selection candidates: {sorted(missing)}")
        candidates = {name: candidates[name] for name in candidate_subset}
    fold_scores: dict[str, dict[int, float]] = {name: {} for name in candidates}
    for fold_index, (origin, _steps, train, y) in enumerate(folds):
        cutoff = origin.date().isoformat()
        fold_seed = seed + 1009 * (fold_index + 1)
        try:
            reference_draws = random_walk.fit_and_sample(
                train, card, cutoff, n_draws, seed=fold_seed
            )
            reference = _components(reference_draws, y)
        except Exception:
            continue

        for model_index, (name, model) in enumerate(candidates.items()):
            try:
                draws = model(train, card, cutoff, n_draws,
                              seed=fold_seed + model_index + 1)
                expected = (n_draws, len(card["targets"]["asset_ids"]),
                            len(card["targets"]["horizons"]))
                if draws.shape != expected or not np.isfinite(draws).all():
                    raise ValueError(f"invalid draws shape or values: {draws.shape}")
                fold_scores[name][fold_index] = _relative_score(draws, y, reference, weights)
            except Exception:
                continue

    min_folds = max(MIN_VALIDATION_ORIGINS, math.ceil(len(folds) / 2))
    means = {
        name: float(np.mean(list(scores.values())))
        for name, scores in fold_scores.items() if len(scores) >= min_folds
    }
    if not means:
        means = {"ARIMA": float("inf")}

    mixture_weights: dict[str, float] = {}
    anchor = "ARIMA"
    if "ARIMA" in means and np.isfinite(means["ARIMA"]):
        # Promote local drift to the anchor only when its paired inner-fold
        # improvement is material and consistent; otherwise retain ARIMA.
        drift_name = "local drift RW"
        if drift_name in means:
            common = set(fold_scores["ARIMA"]).intersection(fold_scores[drift_name])
            if len(common) >= min_folds:
                arima_values = np.array([fold_scores["ARIMA"][i] for i in sorted(common)])
                drift_values = np.array([fold_scores[drift_name][i] for i in sorted(common)])
                drift_gain = float((np.mean(arima_values) - np.mean(drift_values)) /
                                   max(abs(float(np.mean(arima_values))), 1e-8))
                drift_win_rate = float(np.mean(drift_values < arima_values))
                if drift_gain >= LOCAL_DRIFT_ANCHOR_GAIN and drift_win_rate >= MIN_CHALLENGER_WIN_RATE:
                    anchor = drift_name
        anchor_folds = fold_scores[anchor]
        challengers = []
        for name, mean_score in means.items():
            if name == anchor:
                continue
            common = set(anchor_folds).intersection(fold_scores[name])
            if len(common) < min_folds:
                continue
            paired_anchor = np.array([anchor_folds[i] for i in sorted(common)])
            paired_candidate = np.array([fold_scores[name][i] for i in sorted(common)])
            gain = float((np.mean(paired_anchor) - np.mean(paired_candidate))
                         / max(abs(float(np.mean(paired_anchor))), 1e-8))
            win_rate = float(np.mean(paired_candidate < paired_anchor))
            if gain >= MIN_CHALLENGER_GAIN and win_rate >= MIN_CHALLENGER_WIN_RATE:
                challengers.append((name, gain))

        if challengers:
            challenger_mass = min(
                MAX_CHALLENGER_WEIGHT,
                max(0.05, 0.5 * sum(gain for _name, gain in challengers)),
            )
            gain_total = sum(gain for _name, gain in challengers)
            mixture_weights[anchor] = 1.0 - challenger_mass
            for name, gain in challengers:
                mixture_weights[name] = challenger_mass * gain / gain_total
        else:
            mixture_weights[anchor] = 1.0
    else:
        best = min(means, key=means.get)
        anchor = best
        mixture_weights[best] = 1.0

    # Fit complete path draws from every weighted candidate, dropping failed
    # final fits and renormalizing the remaining conservative mixture.
    final_draws: dict[str, np.ndarray] = {}
    attempted = []
    for name in mixture_weights:
        try:
            draws = candidates[name](panel, fit_card, asof, n_draws, seed=seed)
            expected = (n_draws, len(card["targets"]["asset_ids"]),
                        expected_horizons)
            if draws.shape != expected or not np.isfinite(draws).all():
                raise ValueError(f"invalid final draws shape or values: {draws.shape}")
            final_draws[name] = draws
        except Exception as exc:
            attempted.append(f"{name}: {exc}")

    active = {name: mixture_weights[name] for name in final_draws}
    if active:
        total = sum(active.values())
        active = {name: value / total for name, value in active.items()}
        if len(active) == 1:
            selected = next(iter(active))
            output = final_draws[selected]
        else:
            rng = np.random.default_rng(seed + 7919)
            names = list(active)
            choose = rng.choice(len(names), size=n_draws,
                                p=[active[name] for name in names])
            output = np.empty((n_draws, len(card["targets"]["asset_ids"]),
                               expected_horizons), dtype=np.float64)
            for model_index, name in enumerate(names):
                mask = np.flatnonzero(choose == model_index)
                if len(mask):
                    source = rng.integers(0, n_draws, size=len(mask))
                    output[mask] = final_draws[name][source]
        selected = (next(iter(active)) if len(active) == 1
                    else f"mixture centered on {anchor}")

        score_text = ", ".join(
            f"{name}={np.mean(list(fold_scores[name].values())):.3f}"
            f"({len(fold_scores[name])}/{len(folds)} folds)"
            for name in means if fold_scores[name]
        )
        weight_text = ", ".join(f"{name}={weight:.2f}" for name, weight in active.items())
        details = f"validation anchor={anchor}; mixture weights [{weight_text}]; relative losses [{score_text}]"
        return output, selected, details

    # Keep valid output possible when the chosen candidates fail on the final fit.
    fallback = [("ARIMA", arima.fit_and_sample), ("ETS", ets.fit_and_sample),
                ("analog", analog.fit_and_sample),
                ("block bootstrap", block_bootstrap.fit_and_sample),
                ("random walk", random_walk.fit_and_sample)]
    for name, model in fallback:
        try:
            if name == "block bootstrap" and card["targets"]["target_type"] != "log_return":
                continue
            draws = model(panel, fit_card, asof, n_draws, seed=seed)
            note = "selection finalists failed; fallback to " + name
            return draws, name, note
        except Exception as exc:
            attempted.append(f"{name}: {exc}")
    raise RuntimeError("selected F1 models and fallbacks failed: " + " | ".join(attempted))
