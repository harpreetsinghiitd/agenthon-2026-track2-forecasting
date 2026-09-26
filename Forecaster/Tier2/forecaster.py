"""Tier 2 numerical forecaster.

Per-asset ARIMA marginals + Gaussian copula on residuals.

Self-contained: no imports from Forecaster/ or any other project module.
Same interface as the previous numerical forecaster:
    fit_and_sample(panel, card, asof, n_draws) -> ndarray[n_draws, n_assets, n_horizons]
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm


# ==========================================================================
# ARIMA per asset
# ==========================================================================
class ArimaAsset:
    def __init__(self, max_p=3, max_d=1, max_q=3, ic="aic", maxiter=200):
        self.max_p, self.max_d, self.max_q = max_p, max_d, max_q
        self.ic, self.maxiter = ic, maxiter

    def fit(self, y: np.ndarray) -> "ArimaAsset":
        from statsmodels.tsa.arima.model import ARIMA

        y = np.asarray(y, dtype=float)
        y = y[~np.isnan(y)]
        if len(y) < 20:
            raise ValueError(f"series too short: len={len(y)}")

        best_ic, best_model, best_order = np.inf, None, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(self.max_p + 1):
                for d in range(self.max_d + 1):
                    for q in range(self.max_q + 1):
                        if p == 0 and q == 0:
                            continue
                        try:
                            m = ARIMA(
                                y, order=(p, d, q),
                                enforce_stationarity=True,
                                enforce_invertibility=True,
                            ).fit(method_kwargs={"maxiter": self.maxiter})
                        except Exception:
                            continue
                        v = getattr(m, self.ic.lower())
                        if np.isfinite(v) and v < best_ic:
                            best_ic, best_model, best_order = v, m, (p, d, q)

        if best_model is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_model = ARIMA(y, order=(0, 1, 0)).fit()
            best_order = (0, 1, 0)

        self.model_ = best_model
        self.order_ = best_order
        self.resid_ = np.asarray(best_model.resid, dtype=float)
        self.fitted_ = np.asarray(best_model.fittedvalues, dtype=float)
        try:
            fc = best_model.get_forecast(steps=1)
            self.cond_mean_ = float(np.asarray(fc.predicted_mean).ravel()[0])
        except Exception:
            self.cond_mean_ = float(y[-1])
        return self

    def forecast(self, steps: int) -> tuple[np.ndarray, np.ndarray]:
        fc = self.model_.get_forecast(steps=steps)
        means = np.asarray(fc.predicted_mean, dtype=float).ravel()
        ses = np.asarray(fc.se_mean, dtype=float).ravel()
        if means.shape[0] < steps:
            pad = steps - means.shape[0]
            means = np.concatenate([means, np.full(pad, means[-1])])
            ses = np.concatenate([ses, np.full(pad, ses[-1])])
        return means, ses


# ==========================================================================
# Gaussian copula
# ==========================================================================
def _to_pseudo_obs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    T = len(x)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(T, dtype=float)
    ranks[order] = np.arange(1, T + 1)
    return ranks / (T + 1)


def _from_pseudo_obs(u: np.ndarray, sorted_x: np.ndarray) -> np.ndarray:
    T = len(sorted_x)
    p = np.arange(1, T + 1) / (T + 1)
    return np.interp(u, p, sorted_x)


def _fit_gaussian_copula(residuals: np.ndarray, ridge: float = 1e-6):
    residuals = np.asarray(residuals, dtype=float)
    T, K = residuals.shape
    if T < 10 or K < 1:
        raise ValueError(f"bad residuals shape: {residuals.shape}")

    U = np.column_stack([_to_pseudo_obs(residuals[:, k]) for k in range(K)])
    U = np.clip(U, 1e-6, 1.0 - 1e-6)
    Z = norm.ppf(U)

    R = np.atleast_2d(np.corrcoef(Z, rowvar=False))
    R = 0.5 * (R + R.T)
    eigmin = np.linalg.eigvalsh(R).min()
    if eigmin < ridge:
        R = R + np.eye(K) * (ridge - eigmin)
    return R, np.sort(residuals, axis=0)


def _sample_copula(R: np.ndarray, sorted_resid: np.ndarray,
                   n_draws: int, rng: np.random.Generator) -> np.ndarray:
    K = R.shape[0]
    Z = rng.multivariate_normal(np.zeros(K), R, size=n_draws)
    U = norm.cdf(Z)
    out = np.empty_like(U)
    for k in range(K):
        out[:, k] = _from_pseudo_obs(U[:, k], sorted_resid[:, k])
    return out


# ==========================================================================
# Helpers
# ==========================================================================
def _find_column(wide: pd.DataFrame, asset_id: str) -> str:
    if asset_id in wide.columns:
        return asset_id
    lower = {c.lower(): c for c in wide.columns}
    if asset_id.lower() in lower:
        return lower[asset_id.lower()]
    raise KeyError(f"asset {asset_id!r} not found in panel columns {list(wide.columns)}")


def _fallback_draws(series: pd.Series, horizons: list[int],
                    n_draws: int, target_type: str,
                    rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n_draws, len(horizons)))
    if len(series) < 2:
        return out
    if target_type == "level":
        last = float(series.iloc[-1])
        d = series.diff().dropna()
        sigma_d = float(d.std()) if len(d) > 1 else 0.0
        for hi, h in enumerate(horizons):
            out[:, hi] = last + rng.normal(0.0, sigma_d * np.sqrt(h), n_draws)
    elif target_type == "log_return":
        r = series.astype(float).to_numpy()
        log_r = np.log1p(r)
        mu = float(np.mean(log_r))
        sigma = float(np.std(log_r))
        for hi, h in enumerate(horizons):
            out[:, hi] = rng.normal(mu * h, sigma * np.sqrt(h), n_draws)
    return out


def _prep_series(series: pd.Series, target_type: str) -> pd.Series:
    """Transform the panel series into the space the scorer expects."""
    if target_type == "level":
        return series
    if target_type == "log_return":
        r = series.astype(float).to_numpy()
        return pd.Series(np.log1p(r).cumsum())
    raise ValueError(f"unknown target_type {target_type!r}")


# ==========================================================================
# Entry point (same signature as the old numerical forecaster)
# ==========================================================================
def fit_and_sample(panel: pd.DataFrame, card: dict, asof: str,
                   n_draws: int) -> np.ndarray:
    targets = card["targets"]
    asset_ids = list(targets["asset_ids"])
    horizons = list(targets["horizons"])
    target_type = targets["target_type"]

    # --- Preprocess panel ---------------------------------------------------
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    asof_ts = pd.Timestamp(asof)
    panel = panel[panel["date"] <= asof_ts]

    wide = (panel
            .pivot_table(index="date", columns="asset", values="value")
            .sort_index())

    n_assets = len(asset_ids)
    n_horizons = len(horizons)
    max_h = max(horizons)
    draws = np.zeros((n_draws, n_assets, n_horizons), dtype=float)
    rng = np.random.default_rng(seed=42)

    # --- Per-asset ARIMA ----------------------------------------------------
    models: list[ArimaAsset | None] = [None] * n_assets
    residuals: list[np.ndarray | None] = [None] * n_assets
    cond_means = np.zeros((n_assets, n_horizons), dtype=float)
    cond_stds = np.zeros((n_assets, n_horizons), dtype=float)
    fallback_assets: set[int] = set()

    for ai, asset in enumerate(asset_ids):
        try:
            col = _find_column(wide, asset)
        except KeyError:
            fallback_assets.add(ai)
            continue

        raw = wide[col].dropna()
        if len(raw) < 20:
            fallback_assets.add(ai)
            continue

        try:
            series = _prep_series(raw, target_type)
            m = ArimaAsset().fit(series.to_numpy(dtype=float))
            means, ses = m.forecast(max_h)
            models[ai] = m
            residuals[ai] = m.resid_
            for hi, h in enumerate(horizons):
                cond_means[ai, hi] = means[h - 1]
                cond_stds[ai, hi] = ses[h - 1]
        except Exception:
            fallback_assets.add(ai)

    # Fallback for any asset that failed ARIMA
    for ai in fallback_assets:
        try:
            col = _find_column(wide, asset_ids[ai])
            series = wide[col].dropna()
        except KeyError:
            series = pd.Series(dtype=float)
        draws[:, ai, :] = _fallback_draws(series, horizons, n_draws,
                                          target_type, rng)

    # --- Copula on the surviving assets ------------------------------------
    good_idx = [ai for ai in range(n_assets) if ai not in fallback_assets]
    if not good_idx:
        return draws

    def _independent_fill():
        for ai in good_idx:
            r = residuals[ai]
            rstd = float(np.std(r)) if r is not None and len(r) > 1 else 0.0
            for hi in range(n_horizons):
                scale = cond_stds[ai, hi] / max(rstd, 1e-12)
                draws[:, ai, hi] = (rng.standard_normal(n_draws)
                                    * rstd * scale + cond_means[ai, hi])

    min_len = min(len(residuals[ai]) for ai in good_idx)
    if min_len < 30:
        _independent_fill()
        return draws

    resid_matrix = np.column_stack([residuals[ai][-min_len:] for ai in good_idx])
    mask = ~np.isnan(resid_matrix).any(axis=1)
    resid_matrix = resid_matrix[mask]
    if resid_matrix.shape[0] < 30:
        _independent_fill()
        return draws

    R, sorted_resid = _fit_gaussian_copula(resid_matrix)
    resid_stds = np.maximum(resid_matrix.std(axis=0), 1e-12)

    # --- Joint sampling per horizon ----------------------------------------
    for hi, h in enumerate(horizons):
        joint = _sample_copula(R, sorted_resid, n_draws, rng)
        scale = np.array([cond_stds[ai, hi] / resid_stds[k]
                          for k, ai in enumerate(good_idx)], dtype=float)
        means = np.array([cond_means[ai, hi] for ai in good_idx], dtype=float)
        draws[:, good_idx, hi] = joint * scale[None, :] + means[None, :]

    return draws
