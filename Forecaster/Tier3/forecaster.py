"""Tier 3 numerical forecaster.

ARIMA (conditional mean) + GARCH(1,1) skewed-t (vol + tails) +
Gaussian copula on standardized residuals (joint dependence).

Self-contained: no imports from Forecaster/ or any other project module.
Signature unchanged from the previous numerical forecaster:
    fit_and_sample(panel, card, asof, n_draws)
        -> ndarray[n_draws, n_assets, n_horizons]
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model
from arch.univariate.distribution import SkewStudent


# ==========================================================================
# ARIMA
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
                            m = ARIMA(y, order=(p, d, q),
                                      enforce_stationarity=True,
                                      enforce_invertibility=True
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
# GARCH(1,1) with skewed-t
# ==========================================================================
class GarchAsset:
    def __init__(self, p: int = 1, q: int = 1, maxiter: int = 500):
        self.p, self.q, self.maxiter = p, q, maxiter

    def fit(self, resid: np.ndarray) -> "GarchAsset":
        resid = np.asarray(resid, dtype=float)
        resid = resid[~np.isnan(resid)]
        if len(resid) < 50:
            raise ValueError(f"residual too short for GARCH: len={len(resid)}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(resid, vol="GARCH", p=self.p, q=self.q,
                            dist="skewt", rescale=False)
            self.res_ = am.fit(disp="off", show_warning=False,
                               options={"maxiter": self.maxiter})

        self.sigma_ = np.asarray(self.res_.conditional_volatility, dtype=float)
        self.z_ = resid / np.maximum(self.sigma_, 1e-12)
        self.eta_ = float(self.res_.params.get("eta", 5.0))
        self.lambda_ = float(self.res_.params.get("lambda", 0.0))
        self._dist = SkewStudent()
        return self

    def forecast_cum_vol(self, horizons: list[int]) -> np.ndarray:
        max_h = int(max(horizons))
        f = self.res_.forecast(horizon=max_h, reindex=False)
        var_path = np.asarray(f.variance.values[0], dtype=float)
        cum_var = np.cumsum(var_path)
        return np.array([np.sqrt(cum_var[h - 1]) for h in horizons], dtype=float)

    def skewt_ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.asarray(self._dist.ppf(u, [self.eta_, self.lambda_]),
                          dtype=float)


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


def _fit_copula_R(z: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    T, K = z.shape
    U = np.column_stack([_to_pseudo_obs(z[:, k]) for k in range(K)])
    U = np.clip(U, 1e-6, 1.0 - 1e-6)
    Z = norm.ppf(U)
    R = np.atleast_2d(np.corrcoef(Z, rowvar=False))
    R = 0.5 * (R + R.T)
    eigmin = np.linalg.eigvalsh(R).min()
    if eigmin < ridge:
        R = R + np.eye(K) * (ridge - eigmin)
    return R


def _sample_copula_uniform(R: np.ndarray, n_draws: int,
                           rng: np.random.Generator) -> np.ndarray:
    K = R.shape[0]
    Z = rng.multivariate_normal(np.zeros(K), R, size=n_draws)
    return norm.cdf(Z)


# ==========================================================================
# Helpers
# ==========================================================================
def _find_column(wide: pd.DataFrame, asset_id: str) -> str:
    if asset_id in wide.columns:
        return asset_id
    lower = {c.lower(): c for c in wide.columns}
    if asset_id.lower() in lower:
        return lower[asset_id.lower()]
    raise KeyError(f"asset {asset_id!r} not found in panel columns")


def _prep_series(series: pd.Series, target_type: str) -> pd.Series:
    """Transform the panel series into the space the scorer expects."""
    if target_type == "level":
        return series
    if target_type == "log_return":
        r = series.astype(float).to_numpy()
        return pd.Series(np.log1p(r).cumsum())
    raise ValueError(f"unknown target_type {target_type!r}")


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


# ==========================================================================
# Entry point (same signature as the old numerical forecaster)
# ==========================================================================
def fit_and_sample(panel: pd.DataFrame, card: dict, asof: str,
                   n_draws: int) -> np.ndarray:
    targets = card["targets"]
    asset_ids = list(targets["asset_ids"])
    horizons = list(targets["horizons"])
    target_type = targets["target_type"]

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    asof_ts = pd.Timestamp(asof)
    panel = panel[panel["date"] <= asof_ts]

    wide = (panel
            .pivot_table(index="date", columns="asset", values="value")
            .sort_index())

    K = len(asset_ids)
    H = len(horizons)
    max_h = max(horizons)
    draws = np.zeros((n_draws, K, H), dtype=float)
    rng = np.random.default_rng(seed=42)

    arimas: list = [None] * K
    garchs: list = [None] * K
    means = np.zeros((K, H))
    vols = np.zeros((K, H))
    fallback: set[int] = set()

    for ai, asset in enumerate(asset_ids):
        try:
            col = _find_column(wide, asset)
        except KeyError:
            fallback.add(ai)
            continue
        raw = wide[col].dropna()
        if len(raw) < 50:
            fallback.add(ai)
            continue

        try:
            series = _prep_series(raw, target_type)
            a = ArimaAsset().fit(series.to_numpy(dtype=float))
            arimas[ai] = a

            m_hat, se_hat = a.forecast(max_h)

            try:
                g = GarchAsset().fit(a.resid_)
                v_hat = g.forecast_cum_vol(horizons)
                garchs[ai] = g
            except Exception:
                v_hat = np.array([se_hat[h - 1] for h in horizons],
                                 dtype=float)

            for hi, h in enumerate(horizons):
                means[ai, hi] = m_hat[h - 1]
                vols[ai, hi] = v_hat[hi]
        except Exception:
            fallback.add(ai)

    for ai in fallback:
        try:
            col = _find_column(wide, asset_ids[ai])
            series = wide[col].dropna()
        except KeyError:
            series = pd.Series(dtype=float)
        draws[:, ai, :] = _fallback_draws(series, horizons, n_draws,
                                          target_type, rng)

    good_idx = [ai for ai in range(K) if ai not in fallback]
    if not good_idx:
        return draws

    garch_idx = [ai for ai in good_idx if garchs[ai] is not None]
    R = None
    if len(garch_idx) >= 1:
        min_len = min(len(garchs[ai].z_) for ai in garch_idx)
        if min_len >= 30:
            Z = np.column_stack([garchs[ai].z_[-min_len:] for ai in garch_idx])
            mask = ~np.isnan(Z).any(axis=1)
            Z = Z[mask]
            if Z.shape[0] >= 30:
                R = _fit_copula_R(Z)

    for hi in range(H):
        z_star = np.zeros((n_draws, K), dtype=float)

        if R is not None and len(garch_idx) > 0:
            U = _sample_copula_uniform(R, n_draws, rng)
            for j, ai in enumerate(garch_idx):
                z_star[:, ai] = garchs[ai].skewt_ppf(U[:, j])

        for ai in good_idx:
            if ai in garch_idx:
                continue
            z_star[:, ai] = rng.standard_normal(n_draws)

        draws[:, good_idx, hi] = (z_star[:, good_idx]
                                  * vols[good_idx, hi][None, :]
                                  + means[good_idx, hi][None, :])

    return draws
