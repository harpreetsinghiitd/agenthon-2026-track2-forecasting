"""Statistical models — the Phase 2/4 forecast core.

Role in the pipeline
--------------------
Two estimators feed `harness.tools.simulate`:

* ``bootstrap_paths`` turns one asset's demeaned daily changes into many
  simulated innovation paths via the stationary block bootstrap.  The block
  structure keeps short-run autocorrelation that an i.i.d. resample would
  destroy; the demeaning guarantees the innovations carry zero drift, so any
  centre movement enters only through the explicit drift overlay in
  ``simulate_mixture``.
* ``fit_arima`` fits a small ARMA on the same changes.  In Phase 2 its 1-step
  forecast is the drift estimate; from Phase 3 on, the bridge replaces the
  centre adjustment with the text-conditioned vector and the ARIMA fit degrades
  to a degradation signal — if it cannot fit, the fallback forecaster runs
  instead.
* ``fit_copula`` (Phase 4) fits a t-copula on the ARIMA residuals across
  assets, capturing cross-asset tail dependence.  Returns the fitted copula
  parameters for joint simulation in ``simulate_mixture``.

All functions are pure: randomness enters only through a caller-supplied
``numpy.random.Generator``, so determinism lives at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

__all__ = [
    "DEFAULT_ARIMA_ORDER",
    "DEFAULT_BLOCK_SIZE",
    "MIN_ARIMA_OBS",
    "StatsError",
    "bootstrap_paths",
    "fit_arima",
    "fit_copula",
    "CopulaParams",
]

#: Mean block length for the stationary bootstrap.  A week of business days is
#: the playbook's default for capturing short-run persistence without mangling
#: the marginal distribution.
DEFAULT_BLOCK_SIZE: int = 7

#: AR(1) on demeaned changes: the smallest model that admits momentum or mild
#: mean reversion.  Larger orders are a Phase 5 concern, gated by backtests.
DEFAULT_ARIMA_ORDER: tuple[int, int, int] = (1, 0, 0)

#: Fewer observations than this and the ARIMA fit is noise-on-noise; the caller
#: degrades to the fallback forecaster instead.
MIN_ARIMA_OBS: int = 30


class StatsError(ValueError):
    """Raised when an estimator receives unusable input: too short, non-finite,
    or degenerate (zero-variance overlap)."""


def _as_clean_1d(series: Any, *, what: str) -> np.ndarray:
    """Coerce to a finite 1-D float array, rejecting NaN and infinity."""
    arr = np.asarray(series, dtype=float).ravel()
    if arr.size == 0:
        raise StatsError(f"{what}: series is empty")
    if not np.isfinite(arr).all():
        raise StatsError(f"{what}: series contains NaN or infinity")
    return arr


def bootstrap_paths(
    changes: Any,
    n_paths: int,
    horizon: int,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary block bootstrap of pre-demeaned daily changes.

    A path is a concatenation of contiguous blocks drawn from the source series:
    each block starts at a uniform index and has geometric length with mean
    ``block_size`` (Politis–Romano stationary bootstrap).  Wrap-around indexing
    keeps early and late observations exchangeable.

    Args:
        changes: Pre-demeaned daily changes, shape ``(n_obs,)``.  The caller
            demeans so the estimator cannot double-count the sample mean.
        n_paths: Number of paths to draw.
        horizon: Path length in per-step units (business days).
        block_size: Mean block length; 1 recovers i.i.d. resampling.
        rng: Seeded generator; the caller owns determinism.

    Returns:
        Array of shape ``(n_paths, horizon)`` of resampled changes, in the
        panel's own per-step units.

    Raises:
        StatsError: unusable input series or parameter.
    """
    src = _as_clean_1d(changes, what="bootstrap_paths")
    if n_paths < 1:
        raise StatsError(f"n_paths must be positive, got {n_paths}")
    if horizon < 1:
        raise StatsError(f"horizon must be positive, got {horizon}")
    if block_size < 1:
        raise StatsError(f"block_size must be positive, got {block_size}")
    n_obs = src.size
    if n_obs < 2:
        raise StatsError(f"bootstrap_paths: need at least 2 observations, got {n_obs}")

    p_new_block = 1.0 / float(block_size)
    out = np.empty((n_paths, horizon), dtype=float)
    for path in range(n_paths):
        filled = 0
        while filled < horizon:
            length = int(rng.geometric(p_new_block))
            start = int(rng.integers(n_obs))
            take = min(length, horizon - filled)
            idx = (start + np.arange(take)) % n_obs
            out[path, filled : filled + take] = src[idx]
            filled += take
    return out


def fit_arima(
    series: Any,
    *,
    order: tuple[int, int, int] = DEFAULT_ARIMA_ORDER,
    exog: Any | None = None,
    trend: str = "n",
) -> ARIMAResults:
    """Fit an ARIMA on a per-step series.

    The series passed here is the per-step one (diffed levels, or the values
    themselves for a ``log_return`` card), and in Phase 2 it is demeaned — the
    drift enters explicitly downstream, so the fit carries no constant and no
    trend (``trend="n"``).

    Args:
        series: Per-step observations, shape ``(n_obs,)``.
        order: ``(p, d, q)``; the default AR(1) assumes the caller has already
            differenced.
        exog: Optional exogenous regressors aligned with ``series`` (Phase 5);
            must row-align with the series when given.
        trend: statsmodels trend term; ``"n"`` because the caller demeans.

    Returns:
        The fitted ``ARIMAResults``, exposing ``.forecast(steps=...)`` and
        ``.simulate(...)``.

    Raises:
        StatsError: fewer than ``MIN_ARIMA_OBS`` observations or non-finite
            input.  statsmodels' own fit failures propagate to the caller's
            fallback guard unchanged.
    """
    arr = _as_clean_1d(series, what="fit_arima")
    if arr.size < MIN_ARIMA_OBS:
        raise StatsError(
            f"fit_arima: {arr.size} observations, need at least {MIN_ARIMA_OBS}"
        )
    if exog is not None:
        exog_arr = np.asarray(exog, dtype=float)
        if exog_arr.shape[0] != arr.shape[0]:
            raise StatsError(
                f"fit_arima: exog has {exog_arr.shape[0]} rows, series has {arr.shape[0]}"
            )
        exog = exog_arr

    model = ARIMA(arr, order=order, exog=exog, trend=trend)
    return model.fit()


@dataclass(frozen=True, slots=True)
class CopulaParams:
    """Fitted t-copula parameters for joint simulation.

    Attributes:
        corr: Correlation matrix ``(n_assets, n_assets)``.
        df: Degrees of freedom ``> 2`` (tail heaviness).
        marginal_cdfs: List of empirical CDF callables for each margin.
    """
    corr: np.ndarray
    df: float
    marginal_cdfs: list


def fit_copula(
    residual_matrix: np.ndarray,
    *,
    family: str = "t",
    df_guess: float = 6.0,
    rng: np.random.Generator | None = None,
) -> CopulaParams:
    """Fit a t-copula on standardised ARIMA residuals.

    The t-copula captures cross-asset tail dependence.  The procedure:

    1. Convert each asset's residuals to pseudo-observations via the empirical
       CDF (rank transform: ``rank / (n + 1)``).
    2. Transform to t-quantiles using a working ``df_guess``.
    2. Estimate the correlation matrix from the t-quantiles (MLE for Gaussian
       copula, then refine for t-copula).
    3. Profile the log-likelihood over degrees of freedom to get the final ``df``.

    Args:
        residual_matrix: Shape ``(n_assets, n_obs)`` of demeaned ARIMA residuals
            (one row per asset).  Each row must have the same length.
        family: ``"t"`` (Student-t) or ``"gaussian"``.
        df_guess: Starting guess for degrees of freedom (used only for ``"t"``).
        rng: Optional generator for any stochastic steps (currently deterministic).

    Returns:
        ``CopulaParams`` with correlation matrix, df, and marginal ECDF callables.

    Raises:
        StatsError: non-finite input, mismatched row lengths, or optimisation
            failure.
    """
    if residual_matrix.ndim != 2:
        raise StatsError(f"residual_matrix must be 2-D (n_assets, n_obs), got {residual_matrix.ndim}-D")
    n_assets, n_obs = residual_matrix.shape
    if n_assets < 2:
        raise StatsError(f"copula needs at least 2 assets, got {n_assets}")
    if n_obs < 30:
        raise StatsError(f"copula needs at least 30 observations, got {n_obs}")
    if not np.isfinite(residual_matrix).all():
        raise StatsError("residual_matrix contains NaN or infinity")

    # 1. Pseudo-observations (empirical CDF)
    ranks = np.argsort(np.argsort(residual_matrix, axis=1), axis=1) + 1
    u = ranks / (n_obs + 1.0)  # shape (n_assets, n_obs), values in (0, 1)

    if family == "gaussian":
        # Gaussian copula: correlation = Pearson on probit(u)
        from scipy.stats import norm
        z = norm.ppf(u)
        corr = np.corrcoef(z)
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals[0] <= 1e-10:
            corr = corr + np.eye(n_assets) * 1e-6
        return CopulaParams(
            corr=corr,
            df=np.inf,
            marginal_cdfs=[lambda x, u=u_i: np.searchsorted(np.sort(u_i), x, side="right") / (n_obs + 1)
                          for u_i in u],
        )

    # 2. t-copula: profile likelihood over df
    # Initial correlation from probit transform (Gaussian approximation)
    from scipy.stats import norm
    z = norm.ppf(u)
    corr_init = np.corrcoef(z)
    # Ensure PD
    eigvals = np.linalg.eigvalsh(corr_init)
    if eigvals[0] <= 1e-10:
        corr_init = corr_init + np.eye(n_assets) * 1e-6

    def neg_loglik_t(df: float) -> float:
        """Negative log-likelihood for t-copula with fixed df."""
        # t-quantiles
        t_quant = student_t.ppf(u, df=df)
        # Correlation from t-quantiles
        corr = np.corrcoef(t_quant)
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals[0] <= 1e-10:
            return 1e12
        # Log-likelihood
        sign, logdet = np.linalg.slogdet(corr)
        if sign <= 0:
            return 1e12
        inv_corr = np.linalg.inv(corr)
        quad = np.sum(t_quant * (inv_corr @ t_quant))
        const = n_obs * (np.log(np.pi) * n_assets + logdet)
        # t-copula log-likelihood (up to constants)
        loglik = -0.5 * (n_obs * n_assets * np.log(df) + n_obs * logdet + (df + n_assets) * np.sum(np.log(1 + np.sum(t_quant * (inv_corr @ t_quant) / df, axis=0))))
        return -loglik

    # Optimize df
    res = minimize(neg_loglik_t, x0=[df_guess], bounds=[(2.1, 50.0)], method="L-BFGS-B")
    if not res.success:
        _LOG.warning("t-copula df optimisation failed: %s; using df=6", res.message)
        df_final = 6.0
    else:
        df_final = float(np.clip(res.x[0], 2.1, 50.0))

    # Final correlation at optimal df
    t_quant = student_t.ppf(u, df=df_final)
    corr = np.corrcoef(t_quant)
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals[0] <= 1e-10:
        corr = corr + np.eye(n_assets) * 1e-6

    # Marginal ECDFs for PIT
    ecdfs = []
    for i in range(n_assets):
        sorted_u = np.sort(u[i])
        ecdfs.append(lambda x, su=sorted_u: np.searchsorted(su, x, side="right") / (n_obs + 1))

    return CopulaParams(corr=corr, df=df_final, marginal_cdfs=ecdfs)
