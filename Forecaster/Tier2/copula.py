"""Gaussian copula on residuals. Pure numpy/scipy, no project imports."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def to_pseudo_obs(x: np.ndarray) -> np.ndarray:
    """Rank-transform a 1D array to (0, 1). Ties broken by stable order."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(T, dtype=float)
    ranks[order] = np.arange(1, T + 1)
    return ranks / (T + 1)


def from_pseudo_obs(u: np.ndarray, sorted_x: np.ndarray) -> np.ndarray:
    """Inverse ECDF via linear interpolation. Consistent with to_pseudo_obs."""
    T = len(sorted_x)
    p = np.arange(1, T + 1) / (T + 1)
    return np.interp(u, p, sorted_x)


def fit_gaussian_copula(residuals: np.ndarray, ridge: float = 1e-6):
    """
    residuals : (T, K) aligned residual matrix, no NaNs.
    returns   : (R, sorted_resid)
        R            : (K, K) correlation matrix on normal scores
        sorted_resid : (T, K) column-wise sorted residuals for inversion
    """
    residuals = np.asarray(residuals, dtype=float)
    T, K = residuals.shape
    if T < 10 or K < 1:
        raise ValueError(f"bad residuals shape: {residuals.shape}")

    U = np.column_stack([to_pseudo_obs(residuals[:, k]) for k in range(K)])
    eps = 1e-6
    U = np.clip(U, eps, 1.0 - eps)
    Z = norm.ppf(U)

    R = np.atleast_2d(np.corrcoef(Z, rowvar=False))
    R = 0.5 * (R + R.T)

    eigmin = np.linalg.eigvalsh(R).min()
    if eigmin < ridge:
        R = R + np.eye(K) * (ridge - eigmin)

    return R, np.sort(residuals, axis=0)


def sample_copula(R: np.ndarray,
                  sorted_resid: np.ndarray,
                  n_draws: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Draw n_draws joint residual samples. Returns (n_draws, K)."""
    K = R.shape[0]
    Z = rng.multivariate_normal(np.zeros(K), R, size=n_draws)
    U = norm.cdf(Z)
    out = np.empty_like(U)
    for k in range(K):
        out[:, k] = from_pseudo_obs(U[:, k], sorted_resid[:, k])
    return out
