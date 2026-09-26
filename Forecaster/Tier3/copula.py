"""Gaussian copula utilities. Pure numpy/scipy. No project imports.

Difference from Tier 2: this module does not provide an empirical inverse.
Tier 3 inverts through the skewed-t PPF instead, so the copula only needs
to (a) estimate the correlation matrix on normal scores, and (b) produce
correlated uniforms.
"""
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


def fit_copula_R(z: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Correlation matrix on normal scores of aligned standardized residuals.

    z : (T, K) standardized residual matrix, no NaNs.
    Returns R : (K, K) correlation matrix, ridge-regularised toward I.
    """
    z = np.asarray(z, dtype=float)
    T, K = z.shape
    if T < 10 or K < 1:
        raise ValueError(f"bad z shape: {z.shape}")

    U = np.column_stack([to_pseudo_obs(z[:, k]) for k in range(K)])
    U = np.clip(U, 1e-6, 1.0 - 1e-6)
    Z = norm.ppf(U)

    R = np.atleast_2d(np.corrcoef(Z, rowvar=False))
    R = 0.5 * (R + R.T)

    eigmin = np.linalg.eigvalsh(R).min()
    if eigmin < ridge:
        R = R + np.eye(K) * (ridge - eigmin)
    return R


def sample_copula_uniform(R: np.ndarray,
                          n_draws: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Draw correlated uniforms from the Gaussian copula. Shape (n_draws, K)."""
    K = R.shape[0]
    Z = rng.multivariate_normal(np.zeros(K), R, size=n_draws)
    return norm.cdf(Z)
