"""Executive summary: utilities and copulas for coherent multi-asset draws."""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_changes(panel: pd.DataFrame, assets: list[str], target_type: str) -> np.ndarray:
    """Return aligned numerical changes in target asset order."""
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    wide = data.pivot_table(index="date", columns="asset", values="value").sort_index()
    by_lower = {str(c).lower(): c for c in wide.columns}
    try:
        wide = wide[[by_lower[a.lower()] for a in assets]]
    except KeyError as exc:
        raise ValueError(f"copula target asset missing from panel: {exc}") from exc
    if target_type == "log_return":
        if (wide <= -1).any().any():
            raise ValueError("copula panel contains returns <= -100%")
        changes = np.log1p(wide)
    else:
        changes = wide.diff()
    return changes.dropna().to_numpy(dtype=float)


def correlation_matrix(panel: pd.DataFrame, assets: list[str], target_type: str,
                       minimum_history: int = 10) -> np.ndarray:
    """Estimate a regularized correlation matrix, projecting to PSD if needed."""
    n_assets = len(assets)
    if n_assets < 2:
        return np.eye(n_assets)
    changes = historical_changes(panel, assets, target_type)
    if len(changes) < minimum_history:
        raise ValueError("too little aligned history to estimate dependence")
    corr = np.corrcoef(changes, rowvar=False)
    corr = np.atleast_2d(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    corr = 0.9 * corr + 0.1 * np.eye(n_assets)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    corr = (eigenvectors * np.maximum(eigenvalues, 1e-6)) @ eigenvectors.T
    scale = np.sqrt(np.diag(corr))
    return corr / np.outer(scale, scale)


def impose_copula(draws: np.ndarray, corr: np.ndarray, rng: np.random.Generator,
                  degrees_freedom: float | None = None) -> np.ndarray:
    """Reorder marginal draws according to Gaussian or Student-t latent ranks."""
    values = np.asarray(draws, dtype=float)
    if values.shape[1] < 2:
        return values.copy()
    n_draws, n_assets, n_horizons = values.shape
    corr = np.asarray(corr, dtype=float)
    if corr.shape != (n_assets, n_assets):
        raise ValueError("dependence matrix shape does not match asset axis")
    latent = rng.multivariate_normal(np.zeros(n_assets), corr, size=n_draws)
    if degrees_freedom is not None:
        if degrees_freedom <= 2:
            raise ValueError("Student-t copula degrees_freedom must exceed 2")
        latent /= np.sqrt(rng.chisquare(degrees_freedom, size=n_draws)[:, None] / degrees_freedom)
    order = np.argsort(latent, axis=0)
    out = np.empty_like(values)
    for ai in range(n_assets):
        for hi in range(n_horizons):
            marginal = np.sort(values[:, ai, hi])
            out[order[:, ai], ai, hi] = marginal
    return out
