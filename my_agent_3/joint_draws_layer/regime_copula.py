"""Executive summary: mix calm and stress dependence estimates."""

from __future__ import annotations

import numpy as np

from . import historical_changes, impose_copula


def couple(draws: np.ndarray, panel, card: dict, seed: int = 64) -> np.ndarray:
    assets = [str(a) for a in card["targets"]["asset_ids"]]
    if len(assets) < 2:
        return np.asarray(draws).copy()
    changes = historical_changes(panel, assets, card["targets"]["target_type"])
    if len(changes) < 25:
        raise ValueError("regime copula requires at least 25 aligned observations")
    system_vol = np.sqrt(np.mean(changes ** 2, axis=1))
    cutoff = np.quantile(system_vol, 0.75)
    calm, stress = changes[system_vol <= cutoff], changes[system_vol > cutoff]
    if len(stress) < 8 or len(calm) < 8:
        raise ValueError("insufficient calm/stress samples for regime copula")
    calm_corr = np.corrcoef(calm, rowvar=False)
    stress_corr = np.corrcoef(stress, rowvar=False)
    calm_corr = np.atleast_2d(np.nan_to_num(calm_corr, nan=0.0))
    stress_corr = np.atleast_2d(np.nan_to_num(stress_corr, nan=0.0))
    # PSD projection and modest shrinkage stabilize small regime subsets.
    matrices = []
    for corr in (calm_corr, stress_corr):
        corr = 0.85 * corr + 0.15 * np.eye(len(assets))
        vals, vecs = np.linalg.eigh(corr)
        corr = (vecs * np.maximum(vals, 1e-5)) @ vecs.T
        scale = np.sqrt(np.diag(corr))
        matrices.append(corr / np.outer(scale, scale))
    rng = np.random.default_rng(seed)
    stress_probability = min(0.5, max(0.1, float(np.mean(system_vol[-20:] > cutoff))))
    state = rng.random(draws.shape[0]) < stress_probability
    out = np.empty_like(np.asarray(draws, dtype=float))
    for mask, corr in ((~state, matrices[0]), (state, matrices[1])):
        if mask.any():
            out[mask] = impose_copula(np.asarray(draws)[mask], corr,
                                      np.random.default_rng(rng.integers(0, 2**32)))
    return out
