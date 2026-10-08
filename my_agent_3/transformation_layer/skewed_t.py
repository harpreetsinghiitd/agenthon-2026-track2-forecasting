"""Executive summary: replace Gaussian marginal tails with skewed Student-t draws."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def transform(draws: np.ndarray, seed: int = 52, degrees_freedom: float = 5.0,
              direction: str | None = None, strength: float = 0.25) -> np.ndarray:
    """Apply a standardized fat-tail transform while preserving sample ranks.

    ``seed`` remains in the interface for caller compatibility; the transform is
    deterministic conditional on the incoming Monte Carlo draws.
    """
    values = np.asarray(draws, dtype=float)
    out = np.empty_like(values)
    if values.ndim != 3:
        raise ValueError("draws must have shape [n_draws, n_assets, n_horizons]")
    if degrees_freedom <= 2:
        raise ValueError("Student-t degrees_freedom must exceed 2")
    if direction not in {None, "left", "right"}:
        raise ValueError("direction must be None, 'left', or 'right'")
    for ai in range(values.shape[1]):
        for hi in range(values.shape[2]):
            sample = values[:, ai, hi]
            center = float(np.mean(sample))
            scale = float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0
            if scale <= 1e-12:
                out[:, ai, hi] = sample
                continue
            order = np.argsort(sample, kind="stable")
            probabilities = (np.arange(len(sample), dtype=float) + 0.5) / len(sample)
            z = student_t.ppf(probabilities, degrees_freedom)
            if direction == "right":
                z = z * np.where(z > 0, 1.0 + strength, 1.0 - 0.5 * strength)
            elif direction == "left":
                z = z * np.where(z < 0, 1.0 + strength, 1.0 - 0.5 * strength)
            z -= z.mean()
            z_sd = z.std(ddof=1)
            transformed = center + z * (scale / z_sd if z_sd > 0 else scale)
            out[order, ai, hi] = transformed
    return out
