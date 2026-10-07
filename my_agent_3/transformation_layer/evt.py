"""Executive summary: use fitted generalized-Pareto tails as an optional refinement."""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


def transform(draws: np.ndarray, residuals: dict[str, np.ndarray] | None = None,
              seed: int = 54, threshold_quantile: float = 0.9) -> np.ndarray:
    """Replace extreme sample quantiles using empirical-residual GPD tails."""
    out = np.asarray(draws, dtype=float).copy()
    if residuals is None:
        return out
    rng = np.random.default_rng(seed)
    for ai, asset in enumerate(residuals):
        if ai >= out.shape[1]:
            break
        r = np.asarray(residuals[asset], dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 50:
            continue
        for hi in range(out.shape[2]):
            sample = out[:, ai, hi]
            center = np.median(sample)
            for sign in (-1.0, 1.0):
                tail = sign * (r - np.median(r))
                cutoff = float(np.quantile(tail, threshold_quantile))
                exceed = tail[tail > cutoff] - cutoff
                if len(exceed) < 20:
                    continue
                shape, _, scale = genpareto.fit(exceed, floc=0)
                mask = sign * (sample - center) > np.quantile(sign * (sample - center), threshold_quantile)
                count = int(mask.sum())
                if count:
                    u = np.clip(rng.random(count), 1e-8, 1 - 1e-8)
                    sampled = cutoff + genpareto.ppf(u, shape, loc=0, scale=scale)
                    sample[mask] = center + sign * sampled
            out[:, ai, hi] = sample
    return out
