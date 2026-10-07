"""Executive summary: Gaussian C-vine special-case dependence implementation."""

from __future__ import annotations

import numpy as np

from . import correlation_matrix, impose_copula


def couple(draws: np.ndarray, panel, card: dict, seed: int = 63) -> np.ndarray:
    """Apply a Gaussian C-vine special case with shrinkage-stabilized pair links.

    This provides a dependable vine-shaped baseline. Flexible non-Gaussian pair
    copula selection is a later enhancement; this implementation should not be
    described as fitting a general vine family.
    """
    assets = [str(a) for a in card["targets"]["asset_ids"]]
    if len(assets) < 2:
        return np.asarray(draws).copy()
    corr = correlation_matrix(panel, assets, card["targets"]["target_type"],
                               minimum_history=15)
    return impose_copula(np.asarray(draws), corr, np.random.default_rng(seed))
