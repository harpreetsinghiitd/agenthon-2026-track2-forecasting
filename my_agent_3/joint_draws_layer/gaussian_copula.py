"""Executive summary: couple marginal draws using a Gaussian copula."""

from __future__ import annotations

import numpy as np

from . import correlation_matrix, impose_copula


def couple(draws: np.ndarray, panel, card: dict, seed: int = 61) -> np.ndarray:
    assets = [str(a) for a in card["targets"]["asset_ids"]]
    if len(assets) < 2:
        return np.asarray(draws).copy()
    corr = correlation_matrix(panel, assets, card["targets"]["target_type"])
    return impose_copula(np.asarray(draws), corr, np.random.default_rng(seed))
