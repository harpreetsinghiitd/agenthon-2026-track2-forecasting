"""Tier 2 forecaster: per-asset ARIMA + Gaussian copula on residuals."""
from __future__ import annotations
import numpy as np
import pandas as pd

from Forecaster.Tier2.arima import ArimaAsset
from Forecaster.Tier2.copula import fit_gaussian_copula, sample_copula


class CopulaForecaster:
    def __init__(self, n_draws: int = 500, seed: int = 42):
        self.n_draws = n_draws
        self.rng = np.random.default_rng(seed)
        self.assets_: list[str] = []
        self.per_asset_: dict[str, ArimaAsset] = {}
        self.R_: np.ndarray | None = None
        self.sorted_resid_: np.ndarray | None = None
        self.cond_means_: np.ndarray | None = None

    def fit(self, wide: pd.DataFrame) -> "CopulaForecaster":
        self.assets_ = list(wide.columns)
        K = len(self.assets_)
        if K == 0:
            raise ValueError("empty wide panel")

        # 1) Fit ARIMA per asset
        for a in self.assets_:
            y = wide[a].to_numpy(dtype=float)
            self.per_asset_[a] = ArimaAsset().fit(y)

        # 2) Align residuals across assets (truncate to shortest)
        min_len = min(len(self.per_asset_[a].resid_) for a in self.assets_)
        if min_len < 30:
            raise ValueError(f"too few aligned residuals: {min_len}")

        resid = np.column_stack([
            self.per_asset_[a].resid_[-min_len:] for a in self.assets_
        ])
        mask = ~np.isnan(resid).any(axis=1)
        resid = resid[mask]
        if resid.shape[0] < 30:
            raise ValueError(f"too few clean residuals: {resid.shape}")

        # 3) Fit copula
        self.R_, self.sorted_resid_ = fit_gaussian_copula(resid)

        # 4) Store conditional means at h=1
        self.cond_means_ = np.array(
            [self.per_asset_[a].cond_mean_ for a in self.assets_],
            dtype=float,
        )
        return self

    def predict(self, n_draws: int | None = None) -> pd.DataFrame:
        if self.R_ is None:
            raise RuntimeError("fit() before predict()")
        n = n_draws or self.n_draws
        joint = sample_copula(self.R_, self.sorted_resid_, n, self.rng)
        draws = joint + self.cond_means_[None, :]
        return pd.DataFrame(draws, columns=self.assets_)
