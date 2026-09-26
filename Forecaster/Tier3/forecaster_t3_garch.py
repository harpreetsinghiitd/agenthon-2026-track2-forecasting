"""Tier 3 forecaster: ARIMA mean + GARCH skewed-t vol + Gaussian copula."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

from Forecaster.Tier3.arima import ArimaAsset
from Forecaster.Tier3.garch import GarchAsset
from Forecaster.Tier3.copula import fit_copula_R, sample_copula_uniform


class GarchCopulaForecaster:
    def __init__(self, n_draws: int = 500, seed: int = 42):
        self.n_draws = n_draws
        self.rng = np.random.default_rng(seed)
        self.assets_: list[str] = []
        self.horizons_: list[int] = []
        self.arima_: dict[str, ArimaAsset] = {}
        self.garch_: dict[str, GarchAsset | None] = {}
        self.R_: np.ndarray | None = None
        self.means_: np.ndarray | None = None   # (K, H)
        self.vols_: np.ndarray | None = None    # (K, H)
        self._garch_assets: list[str] = []

    def fit(self, wide: pd.DataFrame,
            horizons: list[int]) -> "GarchCopulaForecaster":
        self.assets_ = list(wide.columns)
        self.horizons_ = list(horizons)
        K = len(self.assets_)
        H = len(self.horizons_)
        if K == 0:
            raise ValueError("empty wide panel")

        self.means_ = np.zeros((K, H))
        self.vols_ = np.zeros((K, H))

        for k, a in enumerate(self.assets_):
            y = wide[a].to_numpy(dtype=float)

            arima = ArimaAsset().fit(y)
            self.arima_[a] = arima

            try:
                garch = GarchAsset().fit(arima.resid_)
                vols = garch.forecast_cum_vol(horizons)
            except Exception:
                garch = None
                _, ses = arima.forecast(max(horizons))
                vols = np.array([ses[h - 1] for h in horizons], dtype=float)
            self.garch_[a] = garch

            means, _ = arima.forecast(max(horizons))
            for hi, h in enumerate(horizons):
                self.means_[k, hi] = means[h - 1]
                self.vols_[k, hi] = vols[hi]

        # Copula on standardized residuals of the GARCH-successful assets
        garch_assets = [a for a in self.assets_ if self.garch_[a] is not None]
        if len(garch_assets) < 1:
            raise ValueError("no asset produced GARCH residuals")

        min_len = min(len(self.garch_[a].z_) for a in garch_assets)
        if min_len < 30:
            raise ValueError(f"too few aligned z residuals: {min_len}")

        Z = np.column_stack([self.garch_[a].z_[-min_len:] for a in garch_assets])
        mask = ~np.isnan(Z).any(axis=1)
        Z = Z[mask]
        if Z.shape[0] < 30:
            raise ValueError(f"too few clean z rows: {Z.shape}")

        self.R_ = fit_copula_R(Z)
        self._garch_assets = garch_assets
        return self

    def predict(self, n_draws: int | None = None) -> np.ndarray:
        """Returns (n_draws, K, H)."""
        if self.R_ is None:
            raise RuntimeError("fit() before predict()")

        n = n_draws or self.n_draws
        K = len(self.assets_)
        H = len(self.horizons_)
        draws = np.zeros((n, K, H), dtype=float)

        garch_idx = [self.assets_.index(a) for a in self._garch_assets]

        for hi in range(H):
            z_star = np.zeros((n, K), dtype=float)

            # Copula-based draws for GARCH-successful assets
            U = sample_copula_uniform(self.R_, n, self.rng)
            for j, ai in enumerate(garch_idx):
                a = self.assets_[ai]
                z_star[:, ai] = self.garch_[a].skewt_ppf(U[:, j])

            # Independent standard normal for fallback assets
            for ai in range(K):
                if ai in garch_idx:
                    continue
                z_star[:, ai] = self.rng.standard_normal(n)

            draws[:, :, hi] = (z_star * self.vols_[:, hi][None, :]
                               + self.means_[:, hi][None, :])

        return draws

    def predict_df(self, n_draws: int | None = None) -> pd.DataFrame:
        """Flat (n_draws, K) DataFrame at the first horizon, for local checks."""
        return pd.DataFrame(self.predict(n_draws), columns=self.assets_)
