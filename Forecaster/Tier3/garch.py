"""GARCH(1,1) with skewed-t innovations. Wraps the `arch` package.

Self-contained. No project imports.
"""
from __future__ import annotations
import warnings
import numpy as np
from arch import arch_model
from arch.univariate.distribution import SkewStudent


class GarchAsset:
    """
    Fit GARCH(p,q) with skewed-t innovations on a 1D residual series.

    Public after .fit(resid):
        sigma_  : np.ndarray   conditional volatility series
        z_      : np.ndarray   standardized residuals  z = resid / sigma_
        eta_    : float        skewed-t degrees of freedom
        lambda_ : float        skewed-t skewness in (-1, 1)
        res_    : arch results object

    Public methods:
        forecast_cum_vol(horizons) -> ndarray, cumulative vol at each horizon
        skewt_ppf(u)               -> ndarray, inverse skewed-t CDF
        skewt_cdf(z)               -> ndarray, skewed-t CDF
    """

    def __init__(self, p: int = 1, q: int = 1, maxiter: int = 500):
        self.p = p
        self.q = q
        self.maxiter = maxiter

    def fit(self, resid: np.ndarray) -> "GarchAsset":
        resid = np.asarray(resid, dtype=float)
        resid = resid[~np.isnan(resid)]
        if len(resid) < 50:
            raise ValueError(f"residual too short for GARCH: len={len(resid)}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                resid, vol="GARCH", p=self.p, q=self.q,
                dist="skewt", rescale=False,
            )
            self.res_ = am.fit(
                disp="off", show_warning=False,
                options={"maxiter": self.maxiter},
            )

        self.sigma_ = np.asarray(self.res_.conditional_volatility, dtype=float)
        self.z_ = resid / np.maximum(self.sigma_, 1e-12)
        self.eta_ = float(self.res_.params.get("eta", 5.0))
        self.lambda_ = float(self.res_.params.get("lambda", 0.0))
        self._dist = SkewStudent()
        return self

    def forecast_cum_vol(self, horizons: list[int]) -> np.ndarray:
        """sqrt(cumulative variance up to h) at each horizon h."""
        max_h = int(max(horizons))
        f = self.res_.forecast(horizon=max_h, reindex=False)
        var_path = np.asarray(f.variance.values[0], dtype=float)  # length max_h
        cum_var = np.cumsum(var_path)
        return np.array([np.sqrt(cum_var[h - 1]) for h in horizons], dtype=float)

    def skewt_ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 1e-6, 1.0 - 1e-6)
        out = self._dist.ppf(u, [self.eta_, self.lambda_])
        return np.asarray(out, dtype=float)

    def skewt_cdf(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        out = self._dist.cdf(z, [self.eta_, self.lambda_])
        return np.asarray(out, dtype=float)
