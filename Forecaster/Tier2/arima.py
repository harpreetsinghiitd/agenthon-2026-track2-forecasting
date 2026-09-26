"""Self-contained per-asset ARIMA. No project imports."""
from __future__ import annotations
import warnings
import numpy as np


class ArimaAsset:
    """
    Fit ARIMA(p,d,q) on a single 1D series via AIC order search.

    Public attributes after .fit(y):
        cond_mean_ : float         1-step-ahead conditional mean
        resid_     : np.ndarray    in-sample residuals
        fitted_    : np.ndarray    in-sample fitted values
        order_     : (p, d, q)
        model_     : fitted statsmodels ARIMAResults

    Public method:
        forecast(steps) -> (means, std_errors), each shape (steps,)
    """

    def __init__(self, max_p: int = 3, max_d: int = 1, max_q: int = 3,
                 ic: str = "aic", maxiter: int = 200):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.ic = ic
        self.maxiter = maxiter

    def fit(self, y: np.ndarray) -> "ArimaAsset":
        from statsmodels.tsa.arima.model import ARIMA

        y = np.asarray(y, dtype=float)
        y = y[~np.isnan(y)]
        if len(y) < 20:
            raise ValueError(f"series too short: len={len(y)}")

        best_ic = np.inf
        best_model = None
        best_order = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(self.max_p + 1):
                for d in range(self.max_d + 1):
                    for q in range(self.max_q + 1):
                        if p == 0 and q == 0:
                            continue
                        try:
                            m = ARIMA(
                                y, order=(p, d, q),
                                enforce_stationarity=True,
                                enforce_invertibility=True,
                            ).fit(method_kwargs={"maxiter": self.maxiter})
                        except Exception:
                            continue
                        ic_val = getattr(m, self.ic.lower())
                        if np.isfinite(ic_val) and ic_val < best_ic:
                            best_ic = ic_val
                            best_model = m
                            best_order = (p, d, q)

        if best_model is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_model = ARIMA(y, order=(0, 1, 0)).fit()
            best_order = (0, 1, 0)

        self.model_ = best_model
        self.order_ = best_order
        self.resid_ = np.asarray(best_model.resid, dtype=float)
        self.fitted_ = np.asarray(best_model.fittedvalues, dtype=float)

        try:
            fc = best_model.get_forecast(steps=1)
            self.cond_mean_ = float(np.asarray(fc.predicted_mean).ravel()[0])
        except Exception:
            self.cond_mean_ = float(y[-1])
        return self

    def forecast(self, steps: int) -> tuple[np.ndarray, np.ndarray]:
        fc = self.model_.get_forecast(steps=steps)
        means = np.asarray(fc.predicted_mean, dtype=float).ravel()
        ses = np.asarray(fc.se_mean, dtype=float).ravel()
        if means.shape[0] < steps:
            pad = steps - means.shape[0]
            means = np.concatenate([means, np.full(pad, means[-1])])
            ses = np.concatenate([ses, np.full(pad, ses[-1])])
        return means, ses
