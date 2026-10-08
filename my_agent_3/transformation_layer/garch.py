"""Executive summary: adjust marginal spread using fitted GARCH volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statistical_layer.arima import _find_column, _horizon_steps


def transform(draws: np.ndarray, panel: pd.DataFrame, card: dict, asof: str,
              seed: int = 51, garch_weight: float = 0.5,
              scale_bounds: tuple[float, float] = (0.5, 2.0),
              ewma_decay: float = 0.94) -> np.ndarray:
    """Blend conditional and empirical horizon variance with bounded changes.

    Uses GARCH when ``arch`` is installed and fits successfully, otherwise a
    dependency-free EWMA variance forecast. ``garch_weight`` shrinks the
    conditional scale ratio toward one to prevent unstable fits from
    dominating the baseline draws.
    """
    if not 0.0 <= garch_weight <= 1.0:
        raise ValueError("garch_weight must be between zero and one")
    lower, upper = map(float, scale_bounds)
    if lower <= 0 or upper < lower:
        raise ValueError("scale_bounds must be positive and ordered")
    if not 0.0 < ewma_decay < 1.0:
        raise ValueError("ewma_decay must be between zero and one")
    try:
        from arch import arch_model
    except ImportError:
        arch_model = None
    out = np.asarray(draws, dtype=float).copy()
    targets = card["targets"]
    assets = [str(a) for a in targets["asset_ids"]]
    horizons = [int(h) for h in targets["horizons"]]
    target_type = targets["target_type"]
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.loc[data["date"] <= pd.Timestamp(asof)]
    steps = _horizon_steps(data, card, asof, assets, horizons)
    wide = data.pivot_table(index="date", columns="asset", values="value").sort_index()
    for ai, asset in enumerate(assets):
        series = wide[_find_column(wide, asset)].dropna().astype(float)
        if target_type == "log_return":
            if (series <= -1).any():
                raise ValueError(f"asset {asset!r} contains returns <= -100%")
            innovations = np.log1p(series).to_numpy() * 100.0
        else:
            innovations = series.diff().dropna().to_numpy(dtype=float) * 100.0
        if len(innovations) < 40:
            raise ValueError(f"too little history to fit GARCH for {asset!r}")
        baseline_variance = float(np.var(innovations / 100.0, ddof=1))
        variances = None
        if arch_model is not None:
            try:
                fitted = arch_model(innovations, mean="Zero", vol="GARCH", p=1, q=1,
                                    dist="skewt", rescale=False).fit(
                                        disp="off", show_warning=False)
                forecast = fitted.forecast(horizon=int(steps[ai].max()), reindex=False)
                variances = np.asarray(forecast.variance.iloc[-1], dtype=float) / 10000.0
            except Exception:
                variances = None
        if variances is None:
            # EWMA variance with mean reversion to the expanding sample variance.
            normalized = innovations / 100.0
            center = float(np.mean(normalized))
            residuals = normalized - center
            long_run = max(baseline_variance, 1e-15)
            ewma_var = long_run
            for residual in residuals:
                ewma_var = ewma_decay * ewma_var + (1.0 - ewma_decay) * float(residual**2)
            horizons_ahead = np.arange(1, int(steps[ai].max()) + 1, dtype=float)
            variances = long_run + (ewma_var - long_run) * (ewma_decay ** horizons_ahead)
        for hi, step in enumerate(steps[ai]):
            desired_sd = float(np.sqrt(np.maximum(variances[:int(step)], 0).sum()))
            baseline_sd = float(np.sqrt(max(baseline_variance, 0.0) * int(step)))
            if baseline_sd > 0 and np.isfinite(desired_sd):
                raw_ratio = desired_sd / baseline_sd
                bounded_ratio = float(np.clip(raw_ratio, lower, upper))
                desired_sd = baseline_sd * (1.0 + garch_weight * (bounded_ratio - 1.0))
            sample = out[:, ai, hi]
            center = float(np.mean(sample))
            current_sd = float(np.std(sample, ddof=1))
            if np.isfinite(desired_sd) and desired_sd > 0 and current_sd > 1e-12:
                out[:, ai, hi] = center + (sample - center) * desired_sd / current_sd
    return out
