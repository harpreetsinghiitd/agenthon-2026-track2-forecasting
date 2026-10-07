"""Executive summary: adjust marginal spread using fitted GARCH volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statistical_layer.arima import _find_column, _horizon_steps


def transform(draws: np.ndarray, panel: pd.DataFrame, card: dict, asof: str,
              seed: int = 51) -> np.ndarray:
    """Rescale each marginal to its GARCH forecast volatility; preserve shape."""
    try:
        from arch import arch_model
    except ImportError as exc:
        raise RuntimeError("GARCH transform requires the `arch` package") from exc
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
            innovations = series.diff().dropna().to_numpy(dtype=float)
        if len(innovations) < 40:
            raise ValueError(f"too little history to fit GARCH for {asset!r}")
        fitted = arch_model(innovations, mean="Zero", vol="GARCH", p=1, q=1,
                            dist="skewt", rescale=False).fit(disp="off", show_warning=False)
        forecast = fitted.forecast(horizon=int(steps[ai].max()), reindex=False)
        variances = np.asarray(forecast.variance.iloc[-1], dtype=float) / 10000.0
        for hi, step in enumerate(steps[ai]):
            desired_sd = float(np.sqrt(np.maximum(variances[:int(step)], 0).sum()))
            if target_type == "level":
                # A level draw's uncertainty is centered around its sample mean.
                pass
            sample = out[:, ai, hi]
            center = float(np.mean(sample))
            current_sd = float(np.std(sample, ddof=1))
            if np.isfinite(desired_sd) and desired_sd > 0 and current_sd > 1e-12:
                out[:, ai, hi] = center + (sample - center) * desired_sd / current_sd
    return out
