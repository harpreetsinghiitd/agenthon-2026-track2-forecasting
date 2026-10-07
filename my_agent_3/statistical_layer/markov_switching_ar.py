"""Executive summary: two-state Markov-switching autoregression sampler."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .arima import _find_column, _horizon_steps


def fit_and_sample(panel: pd.DataFrame, card: dict, asof: str, n_draws: int,
                   seed: int = 46) -> np.ndarray:
    """Fit a simple two-regime AR using volatility-clustered historical states.

    Regime-specific AR coefficients, innovation scales, and transition
    probabilities are estimated from historical observations. This lightweight
    implementation avoids a fragile nonlinear optimizer in the fallback path.
    """
    targets = card["targets"]
    assets = [str(x) for x in targets["asset_ids"]]
    horizons = [int(x) for x in targets["horizons"]]
    target_type = targets["target_type"]
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.loc[data["date"] <= pd.Timestamp(asof)]
    steps = _horizon_steps(data, card, asof, assets, horizons)
    max_steps = int(steps.max())
    wide = data.pivot_table(index="date", columns="asset", values="value").sort_index()
    out = np.empty((n_draws, len(assets), len(horizons)), dtype=float)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(assets))]
    for ai, asset in enumerate(assets):
        y = wide[_find_column(wide, asset)].dropna().to_numpy(dtype=float)
        if target_type == "log_return":
            if np.any(y <= -1):
                raise ValueError(f"asset {asset!r} contains returns <= -100%")
            y = np.log1p(y)
            cumulative = True
        elif target_type == "level":
            cumulative = False
        else:
            raise ValueError(f"unsupported target_type {target_type!r}")
        if len(y) < 40:
            raise ValueError("Markov-switching AR needs at least 40 observations")
        residual_scale = np.abs(np.diff(y))
        threshold = np.quantile(residual_scale, 0.65)
        state = np.zeros(len(y) - 1, dtype=int)
        state[residual_scale > threshold] = 1
        transition = np.ones((2, 2), dtype=float)
        for a, b in zip(state[:-1], state[1:]):
            transition[a, b] += 1
        transition /= transition.sum(axis=1, keepdims=True)
        coeffs, intercepts, sigmas = [], [], []
        dy = np.diff(y)
        for regime in range(2):
            selected = state == regime
            x = y[:-1][selected]
            z = dy[selected]
            if len(z) < 5:
                x, z = y[:-1], dy
            design = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(design, z, rcond=None)
            errors = z - design @ beta
            coeffs.append(float(beta[1]))
            intercepts.append(float(beta[0]))
            sigmas.append(max(float(np.std(errors, ddof=1)), 1e-8))
        state_now = int(state[-1])
        paths = np.empty((n_draws, max_steps), dtype=float)
        current = np.full(n_draws, float(y[-1]))
        for t in range(max_steps):
            states = np.empty(n_draws, dtype=int)
            if t == 0:
                states[:] = state_now
            else:
                switch_prob = transition[prev_states, 1]
                states = (rngs[ai].random(n_draws) < switch_prob).astype(int)
            delta = np.asarray(intercepts)[states] + np.asarray(coeffs)[states] * current
            if cumulative:
                current = delta + rngs[ai].normal(size=n_draws) * np.asarray(sigmas)[states]
                paths[:, t] = current
            else:
                current = current + delta + rngs[ai].normal(size=n_draws) * np.asarray(sigmas)[states]
                paths[:, t] = current
            prev_states = states
        for hi, step in enumerate(steps[ai]):
            out[:, ai, hi] = paths[:, :int(step)].sum(axis=1) if cumulative else paths[:, int(step)-1]
    return out
