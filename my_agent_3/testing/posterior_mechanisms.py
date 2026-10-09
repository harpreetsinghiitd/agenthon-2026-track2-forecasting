"""Posterior-generation mechanisms: every mathematical assumption for turning
the elicited signal vector into posterior draws.

Each mechanism is a pure function ``(prior_draws, bounded_signal, seed) ->
posterior_draws`` consuming the SAME production signal schema
(``llm.signals.NEUTRAL`` keys and bounds). Four assumption families:

* parametric — the production conditioners (shift / scale / tail stretch /
  common-shock blend) in isolation, orders, and combinations;
* tilting — Meucci-style entropy pooling: adjustment by probability mass on
  the prior draws (location view; location+dispersion; ESS trust region);
* mixture — scenario re-composition: multi-modal posteriors built from the
  signal (2-scenario hedge, 3-scenario with an amplified tail branch);
* mapping — asymmetric quantile warping and shock injection.

Signal-interpretation constants (hedge weights, amplifications, displacements)
are the mathematical assumptions under test; they are documented per
mechanism and fixed across the benchmark.

Determinism: every stochastic step (assignment, resampling) takes the seed
the harness derives per (card, origin); tilting solves are deterministic
Newton iterations from a zero start.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from llm import copula_adjustment, mean_shift, signals, tail_skew, variance_scale  # noqa: E402

__all__ = ["MECHANISMS", "apply_mechanism", "ess_of_weights"]

# ---------------------------------------------------------------------------
# Entropy pooling (exponential tilting) per asset x horizon cell
# ---------------------------------------------------------------------------


def _cell(draws: np.ndarray, ai: int, hi: int) -> np.ndarray:
    return draws[:, ai, hi].astype(float)


def _solve_location(u: np.ndarray, target_mean: float,
                     max_iter: int = 60) -> float:
    """1-D dual solve: weights w ∝ exp(lam*u) with E_w[u] = target_mean.

    ``u`` is standardized (mean 0, sd 1) per cell; the dual is convex and the
    Newton step is monotone in lam, so a zero start converges deterministically.
    """
    lam = 0.0
    for _ in range(max_iter):
        z = lam * u
        w = np.exp(z - z.max())
        w /= w.sum()
        m = float(w @ u)
        var = float(w @ (u * u)) - m * m
        step = (m - target_mean) / max(var, 1e-8)
        lam -= step
        if abs(step) < 1e-10:
            break
    return lam


def _solve_loc_disp(u: np.ndarray, target_mean: float, target_second: float,
                    max_iter: int = 80) -> np.ndarray:
    """2-D dual solve on features (u, u^2): match E[u] and E[u^2].

    Returns the dual-optimal lambda (2,) or None when the Hessian is
    degenerate (caller falls back to the location-only solve).
    """
    feats = np.vstack([u, u * u])          # (2, n)
    c = np.array([target_mean, target_second])
    lam = np.zeros(2)
    for _ in range(max_iter):
        z = lam @ feats
        w = np.exp(z - z.max())
        w /= w.sum()
        ew = feats @ w                      # (2,) weighted feature means
        grad = ew - c
        cov = (feats * w) @ feats.T - np.outer(ew, ew)
        try:
            step = np.linalg.solve(cov + 1e-9 * np.eye(2), grad)
        except np.linalg.LinAlgError:
            return None
        lam -= step
        if np.abs(step).max() < 1e-10:
            break
    return lam


def _weights(feats: np.ndarray, lam: np.ndarray) -> np.ndarray:
    z = lam @ feats
    w = np.exp(z - z.max())
    return w / w.sum()


def ess_of_weights(w: np.ndarray) -> float:
    """Effective sample size of an importance-weight vector."""
    return float(1.0 / np.sum(w * w))


def _shrink_to_ess(feats: np.ndarray, lam: np.ndarray, floor: float) -> np.ndarray:
    """Trust region: scale lambda toward zero until ESS >= floor.

    Binary search on the scale in [0, 1]; lam = 0 has ESS = n_draws, and ESS
    is monotone decreasing in |lam| along any ray, so the search is exact.
    """
    n = feats.shape[1]
    if ess_of_weights(_weights(feats, lam)) >= floor or n < floor:
        return lam
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ess_of_weights(_weights(feats, mid * lam)) >= floor:
            lo = mid
        else:
            hi = mid
    return lo * lam


def _pool(draws: np.ndarray, signal: dict, seed: int, *, dispersion: bool,
          ess_floor: float | None) -> np.ndarray:
    """Entropy-pool every cell toward the signal's implied moment views."""
    out = np.asarray(draws, dtype=float).copy()
    n = out.shape[0]
    rng = np.random.default_rng(seed)
    shift = float(signal.get("mean_shift_sigma", 0.0))
    scale = float(signal.get("variance_scale", 1.0))
    if shift == 0.0 and scale == 1.0:
        return out
    for ai in range(out.shape[1]):
        for hi in range(out.shape[2]):
            x = _cell(out, ai, hi)
            mu, sd = float(x.mean()), float(x.std())
            if sd < 1e-12:
                continue
            u = (x - mu) / sd
            feats = np.vstack([u, u * u])
            if dispersion:
                lam = _solve_loc_disp(u, shift, max(scale * scale, 1e-6))
                if lam is None:
                    lam = np.array([_solve_location(u, shift), 0.0])
            else:
                lam = np.array([_solve_location(u, shift), 0.0])
            if ess_floor is not None:
                lam = _shrink_to_ess(feats, lam, ess_floor)
            w = _weights(feats, lam)
            if not np.isfinite(w).all() or w.sum() <= 0:
                continue
            idx = rng.choice(n, size=n, p=w)
            out[:, ai, hi] = x[idx]
    return out


# ---------------------------------------------------------------------------
# Scenario mixtures
# ---------------------------------------------------------------------------


def _mixture2(draws, signal: dict, seed: int):
    """Two-scenario hedge: 70% of draws take the signal's drift, 30% ignore it.

    Constants: hedge weight 0.3 fixed. The implied centre moves by only
    0.7 x shift -- the price of humility -- while the draw set stays
    bimodal. That trade (weaker centring vs. honest spread) is the assumption
    under test.
    """
    shift = float(signal.get("mean_shift_sigma", 0.0))
    if shift == 0.0:
        return draws
    out = np.asarray(draws, dtype=float).copy()
    sd = out.std(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    take = rng.random(out.shape[0]) < 0.7
    out[take] = out[take] + shift * sd
    return out


def _mixture3(draws, signal: dict, seed: int):
    """Three scenarios: 60% signal drift, 20% amplified tail branch,
    20% counter-move.

    Constants: tail branch amplified 2x the tail strength in the signal's
    tail direction; counter-branch at half the drift, opposite sign. Models
    the playbook's branching-outcome posteriors (hike vs hold).
    """
    shift = float(signal.get("mean_shift_sigma", 0.0))
    direction = str(signal.get("tail_direction", "none"))
    strength = float(signal.get("tail_strength", 0.0))
    if shift == 0.0 and strength <= 0.0:
        return draws
    out = np.asarray(draws, dtype=float).copy()
    sd = out.std(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    r = rng.random(out.shape[0])
    centre = r < 0.6
    tail = (r >= 0.6) & (r < 0.8)
    counter = r >= 0.8
    out[centre] = out[centre] + shift * sd
    if strength > 0.0 and direction in ("left", "right"):
        sign = -1.0 if direction == "left" else 1.0
        out[tail] = out[tail] + sign * 2.0 * strength * sd
    out[counter] = out[counter] - 0.5 * shift * sd
    return out


# ---------------------------------------------------------------------------
# Shape mappings
# ---------------------------------------------------------------------------


def _quantile_warp(draws, signal: dict):
    """Asymmetric half-distribution scaling: widen ONLY the half the signal's
    tail direction points at, by variance_scale, around the median.

    Differs from the tail-stretch conditioner (outer decile only) and from
    symmetric variance scaling (both halves): the whole indicated half moves.
    """
    scale = float(signal.get("variance_scale", 1.0))
    direction = str(signal.get("tail_direction", "none"))
    if scale == 1.0 or direction not in ("left", "right"):
        return draws
    out = np.asarray(draws, dtype=float).copy()
    med = np.median(out, axis=0, keepdims=True)
    mask = out < med if direction == "left" else out > med
    stretched = med + (out - med) * scale
    return np.where(mask, stretched, out)


def _shock_inject(draws, signal: dict, seed: int):
    """Displace a shock_probability fraction of draws by 1 sd in the tail
    direction (seeded selection).

    Constants: displacement magnitude 1 sd; the fraction is the signal's own
    shock_probability, so the net centre move is p x sd -- inside the +-0.5 sd
    envelope for any oracle-calibrated p.
    """
    p = float(signal.get("shock_probability", 0.0))
    direction = str(signal.get("tail_direction", "none"))
    if p <= 0.0 or direction not in ("left", "right"):
        return draws
    out = np.asarray(draws, dtype=float).copy()
    sd = out.std(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    hit = rng.random(out.shape[0]) < min(p, 1.0)
    sign = -1.0 if direction == "left" else 1.0
    out[hit] = out[hit] + sign * sd
    return out


def _affine_mean(draws, signal: dict):
    """One-step affine moment matching around the MEAN (vs. the production
    median centring): draws' = mean + (draws - mean) x scale + shift x sd."""
    shift = float(signal.get("mean_shift_sigma", 0.0))
    scale = float(signal.get("variance_scale", 1.0))
    if shift == 0.0 and scale == 1.0:
        return draws
    out = np.asarray(draws, dtype=float).copy()
    mean = out.mean(axis=0, keepdims=True)
    sd = out.std(axis=0, keepdims=True)
    out = mean + (out - mean) * scale + shift * np.where(sd > 0, sd, 0.0)
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


MECHANISMS = {
    # baselines
    "none": lambda d, s, seed: d,
    # parametric family (production conditioners)
    "shift": lambda d, s, seed: mean_shift.apply(d, s),
    "scale": lambda d, s, seed: variance_scale.apply(d, s),
    "tail": lambda d, s, seed: tail_skew.apply(d, s, seed=seed),
    "shift_then_scale": lambda d, s, seed: variance_scale.apply(
        mean_shift.apply(d, s), s),
    "scale_then_shift": lambda d, s, seed: mean_shift.apply(
        variance_scale.apply(d, s), s),
    "affine_mean_centred": lambda d, s, seed: _affine_mean(d, s),
    "full_param": lambda d, s, seed: tail_skew.apply(
        variance_scale.apply(mean_shift.apply(d, s), s), s, seed=seed),
    "full_param_joint": lambda d, s, seed: copula_adjustment.apply(
        tail_skew.apply(variance_scale.apply(mean_shift.apply(d, s), s), s,
                        seed=seed), None, {}, s, seed=seed),
    # tilting family (entropy pooling)
    "tilt_loc": lambda d, s, seed: _pool(d, s, seed, dispersion=False,
                                         ess_floor=None),
    "tilt_loc_disp": lambda d, s, seed: _pool(d, s, seed, dispersion=True,
                                              ess_floor=None),
    "tilt_loc_disp_ess": lambda d, s, seed: _pool(d, s, seed, dispersion=True,
                                                   ess_floor=100.0),
    # mixture family
    "mixture2": _mixture2,
    "mixture3": _mixture3,
    "mixture3_scale": lambda d, s, seed: variance_scale.apply(
        _mixture3(d, s, seed), s),
    # mapping family
    "quantile_warp": lambda d, s, seed: _quantile_warp(d, s),
    "shock_inject": _shock_inject,
}

#: Group labels for the report.
FAMILY_OF = {
    "none": "baseline",
    "shift": "parametric", "scale": "parametric", "tail": "parametric",
    "shift_then_scale": "parametric", "scale_then_shift": "parametric",
    "affine_mean_centred": "parametric", "full_param": "parametric",
    "full_param_joint": "parametric",
    "tilt_loc": "tilting", "tilt_loc_disp": "tilting",
    "tilt_loc_disp_ess": "tilting",
    "mixture2": "mixture", "mixture3": "mixture", "mixture3_scale": "mixture",
    "quantile_warp": "mapping", "shock_inject": "mapping",
}


def apply_mechanism(name: str, draws: np.ndarray, signal: dict,
                    seed: int) -> np.ndarray:
    """Apply one mechanism; never raises past the prior (returns it on error)."""
    try:
        out = MECHANISMS[name](draws, signal, seed)
        out = np.asarray(out, dtype=float)
        if out.shape != draws.shape or not np.isfinite(out).all():
            return draws
        return out
    except Exception:
        return draws
