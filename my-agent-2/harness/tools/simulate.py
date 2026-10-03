"""Draw construction — innovations plus the deterministic overlay.

Role in the pipeline
--------------------
``simulate_mixture`` is where per-step innovations became candidate forecast
paths.  Every number that leaves this module is a deterministic function of its
inputs plus the provided seed; the LLM may later *choose the overlay* (drift,
width, skew per draw) but can never touch the innovations or the accumulator,
which preserves the one rule: no model-written forecast values.

Shapes
------
``innovations`` is ``(n_draws, n_steps)`` of per-step changes.  The output is
the *cumulative path offset* with the same shape: entry ``[d, t]`` is the
change from the anchor after ``t + 1`` steps.  ``agent.main`` turns offsets
into level values; for a ``log_return`` card the offset is itself the answer.

Seed
----
`seed` shuffles which bootstrap path receives which overlay row.  Phase 2's
overlay is constant, so the shuffle is a no-op in distribution; Phase 3's
per-draw scenario overlays make the pairing meaningful.  Keeping it here from
the start keeps the signature — and determinism contract — stable.

Phase 4: Joint simulation
-------------------------
``simulate_mixture_joint`` takes per-asset innovations and a copula correlation
matrix, and produces correlated multi-asset paths using the M0 trick
(cross-horizon dependence via ``min(s_i, s_j)`` accumulation on the
text-conditioned mean).
"""

from __future__ import annotations

import zlib

import numpy as np

__all__ = [
    "SimulateError",
    "seed_from_unit_id",
    "simulate_mixture",
    "simulate_mixture_joint",
]


class SimulateError(ValueError):
    """Raised when innovations or overlays carry unusable shapes or values."""


def seed_from_unit_id(unit_id: str) -> int:
    """Derive the pipeline's RNG seed from the card identity.

    `crc32(unit_id) & 0x7FFFFFFF` is the contract from the build prompt: the
    same card always drives the same draws, which is what makes re-runs
    byte-identical.
    """
    return zlib.crc32(unit_id.encode("utf-8")) & 0x7FFFFFFF


def simulate_mixture(
    innovations: np.ndarray,
    drift_per_draw: np.ndarray | float,
    width_mult: np.ndarray | float,
    seed: int,
) -> np.ndarray:
    """Combine bootstrap innovations with a per-draw overlay.

    Args:
        innovations: Demeaned per-step changes, shape ``(n_draws, n_steps)``.
        drift_per_draw: Per-step drift added to every step of a draw; a scalar
            applies the same drift to all rows, a ``(n_draws,)`` vector varies
            it per draw (Phase 3's scenario weights).
        width_mult: Multiplier on the innovation magnitude; scalar or
            ``(n_draws,)``.  Zero collapses a draw to its drift line.
        seed: Shuffles the pairing of innovation rows to overlay rows.

    Returns:
        Cumulative path offsets, shape ``(n_draws, n_steps)``, in the panel's
        own units.

    Raises:
        SimulateError: shape mismatch, non-finite values, or negative width.
    """
    innov = np.asarray(innovations, dtype=float)
    if innov.ndim != 2:
        raise SimulateError(f"innovations must be (n_draws, n_steps), got {innov.ndim}-D")
    n_draws, n_steps = innov.shape
    if not np.isfinite(innov).all():
        raise SimulateError("innovations contain NaN or infinity")

    drift = np.asarray(drift_per_draw, dtype=float)
    if drift.ndim == 0:
        drift = np.full(n_draws, float(drift))
    elif drift.shape == (n_draws,):
        pass
    else:
        raise SimulateError(
            f"drift_per_draw must be scalar or ({n_draws},), got shape {drift.shape}"
        )
    if not np.isfinite(drift).all():
        raise SimulateError("drift_per_draw contains NaN or infinity")

    width = np.asarray(width_mult, dtype=float)
    if width.ndim == 0:
        width = np.full(n_draws, float(width))
    elif width.shape == (n_draws,):
        pass
    else:
        raise SimulateError(
            f"width_mult must be scalar or ({n_draws},), got shape {width.shape}"
        )
    if not np.isfinite(width).all():
        raise SimulateError("width_mult contains NaN or infinity")
    if (width < 0).any():
        raise SimulateError("width_mult must be non-negative")

    order = np.random.default_rng(int(seed)).permutation(n_draws)
    mixed = innov[order]

    steps = mixed * width[:, None] + drift[:, None]
    return np.cumsum(steps, axis=1)


def _m0_trick(corr: np.ndarray, n_steps: int) -> np.ndarray:
    """Cross-horizon dependence via the M0 trick.

    For a correlation matrix `corr`, the cross-horizon correlation between
    step `i` and step `j` is `corr * min(i+1, j+1) / sqrt((i+1)*(j+1))`.
    This gives the covariance matrix for the cumulative path offsets.

    Args:
        corr: Correlation matrix ``(n_assets, n_assets)``.
        n_steps: Number of steps (max horizon).

    Returns:
        Covariance matrix of shape ``(n_assets * n_steps, n_assets * n_steps)``.
    """
    n_assets = corr.shape[0]
    step_idx = np.arange(n_steps)
    cov = np.zeros((n_assets * n_steps, n_assets * n_steps))
    for a in range(n_assets):
        for b in range(n_assets):
            rab = corr[a, b]
            min_idx = np.minimum(step_idx[:, None], step_idx[None, :])
            denom = np.sqrt((step_idx[:, None] + 1) * (step_idx[None, :] + 1))
            cross = rab * min_idx / denom
            cov[a * n_steps : (a + 1) * n_steps, b * n_steps : (b + 1) * n_steps] = cross
    return cov


def simulate_mixture_joint(
    innovations_per_asset: list[np.ndarray],
    drift_per_asset: list[np.ndarray | float],
    width_per_asset: list[np.ndarray | float],
    corr: np.ndarray,
    seed: int,
) -> list[np.ndarray]:
    """Joint simulation across assets with copula-induced correlation.

    This implements the M0 trick: one correlated shock drives all assets in a
    draw, with cross-horizon dependence via min(s_i, s_j) accumulation.

    Args:
        innovations_per_asset: List of ``(n_draws, n_steps)`` bootstrap
            innovations, one per asset.  All must have same shape.
        drift_per_asset: Per-asset drift (scalar or ``(n_draws,)``).
        width_per_asset: Per-asset width multiplier (scalar or ``(n_draws,)``).
        corr: Asset correlation matrix ``(n_assets, n_assets)`` from copula.
        seed: Master seed for the draw.

    Returns:
        List of ``(n_draws, n_steps)`` cumulative offsets, one per asset.

    Raises:
        SimulateError: shape mismatch, non-finite values, non-PD correlation.
    """
    if not innovations_per_asset:
        raise SimulateError("innovations_per_asset must be non-empty")
    n_assets = len(innovations_per_asset)
    n_draws, n_steps = innovations_per_asset[0].shape

    if corr.shape != (n_assets, n_assets):
        raise SimulateError(f"corr shape {corr.shape} != ({n_assets}, {n_assets})")
    if len(drift_per_asset) != n_assets or len(width_per_asset) != n_assets:
        raise SimulateError("drift/width lists must match n_assets")

    for a, innov in enumerate(innovations_per_asset):
        if innov.shape != (n_draws, n_steps):
            raise SimulateError(f"asset {a}: innovations shape {innov.shape} != ({n_draws}, {n_steps})")
        if not np.isfinite(innov).all():
            raise SimulateError(f"asset {a}: innovations contain NaN or infinity")

    drifts = []
    widths = []
    for a in range(n_assets):
        drift = np.asarray(drift_per_asset[a], dtype=float)
        if drift.ndim == 0:
            drift = np.full(n_draws, float(drift))
        elif drift.shape == (n_draws,):
            pass
        else:
            raise SimulateError(f"asset {a}: drift_per_draw shape {drift.shape}")
        drifts.append(drift)

        width = np.asarray(width_per_asset[a], dtype=float)
        if width.ndim == 0:
            width = np.full(n_draws, float(width))
        elif width.shape == (n_draws,):
            pass
        else:
            raise SimulateError(f"asset {a}: width_mult shape {width.shape}")
        widths.append(width)

    cov = _m0_trick(corr, n_steps)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise SimulateError("correlation matrix not positive definite")

    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal(size=(n_draws, n_assets * n_steps))
    shocks = z @ chol.T

    out = []
    for a in range(n_assets):
        asset_shocks = shocks[:, a * n_steps : (a + 1) * n_steps]
        steps = asset_shocks * widths[a][:, None] + drifts[a][:, None]
        offsets = np.cumsum(steps, axis=1)
        out.append(offsets)

    return out


class SimulateError(ValueError):
    """Raised when innovations or overlays carry unusable shapes or values."""


def seed_from_unit_id(unit_id: str) -> int:
    """Derive the pipeline's RNG seed from the card identity.

    `crc32(unit_id) & 0x7FFFFFFF` is the contract from the build prompt: the
    same card always drives the same draws, which is what makes re-runs
    byte-identical.
    """
    return zlib.crc32(unit_id.encode("utf-8")) & 0x7FFFFFFF


def simulate_mixture(
    innovations: np.ndarray,
    drift_per_draw: np.ndarray | float,
    width_mult: np.ndarray | float,
    seed: int,
) -> np.ndarray:
    """Combine bootstrap innovations with a per-draw overlay.

    Args:
        innovations: Demeaned per-step changes, shape ``(n_draws, n_steps)``.
        drift_per_draw: Per-step drift added to every step of a draw; a scalar
            applies the same drift to all rows, a ``(n_draws,)`` vector varies
            it per draw (Phase 3's scenario weights).
        width_mult: Multiplier on the innovation magnitude; scalar or
            ``(n_draws,)``.  Zero collapses a draw to its drift line.
        seed: Shuffles the pairing of innovation rows to overlay rows.

    Returns:
        Cumulative path offsets, shape ``(n_draws, n_steps)``, in the panel's
        own units.

    Raises:
        SimulateError: shape mismatch, non-finite values, or negative width.
    """
    innov = np.asarray(innovations, dtype=float)
    if innov.ndim != 2:
        raise SimulateError(f"innovations must be (n_draws, n_steps), got {innov.ndim}-D")
    n_draws, n_steps = innov.shape
    if not np.isfinite(innov).all():
        raise SimulateError("innovations contain NaN or infinity")

    drift = np.asarray(drift_per_draw, dtype=float)
    if drift.ndim == 0:
        drift = np.full(n_draws, float(drift))
    elif drift.shape == (n_draws,):
        pass
    else:
        raise SimulateError(
            f"drift_per_draw must be scalar or ({n_draws},), got shape {drift.shape}"
        )
    if not np.isfinite(drift).all():
        raise SimulateError("drift_per_draw contains NaN or infinity")

    width = np.asarray(width_mult, dtype=float)
    if width.ndim == 0:
        width = np.full(n_draws, float(width))
    elif width.shape == (n_draws,):
        pass
    else:
        raise SimulateError(
            f"width_mult must be scalar or ({n_draws},), got shape {width.shape}"
        )
    if not np.isfinite(width).all():
        raise SimulateError("width_mult contains NaN or infinity")
    if (width < 0).any():
        raise SimulateError("width_mult must be non-negative")

    order = np.random.default_rng(int(seed)).permutation(n_draws)
    mixed = innov[order]

    steps = mixed * width[:, None] + drift[:, None]
    return np.cumsum(steps, axis=1)