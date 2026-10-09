"""Benchmark every posterior-generation mechanism against known outcomes.

Train/validation design: leakage-safe rolling origins (the same folds as
run_layer_backtest). Each origin fits the family pipeline on panel data
through t1 only; observations in (t1, t2] are the validation target.

For every (card, origin) the harness:
1. generates the text-blind prior draws once (per-family production
   composition, fixed ARIMA base) and caches them;
2. derives an ORACLE signal from the realized validation outcome, expressed
   in the production signal's own units and bounds -- the best signal the
   vocabulary could ever carry (ceiling analysis; it uses the outcome by
   design, never for submission);
3. degrades it: +-25% / 50% multiplicative noise, and an adversarial
   inversion (confidently wrong), plus the neutral signal;
4. applies EVERY mechanism in posterior_mechanisms.MECHANISMS;
5. scores CRPS / variogram / tail-pinball against the realized outcome with
   the card's own composite weights, as a ratio to the UNCONDITIONED prior.

Reads: units/*/card.toml + panels. Writes: CSV rows + POSTERIOR_BENCHMARK.md.

Example: python3 my_agent_3/testing/run_posterior_benchmark.py --origins 2
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "my_agent_3"))

from qfbench2_common.scoring import crps
from qfbench2_track_forecasting.tail import tail_pinball

import input as input_io
from joint_draws_layer import gaussian_copula, student_t_copula
from llm import signals as llm_signals
from statistical_layer import arima, ets, random_walk
from transformation_layer import garch, skewed_t

import posterior_mechanisms as pm

# Reuse the layer backtest's fold construction and card weights verbatim.
_spec = importlib.util.spec_from_file_location(
    "run_layer_backtest", pathlib.Path(__file__).with_name("run_layer_backtest.py"))
_lbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lbt)
_origins, _weights = _lbt._origins, _lbt._weights

LEVELS = (0.01, 0.05, 0.95, 0.99)
NOISE_LEVELS = (0.25, 0.50)


# ---------------------------------------------------------------------------
# Prior generation (once per card-origin, cached)
# ---------------------------------------------------------------------------


def _base_forecast(train, card, origin, n_draws, seed):
    try:
        return arima.fit_and_sample(train, card, origin, n_draws, seed=seed,
                                    order=(1, 0, 1)), "ARIMA(1,0,1)"
    except Exception:
        try:
            return ets.fit_and_sample(train, card, origin, n_draws,
                                      seed=seed), "ETS"
        except Exception:
            return random_walk.fit_and_sample(train, card, origin, n_draws,
                                              seed=seed), "random walk"


def _prior(train, card, origin, n_draws, seed):
    """The production text-blind composition per family (the prior we post-
    process). Mirrors main.py's family dispatch with the fixed ARIMA base the
    layer backtest uses."""
    family = str(card.get("metadata", {}).get("category",
                                              card["task"]["id"])).upper()
    multi = len(card["targets"]["asset_ids"]) > 1
    notes = []
    if "F1" in family:
        try:
            draws, method = random_walk.fit_and_sample(
                train, card, origin, n_draws, seed=seed,
                drift_window=252, drift_shrink=1.0), "local-drift RW 252/1.0"
        except Exception:
            draws, method = _base_forecast(train, card, origin, n_draws, seed)
        return draws, f"F1 prior: {method}"
    draws, method = _base_forecast(train, card, origin, n_draws, seed)
    notes.append(method)
    if "F3" in family or "F4" in family:
        try:
            draws = garch.transform(draws, train, card, origin, seed=seed + 4,
                                    garch_weight=0.5)
            notes.append("GARCH")
        except Exception:
            notes.append("GARCH skipped")
    if "F4" in family:
        draws = skewed_t.transform(draws, seed=seed + 5)
        notes.append("skewed-t")
    if multi and ("F3" in family or "F4" in family):
        try:
            draws = student_t_copula.couple(draws, train, card, seed=seed + 6)
            notes.append("t-copula")
        except Exception:
            try:
                draws = gaussian_copula.couple(draws, train, card,
                                                seed=seed + 6)
                notes.append("gaussian copula")
            except Exception:
                notes.append("independent marginals")
    return draws, f"prior: {'+'.join(notes)}"


# ---------------------------------------------------------------------------
# Oracle + degraded signals (the elicitation simulator)
# ---------------------------------------------------------------------------


def _oracle_signal(prior: np.ndarray, truth: np.ndarray) -> dict:
    """Best-in-vocabulary signal implied by the realized outcome.

    Per-cell z = (truth - prior median) / prior sd; the per-card signal is
    the clipped cross-cell aggregate, exactly as one LLM signal must cover
    the whole card. Uses the validation outcome BY DESIGN (ceiling analysis).
    """
    med = np.median(prior, axis=0)
    sd = prior.std(axis=0)
    safe = np.where(sd > 1e-12, sd, np.nan)
    z = np.where(np.isfinite(safe), (truth - med) / safe, 0.0)
    mean_z = float(np.mean(z))
    abs_z = float(np.mean(np.abs(z)))
    direction = "none" if abs(mean_z) < 0.05 else ("left" if mean_z < 0 else "right")
    # multi-asset co-movement: tighten dependence when every asset's realized
    # z shares the sign of the card-level read
    corr = 1.0
    if prior.shape[1] > 1:
        asset_z = np.nanmean(z, axis=1)
        if np.all(asset_z > 0) or np.all(asset_z < 0):
            corr = 1.3
    return {
        "mean_shift_sigma": float(np.clip(mean_z, -0.5, 0.5)),
        "variance_scale": float(np.clip(abs_z / 1.645, 0.75, 1.75)),
        "tail_direction": direction,
        "tail_strength": float(np.clip(abs_z - 1.0, 0.0, 1.0)),
        "correlation_scale": corr,
        "shock_probability": float(np.clip(abs_z / 2.33, 0.0, 1.0)),
        "basis": "established",
        "evidence": "oracle signal derived from the realized validation outcome",
    }


def _rebound(raw: dict) -> dict:
    """Re-clamp a (possibly out-of-bounds) signal dict through the production
    validator so every degraded signal obeys the same contract."""
    return llm_signals.parse_signals(json.dumps(raw))


def _noisy(oracle: dict, q: float, rng: np.random.Generator) -> dict:
    out = dict(oracle)
    for key in ("mean_shift_sigma", "variance_scale", "tail_strength",
                "correlation_scale", "shock_probability"):
        out[key] = float(oracle[key]) * (1.0 + q * rng.standard_normal())
    return _rebound(out)


def _adversarial(oracle: dict) -> dict:
    """Confidently wrong: every knob inverted within bounds."""
    out = dict(oracle)
    out["mean_shift_sigma"] = -float(oracle["mean_shift_sigma"])
    out["variance_scale"] = 1.0 / max(float(oracle["variance_scale"]), 1e-6)
    out["tail_direction"] = {"left": "right", "right": "left"}.get(
        oracle["tail_direction"], "left")
    out["correlation_scale"] = 1.0
    return _rebound(out)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _components(flat_draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([
        crps.crps_marginal(flat_draws, y),
        crps.variogram_score(flat_draws, y, p=0.5),
        tail_pinball(flat_draws, y, LEVELS),
    ], dtype=float)


def _ratio_row(draws, truth, prior_losses, weights) -> dict:
    flat = np.asarray(draws, dtype=float).reshape(draws.shape[0], -1)
    y = truth.reshape(-1)
    losses = _components(flat, y)
    denom = np.maximum(np.abs(prior_losses), np.finfo(float).eps)
    return {
        "crps": float(losses[0]), "variogram": float(losses[1]),
        "tail": float(losses[2]),
        "composite_ratio_prior": float(np.dot(weights, losses / denom)),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origins", type=int, default=2)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--cards", nargs="*")
    parser.add_argument("--families", nargs="*",
                        choices=["T2-F1", "T2-F2", "T2-F3", "T2-F4"])
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--refresh", action="store_true",
                        help="regenerate cached priors")
    args = parser.parse_args()

    out_dir = args.output_dir or ROOT / "my_agent_3" / "testing" / "posterior_benchmark"
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    selected = set(args.cards or [])
    rows, excluded = [], []
    card_paths = sorted((ROOT / "units").glob("*/card.toml"))
    for card_path in card_paths:
        card = input_io.load_card(card_path)
        cid = card.get("task", {}).get("id")
        if selected and cid not in selected:
            continue
        family = str(card.get("metadata", {}).get("category",
                                                  card["task"]["id"])).upper()
        if args.families and family not in args.families:
            continue
        try:
            panel = input_io.load_panel(card_path.parent,
                                        input_io.get_panel_id(card), "2100-01-01")
            folds = _origins(panel, card, args.origins)
            if not folds:
                excluded.append({"card": cid, "reason": "no complete origins"})
                continue
        except Exception as exc:
            excluded.append({"card": cid, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        weights = _weights(card)
        for fi, (origin, steps, train, truth) in enumerate(folds):
            seed = args.seed + origin.toordinal() + fi
            key = f"{cid}_{origin.date().isoformat()}"
            npz = cache / f"{key}.npz"
            try:
                if args.refresh or not npz.is_file():
                    prior, method = _prior(train, card, origin.date().isoformat(),
                                           args.draws, seed)
                    np.savez_compressed(npz, prior=prior, truth=truth,
                                        method=method)
                else:
                    blob = np.load(npz)
                    prior, truth, method = (blob["prior"], blob["truth"],
                                            str(blob["method"]))
                expected = (args.draws, len(card["targets"]["asset_ids"]),
                            len(card["targets"]["horizons"]))
                if prior.shape != expected or not np.isfinite(prior).all():
                    raise ValueError(f"bad prior shape {prior.shape}")
            except Exception as exc:
                excluded.append({"card": cid, "origin": origin.date().isoformat(),
                                 "reason": f"{type(exc).__name__}: {exc}"})
                continue

            prior_losses = _components(
                prior.reshape(prior.shape[0], -1), truth.reshape(-1))
            oracle = _oracle_signal(prior, truth)
            rng = np.random.default_rng(seed + 7919)
            signal_sets = {
                "neutral": llm_signals.NEUTRAL,
                "oracle": oracle,
                "noisy25": _noisy(oracle, NOISE_LEVELS[0], rng),
                "noisy50": _noisy(oracle, NOISE_LEVELS[1], rng),
                "adversarial": _adversarial(oracle),
            }
            for quality, signal in signal_sets.items():
                for name in pm.MECHANISMS:
                    mech_seed = seed + sum(ord(c) for c in name) % 997
                    post = pm.apply_mechanism(name, prior, signal, mech_seed)
                    scores = _ratio_row(post, truth, prior_losses, weights)
                    rows.append({
                        "card": cid, "family": family,
                        "origin_t1": origin.date().isoformat(),
                        "mechanism": name,
                        "mechanism_family": pm.FAMILY_OF.get(name, "?"),
                        "signal_quality": quality,
                        "prior_method": method,
                        "n_draws": int(prior.shape[0]),
                        **scores,
                    })
            print(f"[done] {key}: {len(pm.MECHANISMS)} mechanisms x "
                  f"{len(signal_sets)} signals", flush=True)

    csv_path = out_dir / "posterior_scores.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # Aggregate: per (signal_quality, mechanism) across card-origins.
    frame = pd.DataFrame(rows)
    summary_path = out_dir / "aggregate.json"
    agg = {}
    if not frame.empty:
        grouped = frame.groupby(["signal_quality", "mechanism"])
        agg = {
            f"{q}|{m}": {
                "family": pm.FAMILY_OF.get(m, "?"),
                "n": int(len(g)),
                "mean_ratio": float(g["composite_ratio_prior"].mean()),
                "median_ratio": float(g["composite_ratio_prior"].median()),
                "win_rate": float((g["composite_ratio_prior"] < 1.0).mean()),
                "crps_ratio": float(g["crps"].mean()),
                "variogram_ratio": float(g["variogram"].mean()),
                "tail_ratio": float(g["tail"].mean()),
            }
            for (q, m), g in grouped
        }
    summary_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "cards_scored": frame.card.nunique()
                      if not frame.empty else 0,
                      "excluded": len(excluded), "csv": str(csv_path)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
