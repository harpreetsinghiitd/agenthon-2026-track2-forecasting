"""Multi-seed, nested-origin comparison of local-drift random-walk settings."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "my_agent_3"))
import input as input_io
from statistical_layer import arima, model_selection, random_walk

SOURCE = ROOT / "my_agent_3" / "testing" / "f1_nested_backtest.csv"
OUT_CSV = ROOT / "my_agent_3" / "testing" / "f1_drift_grid_backtest.csv"
OUT_REPORT = ROOT / "my_agent_3" / "testing" / "F1_DRIFT_GRID_BACKTEST.md"
N_DRAWS = 500
SEEDS = (27031, 38791, 51277)
WINDOWS = (63, 126, 252, 504, None)
SHRINKS = (0.0, 0.5, 1.0)
BOOTSTRAPS = 4000


def _truth(full_panel: pd.DataFrame, card: dict, origin: pd.Timestamp) -> np.ndarray:
    assets = [str(x) for x in card["targets"]["asset_ids"]]
    horizons = [int(x) for x in card["targets"]["horizons"]]
    train = full_panel.loc[full_panel.date <= origin]
    steps = arima._horizon_steps(train, card, origin.date().isoformat(), assets, horizons)
    y = np.empty((len(assets), len(horizons)), dtype=float)
    for ai, asset in enumerate(assets):
        future = full_panel.loc[
            (full_panel.asset.astype(str).str.lower() == asset.lower()) &
            (full_panel.date > origin)
        ].sort_values("date").value.to_numpy(float)
        for hi, step in enumerate(steps[ai]):
            step = int(step)
            if len(future) < step:
                raise ValueError(f"incomplete target {asset} {origin.date()} step={step}")
            if card["targets"]["target_type"] == "log_return":
                path = future[:step]
                if np.any(path <= -1):
                    raise ValueError("log return <= -100%")
                y[ai, hi] = np.log1p(path).sum()
            else:
                y[ai, hi] = future[step - 1]
    return y


def _relative(draws: np.ndarray, y: np.ndarray, reference: np.ndarray,
              card: dict) -> float:
    losses = model_selection._components(draws, y)
    base = model_selection._components(reference, y)
    ratios = [x / max(abs(b), np.finfo(float).eps) for x, b in zip(losses, base)]
    return float(np.dot(model_selection._weights(card), ratios))


def _composite(draws: np.ndarray, y: np.ndarray, card: dict) -> float:
    losses = model_selection._components(draws, y)
    return float(np.dot(model_selection._weights(card), losses))


def _cluster_bootstrap(card_means: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(card_means)
    samples = rng.integers(0, n, size=(BOOTSTRAPS, n))
    means = card_means[samples].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def main() -> None:
    origins = (pd.read_csv(SOURCE).query("model == 'selector'")
               [["card", "origin"]].drop_duplicates().sort_values(["card", "origin"]))
    configurations = [(int(w) if w is not None else None, float(s))
                       for w in WINDOWS for s in SHRINKS]
    rows = []
    failed = []
    for card_idx, card_id in enumerate(origins.card.unique()):
        card_path = ROOT / "units" / card_id / "card.toml"
        card = input_io.load_card(card_path)
        full = input_io.load_panel(card_path.parent, input_io.get_panel_id(card), "2100-01-01")
        full["date"] = pd.to_datetime(full.date)
        full["value"] = pd.to_numeric(full.value, errors="coerce")
        full = full.dropna(subset=["date", "asset", "value"])
        card_origins = origins.loc[origins.card == card_id, "origin"].tolist()
        for origin_idx, date_text in enumerate(card_origins):
            origin = pd.Timestamp(date_text)
            train = full.loc[full.date <= origin].copy()
            try:
                y = _truth(full, card, origin)
            except Exception as exc:
                failed.append({"card": card_id, "origin": date_text,
                               "stage": "target", "error": repr(exc)})
                continue
            for seed_idx, seed in enumerate(SEEDS):
                run_seed = seed + card_idx * 10007 + origin_idx * 1009
                try:
                    reference = random_walk.fit_and_sample(
                        train, card, date_text, N_DRAWS, seed=run_seed,
                    )
                    ref_score = _relative(reference, y, reference, card)
                    rw_composite = _composite(reference, y, card)
                except Exception as exc:
                    failed.append({"card": card_id, "origin": date_text,
                                   "seed": seed, "stage": "random_walk",
                                   "error": repr(exc)})
                    continue
                rows.append({"card": card_id, "origin": date_text, "seed": seed,
                             "window": "rw", "drift_window": None,
                             "drift_shrink": np.nan, "relative_composite": ref_score,
                             "composite_score": rw_composite,
                             "rw_composite": rw_composite})
                for window, shrink in configurations:
                    label = "full" if window is None else str(window)
                    try:
                        draws = random_walk.fit_and_sample(
                            train, card, date_text, N_DRAWS, seed=run_seed,
                            drift_window=window, drift_shrink=shrink,
                        )
                        ratio = _relative(draws, y, reference, card)
                        rows.append({"card": card_id, "origin": date_text,
                                     "seed": seed, "window": label,
                                     "drift_window": window, "drift_shrink": shrink,
                                     "relative_composite": ratio,
                                     "composite_score": _composite(draws, y, card),
                                     "rw_composite": rw_composite})
                    except Exception as exc:
                        failed.append({"card": card_id, "origin": date_text,
                                       "seed": seed, "stage": f"{label}/{shrink}",
                                       "error": repr(exc)})
        print(f"completed {card_id}: {len(card_origins)} origins", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    rw_keys = result.loc[result.window == "rw", ["card", "origin", "seed", "relative_composite"]].rename(
        columns={"relative_composite": "rw_score"})
    candidate = result.loc[result.window != "rw"].merge(
        rw_keys, on=["card", "origin", "seed"], how="inner")
    candidate["gain_vs_rw"] = candidate.rw_score - candidate.relative_composite
    candidate["raw_gain_vs_rw"] = candidate.rw_composite - candidate.composite_score
    summary = []
    for (window, shrink), g in candidate.groupby(["window", "drift_shrink"], dropna=False):
        card_composite = g.groupby("card").agg(
            candidate_composite=("composite_score", "mean"),
            random_walk_composite=("rw_composite", "mean"),
        )
        card_composite["pooled_ratio"] = (
            card_composite.candidate_composite / card_composite.random_walk_composite
        )
        card_gains = 1.0 - card_composite.pooled_ratio
        ci_low, ci_high = _cluster_bootstrap(card_gains.to_numpy(), 9821 + len(summary))
        summary.append({
            "window": str(window), "drift_shrink": float(shrink),
            "origins": int(g[["card", "origin"]].drop_duplicates().shape[0]),
            "seeded_cases": int(len(g)),
            "mean_relative_composite": float(card_composite.pooled_ratio.mean()),
            "median_card_relative_composite": float(card_composite.pooled_ratio.median()),
            "median_relative_composite": float(g.relative_composite.median()),
            "wins_vs_rw": int((g.gain_vs_rw > 0).sum()),
            "win_cases": int(len(g)),
            "mean_gain_vs_rw": float(card_gains.mean()),
            "card_cluster_ci_low": ci_low, "card_cluster_ci_high": ci_high,
        })
    summary.sort(key=lambda x: x["mean_relative_composite"])
    summary_json = OUT_REPORT.parent / "f1_drift_grid_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_REPORT.parent / "f1_drift_grid_failures.json").write_text(
        json.dumps(failed, indent=2), encoding="utf-8")

    robust = [x for x in summary if x["mean_gain_vs_rw"] > 0 and
              x["card_cluster_ci_low"] > 0 and
              x["wins_vs_rw"] / max(x["win_cases"], 1) > 0.5]
    report = [
        "# F1 local-drift grid: multi-seed outer backtest", "", "## Method", "",
        f"Evaluated {len(origins)} previously defined outer origins across {origins.card.nunique()} eligible F1 cards, using {len(SEEDS)} fixed seeds and {N_DRAWS} draws per run. Grid: windows {list(WINDOWS)} and shrinkage {list(SHRINKS)}. Each origin was fitted using only panel data through t1; post-t1 data was used only for scoring. No text was loaded. The metric is the card-weighted composite divided by same-origin, same-seed random-walk loss; RW = 1.0 and lower is better. Confidence intervals resample cards as clusters ({BOOTSTRAPS} bootstrap replicates), preserving within-card dependence.",
        "", "## Results", "",
        "| Window | Shrink | Macro mean card ratio | Median card ratio | Median origin ratio | Wins vs RW | Mean gain vs RW | 95% card-cluster CI for gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in summary:
        report.append(f"| {x['window']} | {x['drift_shrink']:.2f} | {x['mean_relative_composite']:.4f} | {x['median_card_relative_composite']:.4f} | {x['median_relative_composite']:.4f} | {x['wins_vs_rw']}/{x['win_cases']} | {x['mean_gain_vs_rw']:.4f} | [{x['card_cluster_ci_low']:.4f}, {x['card_cluster_ci_high']:.4f}] |")
    if robust:
        chosen = robust[0]
        decision = (f"**Promotion candidate:** window={chosen['window']}, shrink={chosen['drift_shrink']:.2f}. It had positive mean gain, a positive 95% card-cluster lower confidence bound, and won on more than half of seeded cases. Update production only after a separate paired comparison against the current F1 selector confirms no regression.")
    else:
        decision = "**No setting met the promotion rule.** Keep the production selector unchanged. A lower point estimate without a positive card-cluster confidence bound and a majority win rate is not enough evidence to switch."
    report += ["", "## Decision", "", decision, "",
               f"Forecast-generation failures: {len(failed)}. See [f1_drift_grid_backtest.csv](f1_drift_grid_backtest.csv), [f1_drift_grid_summary.json](f1_drift_grid_summary.json), and the failures JSON for row-level details.", ""]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"origins": len(origins), "configs": len(configurations),
                      "seeded_score_rows": len(candidate), "failures": len(failed),
                      "summary": summary, "robust_promotion_candidates": robust,
                      "report": str(OUT_REPORT)}, indent=2))


if __name__ == "__main__":
    main()
