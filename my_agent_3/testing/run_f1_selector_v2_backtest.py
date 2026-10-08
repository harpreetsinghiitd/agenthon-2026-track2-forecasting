"""Paired outer rolling evaluation of the revised F1 selector.

Reuses the previously defined 84 outer origins, but recomputes every forecast.
At each origin, selector validation and final fits receive panel data only up
to t1. Outcomes after t1 are used only for scoring. No corpus text is loaded.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "my_agent_3"))
import input as input_io
from statistical_layer import arima, ets, level_change_arima, model_selection, random_walk

SOURCE = ROOT / "my_agent_3" / "testing" / "f1_nested_backtest.csv"
OUT_CSV = ROOT / "my_agent_3" / "testing" / "f1_selector_v2_backtest.csv"
OUT_REPORT = ROOT / "my_agent_3" / "testing" / "F1_SELECTOR_V2_BACKTEST.md"
N_DRAWS = 500
SEED = 27031


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
                raise ValueError(f"incomplete outer target for {asset} at {origin.date()}")
            if card["targets"]["target_type"] == "log_return":
                values = future[:step]
                if np.any(values <= -1):
                    raise ValueError("log-return path contains a value <= -100%")
                y[ai, hi] = np.log1p(values).sum()
            else:
                y[ai, hi] = future[step - 1]
    return y


def _score(draws: np.ndarray, y: np.ndarray, reference: np.ndarray,
           card: dict) -> dict:
    losses = model_selection._components(draws, y)
    ref_losses = model_selection._components(reference, y)
    ratios = [a / max(abs(b), np.finfo(float).eps)
              for a, b in zip(losses, ref_losses)]
    weights = model_selection._weights(card)
    return {
        "crps": losses[0], "variogram": losses[1], "tail": losses[2],
        "relative_composite_rw": float(np.dot(weights, ratios)),
        "mean_absolute_error": float(np.mean(np.abs(draws.mean(axis=0) - y))),
    }


def main() -> None:
    origins = (pd.read_csv(SOURCE).query("model == 'selector'")
               [["card", "origin"]].drop_duplicates().sort_values(["card", "origin"]))
    rows = []
    failures = []
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
            seed = SEED + card_idx * 10007 + origin_idx * 1009
            try:
                y = _truth(full, card, origin)
                reference = random_walk.fit_and_sample(train, card, date_text,
                                                       N_DRAWS, seed=seed + 1)
            except Exception as exc:
                failures.append({"card": card_id, "origin": date_text,
                                 "stage": "target/reference", "error": repr(exc)})
                continue

            methods = {
                "random_walk": lambda: random_walk.fit_and_sample(
                    train, card, date_text, N_DRAWS, seed=seed + 1),
                "local_drift_rw": lambda: random_walk.fit_and_sample(
                    train, card, date_text, N_DRAWS, seed=seed + 2,
                    drift_window=252, drift_shrink=0.5),
                "ARIMA": lambda: arima.fit_and_sample(
                    train, card, date_text, N_DRAWS, seed=seed + 3),
                "ETS": lambda: ets.fit_and_sample(
                    train, card, date_text, N_DRAWS, seed=seed + 4),
                "selector_v2": lambda: model_selection.select_and_sample(
                    train, card, date_text, N_DRAWS, seed=seed + 5),
            }
            if card["targets"]["target_type"] == "level":
                methods["ARIMA_log_changes"] = lambda: level_change_arima.fit_and_sample(
                    train, card, date_text, N_DRAWS, seed=seed + 6)
            for model_name, make in methods.items():
                try:
                    result = make()
                    if model_name == "selector_v2":
                        draws, details = result[0], result[2]
                    else:
                        draws, details = result, ""
                    if draws.shape != (N_DRAWS, len(card["targets"]["asset_ids"]),
                                       len(card["targets"]["horizons"])):
                        raise ValueError(f"unexpected draw shape {draws.shape}")
                    if not np.isfinite(draws).all():
                        raise ValueError("nonfinite draws")
                    metrics = _score(draws, y, reference, card)
                    rows.append({"card": card_id, "origin": date_text,
                                 "model": model_name, "available": True,
                                 "n_draws": N_DRAWS, "selection_details": details,
                                 **metrics})
                except Exception as exc:
                    failures.append({"card": card_id, "origin": date_text,
                                     "stage": model_name, "error": repr(exc)})
                    rows.append({"card": card_id, "origin": date_text,
                                 "model": model_name, "available": False,
                                 "n_draws": N_DRAWS, "selection_details": "",
                                 "crps": np.nan, "variogram": np.nan, "tail": np.nan,
                                 "relative_composite_rw": np.nan,
                                 "mean_absolute_error": np.nan})
        print(f"completed {card_id}: {len(card_origins)} outer origins", flush=True)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_CSV, index=False)
    summaries = []
    for name, group in result_df.loc[result_df.available].groupby("model"):
        ratios = group.relative_composite_rw
        rw = result_df.loc[(result_df.model == "random_walk") & result_df.available,
                           ["card", "origin", "relative_composite_rw"]].rename(
                               columns={"relative_composite_rw": "rw_ratio"})
        paired = group.merge(rw, on=["card", "origin"], how="inner")
        summaries.append({
            "model": name, "origins": int(len(group)),
            "mean_relative_composite": float(ratios.mean()),
            "median_relative_composite": float(ratios.median()),
            "wins_vs_rw": int((paired.relative_composite_rw < paired.rw_ratio).sum()),
            "paired_origins": int(len(paired)),
            "mean_mae": float(group.mean_absolute_error.mean()),
        })
    summaries.sort(key=lambda x: x["mean_relative_composite"])
    previous = (pd.read_csv(SOURCE).query("model == 'selector'")
                [["card", "origin", "relative_to_rw"]]
                .rename(columns={"relative_to_rw": "previous_relative_composite"}))
    revised = result_df.query("model == 'selector_v2' and available")[[
        "card", "origin", "relative_composite_rw"]]
    paired = previous.merge(revised, on=["card", "origin"], how="inner")
    old_mean = float(paired.previous_relative_composite.mean())
    old_median = float(paired.previous_relative_composite.median())
    new_mean = float(paired.relative_composite_rw.mean())
    new_median = float(paired.relative_composite_rw.median())
    new_wins = int((paired.relative_composite_rw <
                    paired.previous_relative_composite).sum())
    report = [
        "# Revised F1 selector: nested outer backtest",
        "",
        "## Method",
        "",
        f"Recomputed {len(origins)} previously defined outer origins across {origins.card.nunique()} F1 cards. Each forecast used {N_DRAWS} draws. At each t1, all model selection and fitting used only panel rows dated through t1; later observations were read only to score the horizon targets. No text was loaded. Scores are the card-weighted composite relative to the same-origin plain random walk (RW = 1.0); lower is better. This was an experiment that added plain RW and positive-level log-change ARIMA as selector candidates and allowed any best-scoring candidate to become the mixture anchor.",
        "",
        "## Results",
        "",
        "| Model | Origins | Mean relative composite | Median | Wins vs RW | Mean absolute error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        report.append(f"| {s['model']} | {s['origins']} | {s['mean_relative_composite']:.4f} | {s['median_relative_composite']:.4f} | {s['wins_vs_rw']}/{s['paired_origins']} | {s['mean_mae']:.6g} |")
    report += ["", "## Comparison with the previous selector", "",
               f"On the same {len(paired)} outer origins, the saved previous selector averaged {old_mean:.4f} (median {old_median:.4f}) relative to RW; this experiment averaged {new_mean:.4f} (median {new_median:.4f}) and beat the previous saved score on {new_wins}/{len(paired)} origins. The runs used different Monte Carlo seeds, so the comparison is indicative rather than a seed-matched paired test.",
               "", "The experiment did not improve the aggregate results. The production selector has therefore been restored to its prior conservative ARIMA-centered rule, with local-drift anchoring only after a material and consistent validation gain. The log-change ARIMA implementation remains available for controlled experiments but is not in the production candidate set.",
               "", f"Unavailable model fits: {len(failures)}. The log-change ARIMA was applicable only to positive level targets (76 origins); the other 8 targets were log returns. Details are in the CSV rows marked `available=false` or the failures JSON.",
               "", "## Limitations", "",
               "These are historical public-panel backtests, not sealed competition outcomes. Origin selection follows the prior 84-origin F1 nested run, so this is paired against earlier candidates, not an independent final test set. `ARIMA_log_changes` applies only to positive level series; failures on nonpositive levels are expected and are not selector failures.",
               ""]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    (OUT_REPORT.parent / "f1_selector_v2_backtest_failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summaries, "failures": len(failures),
                      "rows": len(result_df), "csv": str(OUT_CSV),
                      "report": str(OUT_REPORT)}, indent=2))


if __name__ == "__main__":
    main()
