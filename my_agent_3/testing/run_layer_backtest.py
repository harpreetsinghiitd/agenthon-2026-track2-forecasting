"""Executive summary: leakage-safe comparison of the family forecasting routes.

Example: python3 my_agent_3/testing/run_layer_backtest.py --origins 2 --draws 200
No text corpus is loaded. Every forecast uses panel observations through t1;
observations between t1 and t2 are read only to score the forecast.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import tomllib

import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps
from qfbench2_track_forecasting.tail import tail_pinball

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "my_agent_3"))
import input as input_io
from joint_draws_layer import gaussian_copula, student_t_copula
from statistical_layer import arima, ets, markov_switching_ar, random_walk, var
from transformation_layer import garch, skewed_t

LEVELS = (0.01, 0.05, 0.95, 0.99)


def _target(panel: pd.DataFrame, card: dict, origin: pd.Timestamp,
            steps: np.ndarray) -> np.ndarray:
    assets = [str(x) for x in card["targets"]["asset_ids"]]
    horizons = card["targets"]["horizons"]
    result = np.empty((len(assets), len(horizons)), dtype=float)
    for ai, asset in enumerate(assets):
        series = panel.loc[panel.asset.astype(str).str.lower() == asset.lower()].sort_values("date")
        future = series.loc[series.date > origin, "value"].to_numpy(float)
        for hi, _ in enumerate(horizons):
            step = int(steps[ai, hi])
            if len(future) < step:
                raise ValueError("future target window is incomplete")
            window = future[:step]
            if card["targets"]["target_type"] == "log_return":
                if np.any(window <= -1):
                    raise ValueError("invalid simple return in target window")
                result[ai, hi] = np.log1p(window).sum()
            else:
                result[ai, hi] = window[step - 1]
    return result


def _origins(panel: pd.DataFrame, card: dict, n_origins: int) -> list[tuple[pd.Timestamp, np.ndarray, pd.DataFrame, np.ndarray]]:
    assets = [str(a).lower() for a in card["targets"]["asset_ids"]]
    data = panel.copy()
    data["date"] = pd.to_datetime(data.date)
    target_panel = data.loc[data.asset.astype(str).str.lower().isin(assets)]
    dates = pd.DatetimeIndex(target_panel.date.drop_duplicates().sort_values())
    if len(dates) < 35:
        return []
    selected = []
    last_position = len(dates) - 1
    for position in range(last_position - 1, 29, -1):
        origin = dates[position]
        train = target_panel.loc[target_panel.date <= origin].copy()
        try:
            steps = arima._horizon_steps(train, card, origin.date().isoformat(),
                                         card["targets"]["asset_ids"],
                                         [int(h) for h in card["targets"]["horizons"]])
            if np.any(steps <= 0):
                continue
            enough = True
            for asset in card["targets"]["asset_ids"]:
                future_n = int((target_panel.loc[
                    (target_panel.asset.astype(str).str.lower() == str(asset).lower())
                    & (target_panel.date > origin), "date"].nunique()))
                if future_n < int(steps.max()):
                    enough = False
                    break
            if not enough:
                continue
            if selected and dates.get_loc(selected[-1][0]) - position < int(np.max(selected[-1][1])):
                continue
            y = _target(target_panel, card, origin, steps)
            selected.append((origin, steps, train, y))
            if len(selected) >= n_origins:
                break
        except Exception:
            continue
    return sorted(selected, key=lambda x: x[0])


def _score(draws: np.ndarray, truth: np.ndarray, reference: np.ndarray,
           weights: tuple[float, float, float]) -> dict:
    def components(sample):
        flat = np.asarray(sample, dtype=float).reshape(sample.shape[0], -1)
        y = truth.reshape(-1)
        return np.array([
            crps.crps_marginal(flat, y),
            crps.variogram_score(flat, y, p=0.5),
            tail_pinball(flat, y, LEVELS),
        ], dtype=float)
    losses = components(draws)
    baseline = components(reference)
    ratios = losses / np.maximum(np.abs(baseline), np.finfo(float).eps)
    return {
        "crps": float(losses[0]), "variogram": float(losses[1]),
        "tail": float(losses[2]), "composite_ratio_rw": float(np.dot(weights, ratios)),
        "mean_error": float(np.mean(draws, axis=0).reshape(-1).mean() - truth.mean()),
        "mae_mean": float(np.mean(np.abs(np.mean(draws, axis=0) - truth))),
    }


def _weights(card: dict) -> tuple[float, float, float]:
    params = card.get("scoring", {}).get("params", {})
    w = params.get("weights", {})
    m, j, t = (float(w.get(k, d)) for k, d in (("marginal", .5), ("joint", .3), ("tail", .2)))
    cells = len(card["targets"]["asset_ids"]) * len(card["targets"]["horizons"])
    if cells == 1:
        return m / (m + t), 0.0, t / (m + t)
    total = m + j + t
    return m / total, j / total, t / total


def _forecasts(train, card, origin, draws, seed, arima_search=False):
    family = str(card.get("metadata", {}).get("category", card["task"]["id"])).upper()
    # Use the handover's fixed ARIMA order for outer backtesting. This avoids
    # repeatedly searching six AIC orders while still fitting only t0..t1.
    try:
        base = arima.fit_and_sample(
            train, card, origin, draws, seed=seed,
            order=None if arima_search else (1, 0, 1),
        )
        method = ("production ARIMA candidate search; no text" if arima_search
                  else "fixed ARIMA(1,0,1); no text")
    except Exception as arima_error:
        try:
            base = ets.fit_and_sample(train, card, origin, draws, seed=seed)
            method = f"ETS fallback after fixed ARIMA failure: {type(arima_error).__name__}"
        except Exception:
            base = random_walk.fit_and_sample(train, card, origin, draws, seed=seed)
            method = f"random-walk fallback after fixed ARIMA failure: {type(arima_error).__name__}"
    plain_rw = random_walk.fit_and_sample(train, card, origin, draws, seed=seed + 1)
    drift_rw = random_walk.fit_and_sample(train, card, origin, draws, seed=seed + 2,
                                          drift_window=252, drift_shrink=0.5)
    current_drift = random_walk.fit_and_sample(
        train, card, origin, draws, seed=seed + 3,
        drift_window=252, drift_shrink=1.0,
    )
    output = {"layer1_arima_fixed": base, "layer1_local_drift_252_0.5": drift_rw,
              "layer1_f1_current_252_1.0": current_drift,
              "random_walk": plain_rw}
    if "F2" in family:
        try:
            output["layer1_markov_switching_ar"] = markov_switching_ar.fit_and_sample(
                train, card, origin, draws, seed=seed + 10)
        except Exception as exc:
            output["_markov_error"] = f"{type(exc).__name__}: {exc}"
    if "F3" in family:
        try:
            output["layer1_var"] = var.fit_and_sample(
                train, card, origin, draws, seed=seed + 11)
        except Exception as exc:
            output["_var_error"] = f"{type(exc).__name__}: {exc}"
    if "F3" in family or "F4" in family:
        try:
            garch_shrunk = garch.transform(base, train, card, origin, seed=seed + 4,
                                           garch_weight=0.5)
            output["layer2_garch_shrunk"] = garch_shrunk
            if "F4" in family:
                output["layer2_garch_plus_student_t"] = skewed_t.transform(garch_shrunk)
            if "F3" in family and len(card["targets"]["asset_ids"]) > 1:
                output["layer3_independent"] = garch_shrunk
                output["layer3_gaussian"] = gaussian_copula.couple(
                    garch_shrunk, train, card, seed=seed + 5)
                output["layer3_student_t"] = student_t_copula.couple(
                    garch_shrunk, train, card, seed=seed + 6)
                output["f3_full_student_t"] = output["layer3_student_t"]
                output["f3_full_gaussian"] = output["layer3_gaussian"]
                output["f3_no_copula_garch"] = output["layer2_garch_shrunk"]
                if "layer1_var" in output:
                    var_garch = garch.transform(
                        output["layer1_var"], train, card, origin, seed=seed + 12)
                    output["f3_var_garch_student"] = student_t_copula.couple(
                        var_garch, train, card, seed=seed + 13)
            if "F4" in family:
                output["f4_full_tails"] = skewed_t.transform(garch_shrunk)
                output["f4_garch_only"] = garch_shrunk
                output["f4_tails_only"] = skewed_t.transform(base)
                output["f4_student_t_df3"] = skewed_t.transform(
                    garch_shrunk, degrees_freedom=3.0)
                output["f4_student_t_df8"] = skewed_t.transform(
                    garch_shrunk, degrees_freedom=8.0)
        except Exception as exc:
            output["_layer_error"] = str(exc)
    return output, method


def _standardized_points(train: pd.DataFrame, card: dict, steps: np.ndarray,
                         mean_draw: np.ndarray, truth: np.ndarray) -> list[tuple[float, float]]:
    """Standardize each cell by its historical one-step innovation scale."""
    points = []
    for ai, asset in enumerate(card["targets"]["asset_ids"]):
        series = train.loc[train.asset.astype(str).str.lower() == str(asset).lower()].sort_values("date").value.astype(float)
        if card["targets"]["target_type"] == "log_return":
            innovations = np.log1p(series.to_numpy())
            base = 0.0
        else:
            values = series.to_numpy()
            innovations = np.diff(values)
            base = float(values[-1])
        one_step_sd = float(np.std(innovations, ddof=1)) if len(innovations) > 1 else 0.0
        for hi in range(len(card["targets"]["horizons"])):
            scale = one_step_sd * np.sqrt(max(1, int(steps[ai, hi])))
            if not np.isfinite(scale) or scale < 1e-12:
                continue
            points.append(((float(mean_draw[ai, hi]) - base) / scale,
                           (float(truth[ai, hi]) - base) / scale))
    return points


def _svg_scatter(path: pathlib.Path, points: list[tuple[float, float]], title: str,
                 subtitle: str) -> None:
    width, height, margin = 760, 560, 72
    vals = [v for pair in points for v in pair if np.isfinite(v)]
    if not vals:
        return
    low, high = min(vals), max(vals)
    span = max(high - low, 1e-9)
    low -= span * .08
    high += span * .08
    def sx(x): return margin + (x-low)/(high-low)*(width-2*margin)
    def sy(y): return height-margin - (y-low)/(high-low)*(height-2*margin)
    circles = "".join(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="#1769aa" opacity="0.68"/>' for x,y in points)
    ticks = []
    for i in range(5):
        v = low + (high-low)*i/4
        ticks.append(f'<text x="{sx(v):.2f}" y="{height-margin+24}" text-anchor="middle">{v:.3g}</text>')
        ticks.append(f'<text x="{margin-10}" y="{sy(v)+4:.2f}" text-anchor="end">{v:.3g}</text>')
    diagonal = f'<line x1="{sx(low):.2f}" y1="{sy(low):.2f}" x2="{sx(high):.2f}" y2="{sy(high):.2f}" stroke="#b33" stroke-dasharray="7 5"/>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><style>text{{font:13px sans-serif;fill:#222}}.title{{font:bold 20px sans-serif}}</style>
<text class="title" x="{margin}" y="30">{title}</text><text x="{margin}" y="52">{subtitle}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>
{diagonal}{''.join(ticks)}{circles}
<text x="{width/2}" y="{height-12}" text-anchor="middle">Predictive mean / in-sample scale</text>
<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle">Actual target / in-sample scale</text></svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origins", type=int, default=2)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--cards", nargs="*", help="Optional exact card IDs for a quick diagnostic run")
    parser.add_argument("--families", nargs="*", choices=["T2-F1", "T2-F2", "T2-F3", "T2-F4"],
                        help="Optional family subset")
    parser.add_argument("--output-dir", type=pathlib.Path, default=None,
                        help="Output directory; defaults to testing/layer_backtest")
    parser.add_argument("--arima-search", action="store_true",
                        help="Use production ARIMA AIC order search instead of fixed (1,0,1)")
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / "my_agent_3" / "testing" / "layer_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, plot_candidates, excluded, origin_failures = [], {}, [], []
    selected_ids = set(args.cards or [])
    for card_path in sorted((ROOT / "units").glob("*/card.toml")):
        card = input_io.load_card(card_path)
        if selected_ids and card.get("task", {}).get("id") not in selected_ids:
            continue
        unit_dir = card_path.parent
        family = str(card.get("metadata", {}).get("category", card["task"]["id"])).upper()
        if args.families and family not in args.families:
            continue
        try:
            panel = input_io.load_panel(unit_dir, input_io.get_panel_id(card), "2100-01-01")
            folds = _origins(panel, card, args.origins)
            if not folds:
                excluded.append({"card": card["task"]["id"], "reason": "no complete rolling origins"})
                continue
            valid_origins = 0
            for fi, (origin, steps, train, truth) in enumerate(folds):
                row_start = len(rows)
                try:
                    seed = args.seed + origin.toordinal() + fi
                    forecast_map, method = _forecasts(
                        train, card, origin.date().isoformat(), args.draws, seed,
                        arima_search=args.arima_search,
                    )
                    reference = forecast_map["random_walk"]
                    for variant, sample in forecast_map.items():
                        if variant.startswith("_"):
                            continue
                        scores = _score(sample, truth, reference, _weights(card))
                        mean_draw = np.mean(sample, axis=0)
                        standardized = _standardized_points(train, card, steps, mean_draw, truth)
                        if variant == "layer1_arima_fixed":
                            scores["forecast_mean"] = json.dumps(mean_draw.reshape(-1).tolist())
                            scores["actual_values"] = json.dumps(truth.reshape(-1).tolist())
                            scores["standardized_mean_actual_pairs"] = json.dumps(standardized)
                        rows.append({
                            "card": card["task"]["id"], "family": family,
                            "origin_t1": origin.date().isoformat(),
                            "target_end_t2": str((panel.loc[panel.date > origin, "date"].sort_values().iloc[int(steps.max())-1]).date()),
                            "train_start_t0": str(train.date.min().date()),
                            "target_type": card["targets"]["target_type"],
                            "assets": "|".join(map(str, card["targets"]["asset_ids"])),
                            "horizons": "|".join(map(str, card["targets"]["horizons"])),
                            "variant": variant, "agent_layer1_method": method,
                            "n_draws": args.draws, "documents_used": 0,
                            "n_train_rows": len(train), **scores,
                        })
                        if variant == "layer1_arima_fixed" and family[-2:] in {"F1", "F2", "F3", "F4"}:
                            slot = plot_candidates.setdefault(family[-2:], {"points": [], "cards": set()})
                            slot["cards"].add(card["task"]["id"])
                            slot["points"].extend(standardized)
                    valid_origins += 1
                except Exception as exc:
                    del rows[row_start:]
                    origin_failures.append({"card": card["task"]["id"],
                                            "origin_t1": origin.date().isoformat(),
                                            "reason": f"{type(exc).__name__}: {exc}"})
            if valid_origins == 0:
                excluded.append({"card": card["task"]["id"], "reason": "all forecast origins failed"})
        except Exception as exc:
            excluded.append({"card": card.get("task", {}).get("id", card_path.parent.name),
                             "reason": f"{type(exc).__name__}: {exc}"})
    csv_path = output_dir / "all_card_origin_scores.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    for family, item in plot_candidates.items():
        _svg_scatter(output_dir / f"{family.lower()}_actual_vs_draw_mean.svg", item["points"],
                     f"{family}: actual vs predictive mean",
                     f"{len(item['cards'])} cards | {len(item['points'])} standardized origin-target points")
    summary = {
        "design": "Numeric-only rolling origins; panel fit data ends at t1; later panel values through t2 are used only as outcomes; no text documents are loaded or used.",
        "t0": "first target observation available in each card's panel",
        "n_draws": args.draws, "origins_per_card_requested": args.origins,
        "scored_cards": sorted({r["card"] for r in rows}), "excluded_cards": excluded,
        "origin_failures": origin_failures,
        "csv": str(csv_path), "plots": {f: {"cards": len(v["cards"]), "path": str(output_dir / f'{f.lower()}_actual_vs_draw_mean.svg'), "points": len(v["points"])} for f,v in plot_candidates.items()},
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "cards": len(summary["scored_cards"]),
                      "excluded": len(excluded), "output": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
