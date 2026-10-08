"""End-to-end outer check for the promoted 252-step fully shrunk drift model."""

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
OUT = ROOT / "my_agent_3" / "testing" / "f1_selector_drift_promo.csv"
REPORT = ROOT / "my_agent_3" / "testing" / "F1_SELECTOR_DRIFT_PROMOTION.md"
N_DRAWS = 500
SEED = 81173


def _truth(full: pd.DataFrame, card: dict, origin: pd.Timestamp) -> np.ndarray:
    assets = [str(x) for x in card["targets"]["asset_ids"]]
    horizons = [int(x) for x in card["targets"]["horizons"]]
    train = full.loc[full.date <= origin]
    steps = arima._horizon_steps(train, card, origin.date().isoformat(), assets, horizons)
    y = np.empty((len(assets), len(horizons)), dtype=float)
    for ai, asset in enumerate(assets):
        future = full.loc[(full.asset.astype(str).str.lower() == asset.lower()) &
                          (full.date > origin)].sort_values("date").value.to_numpy(float)
        for hi, step in enumerate(steps[ai]):
            step = int(step)
            if len(future) < step:
                raise ValueError(f"incomplete target {asset} at {origin.date()}")
            if card["targets"]["target_type"] == "log_return":
                if np.any(future[:step] <= -1):
                    raise ValueError("return target <= -100%")
                y[ai, hi] = np.log1p(future[:step]).sum()
            else:
                y[ai, hi] = future[step - 1]
    return y


def _composite(draws: np.ndarray, y: np.ndarray, card: dict) -> float:
    return float(np.dot(model_selection._weights(card),
                        model_selection._components(draws, y)))


def main() -> None:
    origins = (pd.read_csv(SOURCE).query("model == 'selector'")
               [["card", "origin"]].drop_duplicates().sort_values(["card", "origin"]))
    rows, failures = [], []
    for card_i, card_id in enumerate(origins.card.unique()):
        card_path = ROOT / "units" / card_id / "card.toml"
        card = input_io.load_card(card_path)
        full = input_io.load_panel(card_path.parent, input_io.get_panel_id(card), "2100-01-01")
        full["date"] = pd.to_datetime(full.date)
        full["value"] = pd.to_numeric(full.value, errors="coerce")
        full = full.dropna(subset=["date", "asset", "value"])
        card_dates = origins.loc[origins.card == card_id, "origin"].tolist()
        for origin_i, date_text in enumerate(card_dates):
            origin = pd.Timestamp(date_text)
            train = full.loc[full.date <= origin].copy()
            seed = SEED + card_i * 10007 + origin_i * 1009
            try:
                y = _truth(full, card, origin)
                baseline_draws = random_walk.fit_and_sample(
                    train, card, date_text, N_DRAWS, seed=seed,
                )
                selector_draws, selected, details = model_selection.select_and_sample(
                    train, card, date_text, N_DRAWS, seed=seed,
                    local_drift_window=252, local_drift_shrink=1.0,
                )
                rw_score = _composite(baseline_draws, y, card)
                selector_score = _composite(selector_draws, y, card)
                rows.append({"card": card_id, "origin": date_text,
                             "rw_composite": rw_score,
                             "selector_composite": selector_score,
                             "relative_composite": selector_score / max(rw_score, 1e-12),
                             "selected": selected, "details": details})
            except Exception as exc:
                failures.append({"card": card_id, "origin": date_text,
                                 "error": repr(exc)})
        print(f"completed {card_id}: {len(card_dates)} outer origins", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    card_scores = df.groupby("card").agg(
        selector=("selector_composite", "mean"),
        random_walk=("rw_composite", "mean"),
    )
    card_scores["relative"] = card_scores.selector / card_scores.random_walk
    card_scores["gain"] = 1 - card_scores.relative
    rng = np.random.default_rng(81173)
    vals = card_scores.gain.to_numpy()
    boot = vals[rng.integers(0, len(vals), size=(4000, len(vals)))].mean(axis=1)
    ci = np.quantile(boot, [0.025, 0.975])
    wins = int((df.selector_composite < df.rw_composite).sum())
    report = [
        "# F1 selector after drift promotion", "", "## Method", "",
        f"Ran the complete production F1 selector on the same {len(df)} outer origins across {df.card.nunique()} eligible F1 cards, with {N_DRAWS} draws and one fixed seed per origin. Each selector run used only data through t1; later observations were used only to score. No text was loaded. Candidate 252-step local drift uses full drift shrinkage, as chosen by the preceding 3-seed drift grid. Card-level ratios pool raw composite loss across origins within each card, then average cards equally. The interval resamples cards as clusters (4,000 replicates).",
        "", "## Result", "",
        f"- Macro mean card ratio to random walk: **{card_scores.relative.mean():.4f}** (RW = 1.0).",
        f"- Median card ratio: **{card_scores.relative.median():.4f}**; median outer-origin ratio: **{df.relative_composite.median():.4f}**.",
        f"- Selector won on **{wins}/{len(df)}** outer origins.",
        f"- Card-cluster 95% interval for relative gain: **[{ci[0]:.4f}, {ci[1]:.4f}]**.",
        "", "## Promotion decision", "",
    ]
    promote = card_scores.relative.mean() < 1 and ci[0] > 0 and wins > len(df) / 2
    report.append(
        "The selector passed the pre-set check against random walk; retain the promoted drift setting."
        if promote else
        "The selector did not pass the full-pipeline check. Revert the drift parameter change and keep the prior production selector."
    )
    report += ["", "The previous saved selector's 84-origin report used a different Monte Carlo seed and a per-origin ratio aggregate; it is not treated as a seed-matched comparison. The preceding drift grid is the multi-seed evidence for the setting. This is still a public-panel backtest, not sealed competition scoring.", ""]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    (REPORT.parent / "f1_selector_drift_promo_failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8")
    print(json.dumps({"origins": len(df), "cards": df.card.nunique(),
                      "failures": failures, "macro_mean_card_ratio": float(card_scores.relative.mean()),
                      "median_card_ratio": float(card_scores.relative.median()),
                      "median_origin_ratio": float(df.relative_composite.median()),
                      "wins": wins, "gain_ci": ci.tolist(), "promote": bool(promote),
                      "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
