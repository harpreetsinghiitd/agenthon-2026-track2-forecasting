"""Plot daily F1 forecasts from the live model-selection pipeline.

Each origin is cutoff-safe: selection and final fitting receive only panel rows
dated on or before that origin. The text corpus is used only for plot markers.
"""

from __future__ import annotations

import json
import pathlib
import sys
import xml.sax.saxutils as xml_escape
import copy

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "my_agent_3"))
import input as input_io
from statistical_layer import random_walk

OUT_DIR = ROOT / "my_agent_3" / "testing" / "f1_daily_pipeline_charts"
CASES = {
    "t2-F1-aud-on-hold-2016": ("AUD", ["2015-05-01", "2016-02-03"]),
    "t2-F1-cad-boc-2017": ("CAD", ["2016-01-08", "2016-10-06"]),
    "t2-F1-eur-range-2017": ("EUR", ["2016-03-08", "2016-12-06"]),
    "t2-F1-jpy-ycc-flex-2018": ("JPY", ["2017-01-30", "2017-10-27"]),
}
# Earlier and later forecast means, respectively. Observed values are purple.
COLORS = ["#1769aa", "#e07a20"]
TRAINING_COLOR = "#7b61a8"


def _chart(path: pathlib.Path, card_id: str, asset: str, unit: str,
           series: pd.Series, forecasts: list[dict], docs: list[dict]) -> None:
    width, height = 1250, 790
    left, right, top, bottom = 95, 45, 82, 270
    plot_w, plot_h = width - left - right, height - top - bottom
    x0 = min(series.index.min(), *(pd.Timestamp(fc["train_start"]) for fc in forecasts))
    x1 = max(max(fc["target_dates"]) for fc in forecasts)
    y_values = series.to_numpy(float).tolist()
    for fc in forecasts:
        y_values.extend(fc["means"])
    y_min, y_max = min(y_values), max(y_values)
    y_pad = max((y_max - y_min) * .08, abs(y_max) * .005, 1e-9)
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def sx(date):
        return left + (pd.Timestamp(date) - x0).total_seconds() / max((x1 - x0).total_seconds(), 1) * plot_w

    def sy(value):
        return top + (y_max - float(value)) / max(y_max - y_min, 1e-12) * plot_h

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg += ['<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:22px;font-weight:bold}.small{font-size:11px}.axis{font-size:12px}</style>',
            f'<text class="title" x="{left}" y="31">F1 — {xml_escape.escape(card_id)}</text>',
            f'<text class="axis" x="{left}" y="55">{xml_escape.escape(asset)} | {xml_escape.escape(unit)} | live F1 selector | daily mean of 200 Monte Carlo draws</text>']

    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = sy(value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        svg.append(f'<text class="axis" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.5g}</text>')
    for i in range(7):
        date = x0 + (x1 - x0) * i / 6
        x = sx(date)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#f0f0f0"/>')
        svg.append(f'<text class="axis" x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle">{date:%Y-%m}</text>')

    # Actual full series is drawn first; colored overlays then identify the
    # complete observations visible to each forecast at its cutoff.
    points = " ".join(f"{sx(date):.1f},{sy(value):.1f}" for date, value in series.items())
    svg.append(f'<polyline points="{points}" fill="none" stroke="{TRAINING_COLOR}" stroke-width="1.8" opacity="0.78"/>')

    # Highlight the exact observed path made available to each forecast origin.
    for i, fc in enumerate(forecasts):
        color = TRAINING_COLOR
        observed = series.loc[(series.index >= pd.Timestamp(fc["train_start"])) &
                              (series.index <= pd.Timestamp(fc["origin"]))]
        points = " ".join(f"{sx(date):.1f},{sy(value):.1f}" for date, value in observed.items())
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.35" opacity="0.72"/>')
        x = sx(fc["origin"])
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="{COLORS[i]}" stroke-width="1.4" stroke-dasharray="5 4"/>')
        svg.append(f'<text class="small" x="{x+4:.1f}" y="{top+15+i*14}" fill="{COLORS[i]}">t1 {fc["origin"]:%Y-%m-%d}</text>')

    for i, fc in enumerate(forecasts):
        color = COLORS[i]
        coords = [(sx(fc["origin"]), sy(fc["origin_value"]))]
        coords.extend((sx(date), sy(value)) for date, value in zip(fc["target_dates"], fc["means"]))
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        # Show a point for every forecast date; this makes the daily path explicit.
        for date, value in zip(fc["target_dates"], fc["means"]):
            svg.append(f'<circle cx="{sx(date):.1f}" cy="{sy(value):.1f}" r="1.8" fill="{color}"/>')

    # Document markers are contextual only; no text is read by the forecaster.
    shown_docs = [d for d in docs if x0 <= d["date"] <= x1]
    for di, doc in enumerate(shown_docs, 1):
        x = sx(doc["date"])
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#777" stroke-width="1" stroke-dasharray="3 5" opacity="0.52"/>')
        svg.append(f'<text class="small" x="{x+2:.1f}" y="{top+plot_h-5-(di%3)*12}" fill="#666" transform="rotate(-90 {x+2:.1f} {top+plot_h-5-(di%3)*12})">D{di}</text>')

    axis_y = top + plot_h
    svg += [f'<line x1="{left}" y1="{axis_y}" x2="{width-right}" y2="{axis_y}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" stroke="#333"/>',
            f'<text class="axis" x="{width/2}" y="{height-12}" text-anchor="middle">Date</text>',
            f'<text class="axis" transform="translate(22 {top+plot_h/2}) rotate(-90)" text-anchor="middle">Target level</text>']

    # Legend also states the true training start and cutoff for each path.
    ly = axis_y + 43
    svg.append(f'<line x1="{left}" y1="{ly}" x2="{left+30}" y2="{ly}" stroke="{TRAINING_COLOR}" stroke-width="2"/><text class="small" x="{left+38}" y="{ly+4}">Given historical data / training input (purple)</text>')
    for i, fc in enumerate(forecasts):
        color = COLORS[i]
        x = left + 160 + i * 490
        svg.append(f'<line x1="{x}" y1="{ly}" x2="{x+30}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        svg.append(f'<text class="small" x="{x+38}" y="{ly+4}">{["Blue", "Orange"][i]} forecast mean; purple training {fc["train_start"]:%Y-%m-%d} to {fc["origin"]:%Y-%m-%d}</text>')
    ly += 24
    svg.append(f'<line x1="{left}" y1="{ly}" x2="{left+30}" y2="{ly}" stroke="#777" stroke-dasharray="3 5"/><text class="small" x="{left+38}" y="{ly+4}">D# = frozen-corpus document timestamp, shown for context only; not input to either forecast</text>')
    for di, doc in enumerate(shown_docs, 1):
        col, row = (di - 1) % 2, (di - 1) // 2
        x, y = left + col * 545, ly + 19 + row * 14
        svg.append(f'<text class="small" x="{x}" y="{y}">D{di} {doc["date"]:%Y-%m-%d} {xml_escape.escape(doc["name"][:65])}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for card_id, (asset, origins) in CASES.items():
        card_path = ROOT / "units" / card_id / "card.toml"
        card = input_io.load_card(card_path)
        unit_dir = card_path.parent
        full_panel = input_io.load_panel(unit_dir, input_io.get_panel_id(card), "2100-01-01")
        full_panel["date"] = pd.to_datetime(full_panel.date)
        full_series = (full_panel.loc[full_panel.asset.astype(str).str.lower() == asset.lower()]
                       .sort_values("date").set_index("date").value.astype(float))
        forecasts = []
        for i, origin_text in enumerate(origins):
            origin = pd.Timestamp(origin_text)
            train = full_panel.loc[full_panel.date <= origin].copy()
            # The selector receives no future rows and no corpus content.
            fit_card = copy.deepcopy(card)
            fit_card["targets"]["horizons"] = list(
                range(1, max(int(h) for h in card["targets"]["horizons"]) + 1)
            )
            draws = random_walk.fit_and_sample(
                train, fit_card, origin_text, n_draws=200, seed=1701 + i,
                drift_window=252, drift_shrink=1.0,
            )
            method = "local-drift random walk (252-step window, full shrinkage)"
            details = "selected as the best standalone F1 configuration in the three-seed drift grid"
            target_assets = [str(x) for x in card["targets"]["asset_ids"]]
            ai = [x.lower() for x in target_assets].index(asset.lower())
            target_dates = full_series.loc[full_series.index > origin].index[:draws.shape[2]]
            if len(target_dates) != draws.shape[2]:
                raise RuntimeError(f"incomplete future dates for {card_id} at {origin_text}")
            if len(card["targets"]["asset_ids"]) != 1:
                raise RuntimeError("daily path chart currently expects one target asset")
            forecasts.append({
                "origin": origin,
                "origin_value": float(full_series.loc[:origin].iloc[-1]),
                "train_start": pd.Timestamp(train.loc[train.asset.astype(str).str.lower() == asset.lower(), "date"].min()),
                "target_dates": target_dates,
                "means": np.mean(draws[:, ai, :], axis=0),
                "method": method,
                "details": details,
                "training_rows": int((train.asset.astype(str).str.lower() == asset.lower()).sum()),
                "draw_shape": list(draws.shape),
            })

        last_forecast_date = max(fc["target_dates"][-1] for fc in forecasts)
        # Show the whole observed history so the colored training windows are
        # literal: each model saw every target observation from t0 through t1.
        series = full_series.loc[full_series.index <= last_forecast_date]
        index_path = unit_dir / "text" / "corpus_index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        docs = [{"date": pd.Timestamp(r["timestamp"]),
                 "name": pathlib.Path(r["file"]).stem}
                for r in index.get("documents", [])]
        docs.sort(key=lambda d: d["date"])
        unit = str(card["targets"].get("value_unit", card["targets"]["target_type"]))
        path = OUT_DIR / f"{card_id}_daily_full_pipeline.svg"
        _chart(path, card_id, asset, unit, series, forecasts, docs)
        manifest.append({
            "card": card_id, "asset": asset, "chart": path.name,
            "pipeline": "main.py F1 route: local-drift random walk, 252-step window, full shrinkage; selector is fallback only",
            "text_used_by_forecast": False,
            "forecasts": [{
                "color": ["blue", "orange"][i],
                "origin_t1": str(fc["origin"].date()),
                "training_start_t0": str(fc["train_start"].date()),
                "training_end_t1": str(fc["origin"].date()),
                "training_rows_for_asset": fc["training_rows"],
                "n_draws": 200,
                "daily_steps": int(len(fc["means"])),
                "mean_path_method": fc["method"],
                "selection_details": fc["details"],
                "draw_shape": fc["draw_shape"],
            } for i, fc in enumerate(forecasts)],
            "document_markers": len(docs),
        })

    (OUT_DIR / "README.md").write_text(
        "# F1 daily full-pipeline forecast charts\n\n"
        "Each SVG uses purple for the given historical data and training inputs, blue for the earlier forecast mean, and orange for the later forecast mean. It shows a predictive mean for every forecast date and marks the two forecast cutoffs. At each t1 cutoff, the production F1 local-drift random walk used only panel rows through that cutoff and generated 200 Monte Carlo paths. Text documents were not passed into the forecast; corpus timestamps are grey context markers only. The rolling model selector remains a fallback if the selected drift model fails. Transformation, joint-draw, and LLM conditioning layers are not activated for F1.\n\n"
        + "\n".join(f"- **{x['card']}** — blue t1 {x['forecasts'][0]['origin_t1']}, orange t1 {x['forecasts'][1]['origin_t1']}; [chart]({x['chart']})" for x in manifest)
        + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "charts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
