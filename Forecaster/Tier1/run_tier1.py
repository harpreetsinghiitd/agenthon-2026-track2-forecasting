#!/usr/bin/env python3
"""Standalone driver for the Tier 1 ARIMA forecaster.

Writes forecast.parquet, forecast_meta.json, forecast_rationale.md
to <this-folder>/output/.
"""
import argparse
import json
import pathlib
import sys
import tomllib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from forecaster_t1_ARIMA import fit_and_sample  # noqa: E402


def find_card(start: pathlib.Path) -> pathlib.Path:
    for d in (start, start.parent, start.parent.parent):
        c = d / "card.toml"
        if c.exists():
            return c
    raise FileNotFoundError(f"no card.toml near {start}")


def load_panel(panels_dir: pathlib.Path, panel_id: str) -> pd.DataFrame:
    for c in (panels_dir / f"{panel_id}.parquet",
              panels_dir / "panels" / f"{panel_id}.parquet"):
        if c.exists():
            return pd.read_parquet(c)
    parquets = sorted(panels_dir.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no parquet under {panels_dir}")
    return pd.read_parquet(parquets[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panels", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--out", default="output/forecast.parquet")
    args = p.parse_args()

    panels_dir = pathlib.Path(args.panels)
    out_path = pathlib.Path(args.out)
    if not out_path.is_absolute():
        out_path = HERE / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    card = tomllib.loads(find_card(panels_dir).read_text())
    panel_id = card["panels"]["panel_ids"][0]
    panel = load_panel(panels_dir, panel_id)

    asset_ids = list(card["targets"]["asset_ids"])
    horizons = list(card["targets"]["horizons"])
    n_draws = 500

    draws = fit_and_sample(panel, card, args.asof, n_draws)

    rows = []
    for di in range(draws.shape[0]):
        for ai, a in enumerate(asset_ids):
            for hi, h in enumerate(horizons):
                rows.append({"draw": di, "asset": a, "horizon": h,
                             "value": float(draws[di, ai, hi])})
    df = pd.DataFrame(rows)
    df["draw"] = df["draw"].astype("int32")
    df["horizon"] = df["horizon"].astype("int32")
    df["value"] = df["value"].astype("float64")
    df.to_parquet(out_path, index=False)

    meta = {
        "unit_id": card["task"]["id"],
        "asof": args.asof,
        "representation": "samples",
        "asset_ids": asset_ids,
        "horizons": horizons,
        "n_draws": n_draws,
        "target": card["targets"]["target_type"],
    }
    (out_path.parent / "forecast_meta.json").write_text(json.dumps(meta, indent=2))

    (out_path.parent / "forecast_rationale.md").write_text(f"""\
# Forecast rationale — {meta["unit_id"]}

Tier 1 ARIMA per asset. {n_draws} draws. Text corpus unread.

## Method

ARIMA with small order search, fitted per asset. Gaussian draws from
the forecast mean and standard error at each horizon.

## What the text corpus contributed

Nothing. Tier 1 is the numerical baseline.

## What would change this forecast

A regime shift or an extreme event the ARIMA residuals do not capture.
""")

    print(f"wrote {out_path}, forecast_meta.json, forecast_rationale.md")
    print(f"  {len(asset_ids)} asset(s) x {len(horizons)} horizon(s), {n_draws} draws")


if __name__ == "__main__":
    main()

# run using
'''
cd /path/to/track2-starter-kit

REPO=$(pwd) python3 Forecaster/Tier1/run_tier1.py \
  --panels units/t2-F1-ai-mom-2024 \
  --text   units/t2-F1-ai-mom-2024/text \
  --asof   2024-05-31 \
  --out    output/forecast.parquet
'''