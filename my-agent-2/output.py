"""Write forecast.parquet, forecast_meta.json, forecast_rationale.md."""
import json
import pathlib

import numpy as np
import pandas as pd


def write_parquet(draws: np.ndarray, asset_ids: list[str], horizons: list[int],
                  out_path: pathlib.Path) -> None:
    rows = []
    for di in range(draws.shape[0]):
        for ai, a in enumerate(asset_ids):
            for hi, h in enumerate(horizons):
                rows.append({
                    "draw": di,
                    "asset": a,
                    "horizon": h,
                    "value": float(draws[di, ai, hi]),
                })
    df = pd.DataFrame(rows)
    df["draw"] = df["draw"].astype("int32")
    df["horizon"] = df["horizon"].astype("int32")
    df["value"] = df["value"].astype("float64")
    df.to_parquet(out_path, index=False)


def write_meta(out_dir: pathlib.Path, card: dict, asof: str, n_draws: int) -> dict:
    targets = card["targets"]
    meta = {
        "unit_id": card["task"]["id"],
        "asof": asof,
        "representation": "samples",
        "asset_ids": list(targets["asset_ids"]),
        "horizons": list(targets["horizons"]),
        "n_draws": n_draws,
        "target": targets["target_type"],
    }
    (out_dir / "forecast_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def write_rationale(out_dir: pathlib.Path, meta: dict, signals: dict | None) -> None:
    if signals is None:
        text_note = ("Nothing in this run. The text corpus is unread. "
                     "The LLM layer is a stub; the combiner is a passthrough.")
    else:
        text_note = signals.get("evidence", "Signals applied (no evidence field).")

    (out_dir / "forecast_rationale.md").write_text(f"""\
# Forecast rationale — {meta["unit_id"]}

**As of {meta["asof"]}. Target: joint distribution over
{", ".join(meta["asset_ids"])} at horizons {meta["horizons"]}.**

## Method

Numerical forecaster with {meta["n_draws"]} draws. Draws are produced jointly
across assets so the dependence structure is preserved.

## What the text corpus contributed

{text_note}

## What would change this forecast

A regime shift or a shock large enough to move the marginals or the tail.
""")


def write_all(draws: np.ndarray, asset_ids: list[str], horizons: list[int],
              card: dict, asof: str, n_draws: int, signals: dict | None,
              out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = out_path.parent

    write_parquet(draws, asset_ids, horizons, out_path)
    meta = write_meta(out_dir, card, asof, n_draws)
    write_rationale(out_dir, meta, signals)