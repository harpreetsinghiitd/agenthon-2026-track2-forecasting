#!/usr/bin/env python3
"""
Track 2 forecasting agent.

Reads:
  --panels   directory containing the parquet panel
  --text     directory of dated text documents
  --asof     information cutoff (YYYY-MM-DD)
  --out      path for the forecast file

Writes, next to --out:
  forecast.parquet
  forecast_meta.json
  forecast_rationale.md
"""
import argparse
import json
import pathlib
import tomllib

import pandas as pd

from baselines.base import ForecastRequest
from baselines.theta_arima import ThetaARIMABaseline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panels", required=True, help="Directory with the parquet panel")
    p.add_argument("--text", required=True, help="Directory with text corpus documents")
    p.add_argument("--asof", required=True, help="Information cutoff (YYYY-MM-DD)")
    p.add_argument("--out", required=True, help="Output path for forecast.parquet")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Card discovery and parsing
# ---------------------------------------------------------------------------

def find_card(start: pathlib.Path) -> pathlib.Path:
    """Search start, its parent, and grandparent for card.toml."""
    for d in (start, start.parent, start.parent.parent):
        c = d / "card.toml"
        if c.exists():
            return c
    raise FileNotFoundError(f"no card.toml found near {start}")


def load_card(path: pathlib.Path) -> dict:
    return tomllib.loads(path.read_text())


def get_panel_id(card: dict) -> str:
    """Read the panel id from either layout: [panels].panel_ids or [[panels]]."""
    panels = card.get("panels", {})
    if isinstance(panels, dict) and "panel_ids" in panels:
        return panels["panel_ids"][0]
    if isinstance(panels, list) and panels:
        return panels[0]["panel_id"]
    raise KeyError(f"cannot determine panel_id from card; panels={panels!r}")


def get_n_draws_min(card: dict, default: int = 200) -> int:
    """Read n_draws_min from [scoring.params] or [scoring]."""
    scoring = card.get("scoring", {})
    if not isinstance(scoring, dict):
        return default
    params = scoring.get("params", {})
    if isinstance(params, dict) and "n_draws_min" in params:
        return int(params["n_draws_min"])
    if "n_draws_min" in scoring:
        return int(scoring["n_draws_min"])
    return default


# ---------------------------------------------------------------------------
# Panel loading
# ---------------------------------------------------------------------------

def load_panel(panels_dir: pathlib.Path, panel_id: str) -> pd.DataFrame:
    """Locate the parquet for panel_id and load it.

    Tries, in order:
      <panels_dir>/<panel_id>.parquet
      <panels_dir>/panels/<panel_id>.parquet
      any *.parquet directly in <panels_dir>
      any *.parquet recursively under <panels_dir>
    """
    candidates = [
        panels_dir / f"{panel_id}.parquet",
        panels_dir / "panels" / f"{panel_id}.parquet",
    ]
    for c in candidates:
        if c.exists():
            return pd.read_parquet(c)

    parquets = sorted(panels_dir.glob("*.parquet"))
    if parquets:
        return pd.read_parquet(parquets[0])

    parquets = sorted(panels_dir.rglob("*.parquet"))
    if parquets:
        return pd.read_parquet(parquets[0])

    raise FileNotFoundError(f"no parquet found under {panels_dir}")


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

def build_request(panel: pd.DataFrame, card: dict, panel_id: str,
                  asof: str, n_draws: int) -> ForecastRequest:
    targets = card["targets"]
    return ForecastRequest(
        panels={panel_id: panel},
        asof=asof,
        asset_ids=list(targets["asset_ids"]),
        horizons=list(targets["horizons"]),
        n_draws=n_draws,
    )


def samples_to_long(samples, assets, horizons) -> pd.DataFrame:
    """Convert [n_draws, n_assets, n_horizons] to long format."""
    rows = []
    for di in range(samples.shape[0]):
        for ai, a in enumerate(assets):
            for hi, h in enumerate(horizons):
                rows.append(
                    {"draw": di, "asset": a, "horizon": h,
                     "value": float(samples[di, ai, hi])}
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_meta(out_path: pathlib.Path, card: dict, asof: str, n_draws: int) -> dict:
    targets = card["targets"]
    meta = {
        "unit_id": card["task"]["id"],          # <- unit_id, not card_id
        "asof": asof,
        "representation": "samples",
        "asset_ids": list(targets["asset_ids"]),
        "horizons": list(targets["horizons"]),
        "n_draws": n_draws,
        "target": targets["target_type"],
    }
    out_path.write_text(json.dumps(meta, indent=2))
    return meta


def write_rationale(out_path: pathlib.Path, meta: dict, impl: str) -> None:
    """Required by gate g1. Never scored. Written for a human reviewer."""
    assets = ", ".join(meta["asset_ids"])
    horizons = meta["horizons"][0]
    out_path.write_text(f"""\
# Forecast rationale — {meta["unit_id"]}

**As of {meta["asof"]}. Target: joint distribution over {assets}
at horizon {horizons} business days.**

## Method

Statistical baseline ({impl}). {meta["n_draws"]} draws, generated jointly across
assets so the dependence structure is preserved — the joint variogram term
penalises draws generated independently per asset.

## What the text corpus contributed

Nothing in this run. This is the numerical baseline; the text corpus at
`/input/text/` is unread. A reasoning agent would read it and adjust the level
and spread of the distribution.

## What would change this forecast

A regime shift or a shock large enough to reprice the front of the curve.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    panels_dir = pathlib.Path(args.panels)
    out_path = pathlib.Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    card = load_card(find_card(panels_dir))
    panel_id = get_panel_id(card)
    panel = load_panel(panels_dir, panel_id)

    n_draws = max(500, get_n_draws_min(card, default=200))

    req = build_request(panel, card, panel_id, args.asof, n_draws)
    res = ThetaARIMABaseline().forecast(req)
    impl = res.metadata.get("implementation", "unknown")

    long_df = samples_to_long(res.samples, req.asset_ids, req.horizons)
    long_df.to_parquet(out_path, index=False)

    meta = write_meta(out_dir / "forecast_meta.json", card, args.asof, n_draws)
    write_rationale(out_dir / "forecast_rationale.md", meta, impl)

    print(f"wrote {out_path}, forecast_meta.json, forecast_rationale.md")
    print(f"  {len(req.asset_ids)} asset(s) x {len(req.horizons)} horizon(s), {n_draws} draws")


if __name__ == "__main__":
    main()