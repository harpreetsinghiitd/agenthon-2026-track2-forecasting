#!/usr/bin/env python3
"""Track 2 forecasting agent — orchestrator."""
import argparse
import pathlib

import input as io
import forecaster
import llm
import combiner
import output as out


AGENT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panels", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def resolve_out(raw: str) -> pathlib.Path:
    """If --out is relative, anchor it to the folder containing this file."""
    p = pathlib.Path(raw)
    if p.is_absolute():
        return p
    return AGENT_DIR / p


def main() -> None:
    args = parse_args()
    panels_dir = pathlib.Path(args.panels)
    text_dir = pathlib.Path(args.text)
    out_path = resolve_out(args.out)

    card = io.load_card(io.find_card(panels_dir))
    panel_id = io.get_panel_id(card)
    panel = io.load_panel(panels_dir, panel_id)
    documents = io.load_text(text_dir, args.asof)

    asset_ids = list(card["targets"]["asset_ids"])
    horizons = list(card["targets"]["horizons"])
    n_draws = max(500, io.get_n_draws_min(card, default=200))

    raw_draws = forecaster.fit_and_sample(panel, card, args.asof, n_draws)
    signals = llm.extract_signals(documents, args.asof, card)
    adjusted_draws = combiner.apply(raw_draws, signals)

    out.write_all(adjusted_draws, asset_ids, horizons, card,
                  args.asof, n_draws, signals, out_path)

    print(f"wrote {out_path}, forecast_meta.json, forecast_rationale.md")
    print(f"  {len(asset_ids)} asset(s) x {len(horizons)} horizon(s), {n_draws} draws")


if __name__ == "__main__":
    main()