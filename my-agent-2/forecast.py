#!/usr/bin/env python3
"""Track 2 forecasting agent — CLI shim over `agent.main`.

The pipeline lives in `agent/main.py` (Phase 2+); this file preserves the
`--panels/--text/--asof/--out` interface that `forecast.sh` and external
harnesses already call.  `--out` names the `forecast.parquet` file; the
gated, atomic three-file write happens in its parent directory.
"""
import argparse
import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from agent.main import run  # noqa: E402
from harness.tools.card import parse_card  # noqa: E402


def find_card(start: pathlib.Path) -> pathlib.Path:
    """Search start, its parent, and grandparent for card.toml."""
    for d in (start, start.parent, start.parent.parent):
        c = d / "card.toml"
        if c.exists():
            return c
    raise FileNotFoundError(f"no card.toml found near {start}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panels", required=True, help="Directory with the parquet panel")
    p.add_argument("--text", required=True, help="Directory with text corpus documents")
    p.add_argument("--asof", required=True, help="Information cutoff (YYYY-MM-DD)")
    p.add_argument("--out", required=True, help="Output path for forecast.parquet")
    p.add_argument("--draws", type=int, default=200, help="Number of draws (>= 200)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    panels_dir = pathlib.Path(args.panels)
    out_path = pathlib.Path(args.out)
    out_dir = out_path.parent

    card_path = find_card(panels_dir)
    card = parse_card(card_path)
    cli_asof = dt.date.fromisoformat(args.asof)
    if cli_asof != card.asof:
        raise SystemExit(
            f"--asof {cli_asof} disagrees with card data_cutoff {card.asof}; "
            "the card is authoritative"
        )

    report = run(card_path, panels_dir, out_dir, args.draws)
    print(
        f"forecast complete for {report.unit_id}: {report.n_draws} draws, "
        f"{report.cell_count} cells, gates passed: {', '.join(report.gates)}"
    )


if __name__ == "__main__":
    main()
