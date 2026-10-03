"""Backtest tool — run the forecaster on practice cards, score against M0.

Role in the pipeline
--------------------
Phase 6 requirement: run the forecaster on N practice cards, compute the
composite score against M0 reconstruction where possible, log results.

This tool is NOT part of the submission Docker image; it is a local development
tool for the participant to iterate.  It uses the same `run()` entrypoint as
the submission but loops over a directory of practice cards and scores the
output using the organiser's scorer (if available) or the local M0 baseline.

Usage:
    python -m harness.tools.backtest --cards-dir units/ --out-dir backtest_out/ [--scorer /path/to/qfbench2]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent.main import run
from harness.tools.card import parse_card

__all__ = ["BacktestConfig", "run_backtest", "main"]

_LOG = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    cards_dir: Path
    out_dir: Path
    n_draws: int = 200
    scorer_path: Path | None = None  # path to qfbench2 binary if available
    m0_dir: Path | None = None  # directory with M0 baseline outputs for comparison


@dataclass
class CardResult:
    """Result for one card."""
    unit_id: str
    passed_gates: bool
    gate_failures: list[str]
    composite_score: float | None
    m0_score: float | None
    runtime_seconds: float


def run_backtest(config: BacktestConfig) -> list[CardResult]:
    """Run the forecaster on all cards in `cards_dir`."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[CardResult] = []

    card_dirs = sorted([d for d in config.cards_dir.iterdir() if d.is_dir() and (d / "card.toml").exists()])
    if not card_dirs:
        _LOG.warning("No cards found in %s", config.cards_dir)
        return results

    for card_dir in card_dirs:
        _LOG.info("Backtesting %s", card_dir.name)
        card_path = card_dir / "card.toml"
        out_dir = config.out_dir / card_dir.name

        try:
            import time
            start = time.perf_counter()
            report = run(card_path, card_dir, out_dir, n_draws=config.n_draws)
            elapsed = time.perf_counter() - start

            result = CardResult(
                unit_id=report.unit_id,
                passed_gates=True,
                gate_failures=[],
                composite_score=None,
                m0_score=None,
                runtime_seconds=elapsed,
            )

            # Score with organiser's scorer if available
            if config.scorer_path and config.scorer_path.exists():
                score = _run_scorer(config.scorer_path, card_dir, out_dir)
                result.composite_score = score

            # Compare with M0 if available
            if config.m0_dir and (config.m0_dir / card_dir.name / "forecast.parquet").exists():
                m0_score = _compare_with_m0(out_dir, config.m0_dir / card_dir.name)
                result.m0_score = m0_score

            results.append(result)
            _LOG.info("  %s: gates=PASS, time=%.2fs, composite=%s, m0=%s",
                      card_dir.name, elapsed,
                      f"{result.composite_score:.4f}" if result.composite_score else "N/A",
                      f"{result.m0_score:.4f}" if result.m0_score else "N/A")

        except Exception as exc:
            _LOG.exception("Card %s failed: %s", card_dir.name, exc)
            results.append(CardResult(
                unit_id=card_dir.name,
                passed_gates=False,
                gate_failures=[str(exc)],
                composite_score=None,
                m0_score=None,
                runtime_seconds=0.0,
            ))

    # Write summary
    summary = config.out_dir / "summary.json"
    summary.write_text(json.dumps([{
        "unit_id": r.unit_id,
        "passed_gates": r.passed_gates,
        "gate_failures": r.gate_failures,
        "composite_score": r.composite_score,
        "m0_score": r.m0_score,
        "runtime_seconds": r.runtime_seconds,
    } for r in results], indent=2))

    _LOG.info("Backtest complete. Summary written to %s", summary)
    return results


def _run_scorer(scorer_path: Path, card_dir: Path, out_dir: Path) -> float | None:
    """Run the organiser's scorer binary, return composite score."""
    try:
        # qfbench2-smoke <unit_dir> <output_dir> --track forecasting
        result = subprocess.run(
            [str(scorer_path), "smoke", str(card_dir), str(out_dir), "--track", "forecasting"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            _LOG.warning("Scorer failed: %s", result.stderr[:500])
            return None
        # Parse composite score from output (format depends on scorer version)
        for line in result.stdout.splitlines():
            if "composite" in line.lower() or "score" in line.lower():
                try:
                    return float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
        return None
    except subprocess.TimeoutExpired:
        _LOG.warning("Scorer timed out")
        return None
    except Exception as exc:
        _LOG.warning("Scorer error: %s", exc)
        return None


def _compare_with_m0(out_dir: Path, m0_dir: Path) -> float | None:
    """Compare forecast vs M0 baseline, return a simple distance metric."""
    try:
        our = pd.read_parquet(out_dir / "forecast.parquet")
        m0 = pd.read_parquet(m0_dir / "forecast.parquet")
        # Align by draw, asset, horizon
        merged = our.merge(m0, on=["draw", "asset", "horizon"], suffixes=("_our", "_m0"))
        if merged.empty:
            return None
        # Mean absolute difference across draws
        diff = (merged["value_our"] - merged["value_m0"]).abs().mean()
        return float(diff)
    except Exception as exc:
        _LOG.warning("M0 comparison failed: %s", exc)
        return None


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run backtest over a card directory")
    parser.add_argument("--cards-dir", required=True, type=Path, help="Directory containing card subdirectories")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for results")
    parser.add_argument("--draws", type=int, default=200, help="Number of draws per card")
    parser.add_argument("--scorer", type=Path, help="Path to qfbench2 scorer binary")
    parser.add_argument("--m0-dir", type=Path, help="Directory with M0 baseline outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = BacktestConfig(
        cards_dir=args.cards_dir,
        out_dir=args.out_dir,
        n_draws=args.draws,
        scorer_path=args.scorer,
        m0_dir=args.m0_dir,
    )

    results = run_backtest(config)
    passed = sum(1 for r in results if r.passed_gates)
    _LOG.info("Backtest complete: %d/%d cards passed gates", passed, len(results))
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()