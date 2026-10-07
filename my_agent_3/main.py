"""Executive summary: run the Track 2 forecast workflow for one unit.

The entrypoint loads the card, panel, and cutoff-safe text; selects statistical
fallbacks and optional distribution/dependence layers by family; then writes the
three required forecast artifacts. Text modules remain neutral templates.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

import input as input_io
import output as output_io
from joint_draws_layer import gaussian_copula, student_t_copula
from llm import signals as llm_signals
from statistical_layer import arima, block_bootstrap, ets, markov_switching_ar, random_walk, var
from transformation_layer import garch, skewed_t


AGENT_DIR = pathlib.Path(__file__).resolve().parent
MAX_DRAWS = 2000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agenthon Track 2 forecasting agent")
    parser.add_argument("--panels", required=True, help="Unit directory containing card and panel")
    parser.add_argument("--text", required=True, help="Frozen text corpus directory")
    parser.add_argument("--asof", required=True, help="Information cutoff in YYYY-MM-DD format")
    parser.add_argument("--out", required=True, help="Output path ending in forecast.parquet")
    parser.add_argument("--n-draws", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--card", default=None, help="Optional explicit card.toml path")
    return parser.parse_args(argv)


def _resolve_out(raw: str) -> pathlib.Path:
    path = pathlib.Path(raw).expanduser()
    return path if path.is_absolute() else AGENT_DIR / path


def _draw_count(card: dict, requested: int | None) -> int:
    category = str(card.get("metadata", {}).get("category", ""))
    task_id = str(card.get("task", {}).get("id", ""))
    family = category or task_id
    minimum = max(200, input_io.get_n_draws_min(card))
    if "F4" in family:
        minimum = max(1000, minimum)
    count = max(minimum, requested or 500)
    if count > MAX_DRAWS:
        raise ValueError(f"n_draws={count} exceeds the contract maximum {MAX_DRAWS}")
    return count


def _sample_with_fallback(panel, card, asof: str, n_draws: int, seed: int,
                          family: str) -> tuple[np.ndarray, str]:
    """Try family-relevant numerical models, then ETS and random walk."""
    if "F1" in family and card.get("targets", {}).get("target_type") == "log_return":
        # Rolling-origin results favor block resampling for the return cards tested;
        # level cards did not show a consistent gain, so they stay ARIMA-first.
        candidates = [("10-step block bootstrap", block_bootstrap.fit_and_sample),
                      ("ARIMA", arima.fit_and_sample), ("ETS", ets.fit_and_sample),
                      ("random walk", random_walk.fit_and_sample)]
    elif "F1" in family:
        candidates = [("ARIMA", arima.fit_and_sample), ("ETS", ets.fit_and_sample),
                      ("random walk", random_walk.fit_and_sample)]
    elif "F3" in family:
        candidates = [("ARIMA", arima.fit_and_sample), ("VAR", var.fit_and_sample),
                      ("ETS", ets.fit_and_sample), ("random walk", random_walk.fit_and_sample)]
    elif "F2" in family:
        candidates = [("ARIMA", arima.fit_and_sample),
                      ("Markov-switching AR", markov_switching_ar.fit_and_sample),
                      ("ETS", ets.fit_and_sample), ("random walk", random_walk.fit_and_sample)]
    else:
        candidates = [("ARIMA", arima.fit_and_sample), ("ETS", ets.fit_and_sample),
                      ("random walk", random_walk.fit_and_sample)]

    errors = []
    for name, model in candidates:
        try:
            draws = model(panel, card, asof, n_draws, seed=seed)
            if draws.shape != (n_draws, len(card["targets"]["asset_ids"]),
                               len(card["targets"]["horizons"])):
                raise ValueError(f"{name} returned unexpected draw shape {draws.shape}")
            if not np.isfinite(draws).all():
                raise ValueError(f"{name} returned non-finite draws")
            return draws, name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    details = " | ".join(errors)
    raise RuntimeError(f"all numerical forecasters failed: {details}")


def run(args: argparse.Namespace) -> dict:
    unit_dir = pathlib.Path(args.panels).expanduser()
    card_path = pathlib.Path(args.card).expanduser() if args.card else input_io.find_card(unit_dir)
    card = input_io.load_card(card_path)
    panel_id = input_io.get_panel_id(card)
    panel = input_io.load_panel(unit_dir, panel_id, args.asof)
    documents = input_io.load_text(pathlib.Path(args.text).expanduser(), args.asof)

    family = str(card.get("metadata", {}).get("category", card["task"]["id"]))
    family = family.upper()
    n_draws = _draw_count(card, args.n_draws)
    draws, method = _sample_with_fallback(panel, card, args.asof, n_draws,
                                          args.seed, family)
    method_notes = [method]
    signals = None

    if "F3" in family:
        try:
            draws = garch.transform(draws, panel, card, args.asof, seed=args.seed + 90)
            method_notes.append("GARCH marginal volatility scaling")
        except Exception as exc:
            method_notes.append(f"GARCH skipped ({exc})")

    if "F3" in family and len(card["targets"]["asset_ids"]) > 1:
        try:
            draws = student_t_copula.couple(draws, panel, card, seed=args.seed + 100)
            method_notes.append("Student-t copula joint coupling")
        except Exception as exc:
            try:
                draws = gaussian_copula.couple(draws, panel, card, seed=args.seed + 100)
                method_notes.append("Gaussian copula fallback")
            except Exception as fallback_exc:
                method_notes.append(f"independent marginals (copula unavailable: {fallback_exc})")

    if "F4" in family:
        try:
            draws = garch.transform(draws, panel, card, args.asof, seed=args.seed + 200)
            method_notes.append("GARCH volatility scaling")
        except Exception as exc:
            method_notes.append(f"GARCH skipped ({exc})")
        draws = skewed_t.transform(draws, seed=args.seed + 201)
        method_notes.append("symmetric Student-t marginal tails")
        if len(card["targets"]["asset_ids"]) > 1:
            try:
                draws = student_t_copula.couple(draws, panel, card, seed=args.seed + 202)
                method_notes.append("Student-t copula joint tail dependence")
            except Exception as exc:
                method_notes.append(f"joint tail coupling skipped ({exc})")

    if "F2" in family or "F4" in family:
        signals = llm_signals.extract_signals(documents, args.asof, card)
        method_notes.append("LLM text conditioning is a neutral template")

    rationale_method = "; ".join(method_notes)
    if signals is None:
        text_summary = f"The text corpus was loaded ({len(documents)} dated documents) but was not used."
    else:
        text_summary = str(signals.get("evidence", "LLM template returned neutral signals."))

    out_path = _resolve_out(args.out)
    metadata = output_io.write_all(
        draws, card, args.asof, out_path, method=rationale_method,
        text_summary=text_summary,
    )
    print(f"wrote {out_path}, forecast_meta.json, forecast_rationale.md")
    print(f"  {len(metadata['asset_ids'])} asset(s) x "
          f"{len(metadata['horizons'])} horizon(s), {metadata['n_draws']} draws")
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"forecast failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
