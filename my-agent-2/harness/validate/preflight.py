"""Preflight validation — the four admissibility gates, mirrored locally.

Role in the pipeline
--------------------
`validate_gates` is the last thing that runs before a submission leaves the
container, and the only thing standing between a buggy pipeline and a card that
scores 4.0.  It reproduces the organizer's gates `g0`-`g3` in code that runs
locally, so a failure is caught here rather than at scoring time.

The four gates, in the order the scorer applies them:

``g0_integrity``
    Exactly three regular files sit in the output directory, named
    `forecast.parquet`, `forecast_meta.json` and `forecast_rationale.md`.  A log
    file or a `__pycache__` beside them fails the card.
``g1_schema``
    The parquet's column set is *exactly* `draw, asset, horizon, value` — an
    extra column is refused, not ignored.  The sidecar validates against the
    declared shape, `n_draws` is an integer inside `[200, 20000]`, and the
    rationale holds at least one non-whitespace character.
``g2_cutoff_resource``
    The sidecar's `asof`, `unit_id` and `target` bind to the trusted card.
``g3_domain_semantics``
    The declared grid is the card's grid, in the card's order, with no repeated
    label; and the parquet fills that grid exactly once — no duplicate cell, no
    missing cell, no non-finite value, draw ids contiguous from zero.

Nothing here prints.  Failures are raised as `GateFailure` and successes are
returned as a `GateReport`, so callers decide what to log.

Provenance of the constants
---------------------------
`PARTICIPANT_FILES`, `FORECAST_COLUMNS`, `MIN_DRAWS`, `MAX_DRAWS` and the gate
names are copied from the organizer's scorer
(`qfbench2_track_forecasting/scoring.py`, `grid.py`, `limits.py`) and from the
authored `units/*/card.toml`.  Where the prose design document disagrees — it
describes a `unit_id` column inside the parquet, which the scorer refuses as an
extra column — the scorer wins, because the scorer is what grades the card.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harness.tools.card import CardSpec

__all__ = [
    "FORECAST_COLUMNS",
    "GATES",
    "MAX_DRAWS",
    "MIN_DRAWS",
    "META_REQUIRED_KEYS",
    "PARTICIPANT_FILES",
    "REPRESENTATION",
    "GateFailure",
    "GateReport",
    "validate_gates",
]

#: The three files a submission consists of, and the whole of it.
PARTICIPANT_FILES: tuple[str, ...] = (
    "forecast.parquet",
    "forecast_meta.json",
    "forecast_rationale.md",
)

#: The parquet's exact column set.  Extra columns are refused, never ignored.
FORECAST_COLUMNS: tuple[str, ...] = ("draw", "asset", "horizon", "value")

#: The only representation this scorer implements; `parametric` has no loader.
REPRESENTATION: str = "samples"

#: Draw-count bounds, mirroring `ParseLimits.min_draws` / `.max_draws`.
MIN_DRAWS: int = 200
MAX_DRAWS: int = 20_000

#: Every key `forecast_meta.json` must declare.
META_REQUIRED_KEYS: tuple[str, ...] = (
    "unit_id",
    "asof",
    "representation",
    "asset_ids",
    "horizons",
    "n_draws",
    "target",
)

#: The gates, in the order the scorer applies them.
GATES: tuple[str, ...] = (
    "g0_integrity",
    "g1_schema",
    "g2_cutoff_resource",
    "g3_domain_semantics",
)


@dataclass
class GateFailure(Exception):
    """A single gate's refusal.

    Attributes:
        gate: The gate that refused, e.g. `g1_schema`.
        reason: What it found, phrased for a log line rather than a stack trace.
    """

    gate: str
    reason: str

    def __post_init__(self) -> None:
        Exception.__init__(self, self.gate, self.reason)

    def __str__(self) -> str:
        return f"{self.gate}: {self.reason}"


@dataclass(frozen=True, slots=True)
class GateReport:
    """Evidence that every gate passed.

    Attributes:
        unit_id: The unit that was validated.
        n_draws: Distinct draws the submission declares.
        cell_count: `len(asset_ids) * len(horizons)`.
        gates: The gates that passed, in the order they ran.
    """

    unit_id: str
    n_draws: int
    cell_count: int
    gates: tuple[str, ...]


def _read_meta(output_dir: Path) -> dict[str, Any]:
    """Load and shape-check `forecast_meta.json`.

    Args:
        output_dir: Directory holding the submission.

    Returns:
        The parsed sidecar.

    Raises:
        GateFailure: `g1_schema` if the file is unreadable, is not a JSON object,
            or is missing any required key.
    """
    path = output_dir / "forecast_meta.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateFailure("g1_schema", f"forecast_meta.json is unreadable: {exc}") from exc
    if not isinstance(meta, dict):
        raise GateFailure("g1_schema", "forecast_meta.json must be a JSON object")
    missing = [key for key in META_REQUIRED_KEYS if key not in meta]
    if missing:
        raise GateFailure("g1_schema", f"forecast_meta.json is missing key(s) {missing}")
    if meta["representation"] != REPRESENTATION:
        raise GateFailure(
            "g1_schema",
            f"representation must be {REPRESENTATION!r}, got {meta['representation']!r}",
        )
    return meta


def _read_draws(output_dir: Path) -> pd.DataFrame:
    """Load `forecast.parquet` and check its column set.

    Args:
        output_dir: Directory holding the submission.

    Returns:
        The frame, with columns exactly `FORECAST_COLUMNS`.

    Raises:
        GateFailure: `g1_schema` if the file is unreadable or its columns are not
            exactly the four required ones.
    """
    path = output_dir / "forecast.parquet"
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as exc:
        raise GateFailure("g1_schema", f"forecast.parquet is unreadable: {exc}") from exc
    got = tuple(frame.columns)
    if got != FORECAST_COLUMNS:
        raise GateFailure(
            "g1_schema",
            f"forecast.parquet columns must be exactly {list(FORECAST_COLUMNS)}, got {list(got)}",
        )
    return frame


def _check_g0(output_dir: Path) -> None:
    """Gate g0: exactly the three participant files, and nothing else.

    Raises:
        GateFailure: `g0_integrity` if the directory is missing, or its entries
            are not exactly `PARTICIPANT_FILES`.
    """
    if not output_dir.is_dir():
        raise GateFailure("g0_integrity", f"output directory does not exist: {output_dir}")
    entries = sorted(entry.name for entry in output_dir.iterdir())
    if entries != sorted(PARTICIPANT_FILES):
        extra = [name for name in entries if name not in PARTICIPANT_FILES]
        absent = [name for name in PARTICIPANT_FILES if name not in entries]
        raise GateFailure(
            "g0_integrity",
            f"output must hold exactly {list(PARTICIPANT_FILES)}; "
            f"unexpected {extra}, missing {absent}",
        )


def _check_g1(output_dir: Path, meta: dict[str, Any]) -> int:
    """Gate g1: exact parquet columns, a sane draw count, a non-blank rationale.

    Args:
        output_dir: Directory holding the submission.
        meta: The already-shaped sidecar.

    Returns:
        The declared `n_draws`.

    Raises:
        GateFailure: `g1_schema` for any of the three checks above.
    """
    declared = meta["n_draws"]
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise GateFailure("g1_schema", "forecast_meta.json n_draws must be an integer")
    if not MIN_DRAWS <= declared <= MAX_DRAWS:
        raise GateFailure(
            "g1_schema", f"n_draws={declared} is outside the contract range [{MIN_DRAWS}, {MAX_DRAWS}]"
        )
    if not isinstance(meta["asset_ids"], list) or not isinstance(meta["horizons"], list):
        raise GateFailure("g1_schema", "asset_ids and horizons must each be an array")
    rationale = output_dir / "forecast_rationale.md"
    try:
        body = rationale.read_text(encoding="utf-8")
    except OSError as exc:
        raise GateFailure("g1_schema", f"forecast_rationale.md is unreadable: {exc}") from exc
    if not body.strip():
        raise GateFailure("g1_schema", "forecast_rationale.md is present but empty")
    return declared


def _check_g2(meta: dict[str, Any], card: CardSpec) -> None:
    """Gate g2: bind the sidecar's declaration to the trusted card.

    Raises:
        GateFailure: `g2_cutoff_resource` if `asof`, `unit_id` or `target`
            disagrees with the card.
    """
    if meta["asof"] != card.asof.isoformat():
        raise GateFailure(
            "g2_cutoff_resource",
            f"forecast_meta.json asof={meta['asof']!r} != card data_cutoff={card.asof.isoformat()!r}",
        )
    if meta["unit_id"] != card.unit_id:
        raise GateFailure(
            "g2_cutoff_resource",
            f"forecast_meta.json unit_id={meta['unit_id']!r} != card id={card.unit_id!r}",
        )
    if meta["target"] != card.target_type:
        raise GateFailure(
            "g2_cutoff_resource",
            f"forecast_meta.json target={meta['target']!r} != card target_type={card.target_type!r}",
        )


def _check_grid_axis(declared: list[Any], expected: tuple[Any, ...], axis: str) -> None:
    """Require a sidecar axis to equal the card's, in order and without repeats.

    Raises:
        GateFailure: `g3_domain_semantics` on a repeated label or a mismatch.
    """
    if len(set(map(str, declared))) != len(declared):
        raise GateFailure(
            "g3_domain_semantics",
            f"forecast_meta.json {axis} repeats a label; a grid axis is a set of distinct labels",
        )
    if tuple(declared) != expected:
        raise GateFailure(
            "g3_domain_semantics",
            f"forecast_meta.json {axis}={list(declared)} != card {axis}={list(expected)} "
            "(order is part of the contract)",
        )


def _check_g3(frame: pd.DataFrame, meta: dict[str, Any], card: CardSpec) -> None:
    """Gate g3: the parquet fills the card's grid exactly once, with finite values.

    Raises:
        GateFailure: `g3_domain_semantics` for a mismatched grid axis, a wrong row
            count, a draw id outside `[0, n_draws)`, a non-numeric or non-finite
            value, or a duplicated/absent cell.
    """
    _check_grid_axis(meta["asset_ids"], card.asset_ids, "asset_ids")
    _check_grid_axis(meta["horizons"], card.horizons, "horizons")

    n_draws = int(meta["n_draws"])
    cell_count = card.cell_count
    expected_rows = n_draws * cell_count
    if len(frame) != expected_rows:
        raise GateFailure(
            "g3_domain_semantics",
            f"forecast.parquet holds {len(frame)} rows; the grid is "
            f"{n_draws} draws x {cell_count} cells = {expected_rows}",
        )

    draws = pd.to_numeric(frame["draw"], errors="coerce").to_numpy()
    if not np.isfinite(draws).all() or (draws != np.floor(draws)).any():
        raise GateFailure("g3_domain_semantics", "forecast.parquet draw ids must be integers")
    draw_ids = draws.astype(np.int64)
    if draw_ids.min() < 0 or draw_ids.max() >= n_draws:
        raise GateFailure(
            "g3_domain_semantics",
            f"draw ids must lie in [0, {n_draws}); found range [{draw_ids.min()}, {draw_ids.max()}]",
        )

    asset_index = {asset: i for i, asset in enumerate(card.asset_ids)}
    horizon_index = {horizon: i for i, horizon in enumerate(card.horizons)}
    assets = frame["asset"].astype(str).to_numpy()
    horizons = pd.to_numeric(frame["horizon"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(horizons).all():
        raise GateFailure("g3_domain_semantics", "forecast.parquet horizon values must be numeric")
    horizon_ids = horizons.astype(np.int64)
    unknown_assets = int(sum(1 for name in assets if name not in asset_index))
    if unknown_assets:
        raise GateFailure(
            "g3_domain_semantics",
            f"{unknown_assets} row(s) name an asset outside the card's grid",
        )
    unknown_horizons = int(sum(1 for h in horizon_ids if h not in horizon_index))
    if unknown_horizons:
        raise GateFailure(
            "g3_domain_semantics",
            f"{unknown_horizons} row(s) name a horizon outside the card's grid",
        )

    values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise GateFailure(
            "g3_domain_semantics",
            f"{nonfinite} non-numeric or non-finite value(s); NaN and infinity never reach the metric",
        )

    asset_codes = np.array([asset_index[name] for name in assets], dtype=np.int64)
    horizon_codes = np.array([horizon_index[int(h)] for h in horizon_ids], dtype=np.int64)
    flat = draw_ids * cell_count + asset_codes * len(card.horizons) + horizon_codes
    occupancy = np.bincount(flat, minlength=expected_rows)
    duplicates = int(np.count_nonzero(occupancy > 1))
    absent = int(np.count_nonzero(occupancy == 0))
    if duplicates or absent:
        raise GateFailure(
            "g3_domain_semantics",
            f"parquet does not cover its grid exactly once: {duplicates} cell(s) claimed more "
            f"than once and {absent} cell(s) absent",
        )


def validate_gates(output_dir: Path, card: CardSpec) -> GateReport:
    """Run the four admissibility gates over a written submission.

    Gates run in the scorer's order, so the first failure reported is the one the
    organizer would have reported.

    Args:
        output_dir: Directory the submission was written to.
        card: The card the submission claims to answer.

    Returns:
        A `GateReport` naming the unit, its draw count, its cell count and the
        gates that passed.

    Raises:
        GateFailure: the first gate that refused, carrying that gate's name.
    """
    _check_g0(output_dir)
    meta = _read_meta(output_dir)
    frame = _read_draws(output_dir)
    n_draws = _check_g1(output_dir, meta)
    _check_g2(meta, card)
    _check_g3(frame, meta, card)
    return GateReport(
        unit_id=card.unit_id,
        n_draws=n_draws,
        cell_count=card.cell_count,
        gates=GATES,
    )
