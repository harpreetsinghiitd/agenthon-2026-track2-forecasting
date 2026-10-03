"""The only writer — turns a draw array into the three scored files.

Role in the pipeline
--------------------
Every draw in `forecast.parquet` arrives here as a `(n_draws, n_assets,
n_horizons)` array produced by `harness.tools.simulate`, and leaves as rows of
`draw, asset, horizon, value`.  No other module in this codebase may touch the
output directory, and nothing that reads the corpus or calls a model may write
to it: that separation is what makes the pipeline gate-compliant by construction
rather than by review.

The write is verified before it is trusted.  Files are written, `validate_gates`
runs against the directory that actually holds them, and a failure unlinks them
again — so an invalid submission never survives, and a successful call means the
bytes on disk passed all four gates, not that they were merely intended to.

Shapes
------
`draws[d, i, j]` is the value of asset `i` at horizon `j` in draw `d`, in the
card's own `value_unit`.  The parquet's row order is draw-major, then asset, then
horizon; `g3` proves the grid is covered exactly once, so the order cannot hide a
duplicate or a hole.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from harness.tools.card import CardSpec
from harness.validate.preflight import (
    FORECAST_COLUMNS,
    MIN_DRAWS,
    GateFailure,
    GateReport,
    validate_gates,
)

__all__ = ["WriteError", "build_meta", "write_deliverables"]

_RATIONALE_NAME = "forecast_rationale.md"


class WriteError(ValueError):
    """Raised when a submission cannot be written: bad shapes, or a failed gate."""


def build_meta(card: CardSpec, n_draws: int) -> dict[str, Any]:
    """Build the `forecast_meta.json` payload for a card.

    Keeping the sidecar's shape in one place is what stops `g1_schema` and
    `g2_cutoff_resource` from being satisfied by two different spellings of the
    same field.

    Args:
        card: The card being answered.
        n_draws: Number of draws actually simulated; must satisfy the floor.

    Returns:
        The sidecar as a plain dict, ready for `json.dumps`.

    Raises:
        WriteError: `n_draws` is below the scorer's floor.
    """
    if n_draws < MIN_DRAWS:
        raise WriteError(f"n_draws={n_draws} is below the scorer's floor of {MIN_DRAWS}")
    return {
        "unit_id": card.unit_id,
        "asof": card.asof.isoformat(),
        "representation": "samples",
        "asset_ids": list(card.asset_ids),
        "horizons": list(card.horizons),
        "n_draws": int(n_draws),
        "target": card.target_type,
    }


def _flatten(draws: np.ndarray, card: CardSpec) -> pd.DataFrame:
    """Explode a `(n_draws, n_assets, n_horizons)` array into the parquet's rows.

    Args:
        draws: The simulated draws.
        card: The card, supplying asset and horizon labels in grid order.

    Returns:
        A frame with columns exactly `FORECAST_COLUMNS`.

    Raises:
        WriteError: the array does not match the card's grid shape, or carries a
            non-finite value.
    """
    n_draws, n_assets, n_horizons = draws.shape
    if (n_assets, n_horizons) != (len(card.asset_ids), len(card.horizons)):
        raise WriteError(
            f"draws have shape {draws.shape} but the card's grid is "
            f"({n_draws}, {len(card.asset_ids)}, {len(card.horizons)})"
        )
    if not np.isfinite(draws).all():
        raise WriteError("draws contain NaN or infinity; g3 would refuse them")

    draw_column = np.repeat(np.arange(n_draws), n_assets * n_horizons)
    asset_column = np.tile(np.repeat(np.arange(n_assets), n_horizons), n_draws)
    horizon_column = np.tile(np.arange(n_horizons), n_draws * n_assets)

    return pd.DataFrame(
        {
            "draw": draw_column.astype(np.int64),
            "asset": np.asarray(card.asset_ids, dtype=object)[asset_column],
            "horizon": np.asarray(card.horizons, dtype=np.int64)[horizon_column],
            "value": draws.reshape(-1).astype(np.float64),
        },
        columns=list(FORECAST_COLUMNS),
    )


def write_deliverables(
    draws: np.ndarray,
    meta: Mapping[str, Any],
    rationale: str,
    output_dir: Path,
    card: CardSpec,
) -> GateReport:
    """Write the three deliverables and prove they pass every gate.

    Files are written into `output_dir`, then validated in place.  If any gate
    refuses, every file this call wrote is removed before the `GateFailure`
    propagates, so the output directory is never left holding a partial or
    inadmissible submission.

    Args:
        draws: `(n_draws, n_assets, n_horizons)` simulated values.
        meta: The sidecar payload, normally from `build_meta`.
        rationale: Markdown derivation; must be non-blank.
        output_dir: Destination directory, created if absent.
        card: The card being answered, used to validate.

    Returns:
        The `GateReport` from validating the bytes just written.

    Raises:
        WriteError: the rationale is blank or the draw array is unusable.
        GateFailure: a gate refused the written submission; the files have been
            removed by the time this propagates.
    """
    if not rationale.strip():
        raise WriteError("rationale is blank; g1_schema requires at least one visible character")

    frame = _flatten(draws, card)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = [output_dir / "forecast.parquet", output_dir / "forecast_meta.json", output_dir / _RATIONALE_NAME]

    try:
        frame.to_parquet(written[0], index=False)
        written[1].write_text(_dump_meta(meta), encoding="utf-8")
        written[2].write_text(rationale, encoding="utf-8")

        return validate_gates(output_dir, card)
    except GateFailure:
        for path in written:
            path.unlink(missing_ok=True)
        raise


def _dump_meta(meta: Mapping[str, Any]) -> str:
    """Serialise the sidecar deterministically: sorted keys, fixed separators.

    Byte-identical re-runs are part of the determinism contract, so the JSON must
    not depend on dict insertion order.
    """
    return json.dumps(dict(meta), sort_keys=True, indent=2, separators=(",", ": ")) + "\n"
