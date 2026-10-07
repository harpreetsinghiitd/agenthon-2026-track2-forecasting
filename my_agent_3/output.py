"""Executive summary: serialize forecast draws in the Track 2 output contract.

This module writes the parquet forecast, metadata sidecar, and human-readable
rationale. It does not fit models, read inputs, or remove existing files.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd


def samples_to_long(
    draws: np.ndarray,
    asset_ids: list[str],
    horizons: list[int],
) -> pd.DataFrame:
    """Convert draws shaped ``[draw, asset, horizon]`` to the required table."""
    draws = np.asarray(draws)
    expected_tail = (len(asset_ids), len(horizons))
    if draws.ndim != 3 or draws.shape[1:] != expected_tail:
        raise ValueError(
            f"draws must have shape [n_draws, {expected_tail[0]}, "
            f"{expected_tail[1]}], got {draws.shape}"
        )
    if draws.shape[0] == 0:
        raise ValueError("forecast must contain at least one draw")
    if not np.isfinite(draws).all():
        raise ValueError("forecast draws contain NaN or infinite values")

    draw_idx, asset_idx, horizon_idx = np.indices(draws.shape)
    return pd.DataFrame({
        "draw": draw_idx.ravel().astype(np.int32),
        "asset": pd.array(
            [asset_ids[i] for i in asset_idx.ravel()], dtype="string"
        ),
        "horizon": np.asarray(horizons, dtype=np.int32)[horizon_idx.ravel()],
        "value": draws.astype(np.float64, copy=False).ravel(),
    })


def build_metadata(card: dict, asof: str, n_draws: int) -> dict:
    """Build metadata using the organizer's required ``unit_id`` key."""
    targets = card["targets"]
    return {
        "unit_id": card["task"]["id"],
        "asof": asof,
        "representation": "samples",
        "asset_ids": list(targets["asset_ids"]),
        "horizons": [int(h) for h in targets["horizons"]],
        "n_draws": int(n_draws),
        "target": targets["target_type"],
    }


def build_rationale(
    metadata: dict,
    method: str,
    text_summary: str = "The text corpus was not used in this forecast.",
) -> str:
    """Return a non-empty rationale that describes the method actually used."""
    assets = ", ".join(metadata["asset_ids"])
    horizons = ", ".join(str(h) for h in metadata["horizons"])
    method = method.strip() or "Numerical forecasting model."
    text_summary = text_summary.strip() or "No text adjustment was applied."
    return (
        f"# Forecast rationale — {metadata['unit_id']}\n\n"
        f"As of {metadata['asof']}, the agent generated "
        f"{metadata['n_draws']} sample draws for {assets} at horizon(s) "
        f"{horizons}. The target type is `{metadata['target']}`.\n\n"
        f"## Method\n\n{method}\n\n"
        f"## Text contribution\n\n{text_summary}\n"
    )


def write_all(
    draws: np.ndarray,
    card: dict,
    asof: str,
    out_path: pathlib.Path,
    method: str,
    text_summary: str = "The text corpus was not used in this forecast.",
) -> dict:
    """Write the three required files beside ``out_path`` and return metadata."""
    out_path = pathlib.Path(out_path)
    if out_path.name != "forecast.parquet":
        raise ValueError("--out must name the required file forecast.parquet")

    targets = card["targets"]
    asset_ids = list(targets["asset_ids"])
    horizons = [int(h) for h in targets["horizons"]]
    draws = np.asarray(draws)
    if draws.ndim != 3:
        raise ValueError(f"draws must be three-dimensional, got {draws.shape}")
    n_draws = int(draws.shape[0])

    table = samples_to_long(draws, asset_ids, horizons)
    metadata = build_metadata(card, asof, n_draws)
    rationale = build_rationale(metadata, method, text_summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    (out_path.parent / "forecast_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (out_path.parent / "forecast_rationale.md").write_text(
        rationale, encoding="utf-8"
    )
    return metadata
