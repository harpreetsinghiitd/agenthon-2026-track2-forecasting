"""Load and validate the card, numerical panel, and frozen text corpus.

This module returns only panel observations and text documents on or before the
requested as-of date. It does not fit forecasting models or write forecast files.
"""

from __future__ import annotations

import json
import pathlib
import tomllib

import pandas as pd


def find_card(start: pathlib.Path) -> pathlib.Path:
    """Find ``card.toml`` at the unit root or in a nearby parent directory."""
    start = pathlib.Path(start)
    candidates = (start / "card.toml", start.parent / "card.toml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"card.toml not found at {start} or its parent")


def load_card(path: pathlib.Path) -> dict:
    """Parse a TOML card into a dictionary."""
    path = pathlib.Path(path)
    with pathlib.Path(path).open("rb") as f:
        card = tomllib.load(f)
    # Monthly target step counts require explicit observation-period mappings.
    # Keep the optional sidecar internal so statistical models can use the shared
    # toolkit helper without changing the authored card or output metadata.
    spec_path = path.parent / "forecast_spec.json"
    if spec_path.is_file():
        card["_forecast_spec"] = json.loads(spec_path.read_text(encoding="utf-8"))
    return card


def get_panel_id(card: dict) -> str:
    """Return the first declared panel ID from the Track 2 card structure."""
    panel_ids = card.get("panels", {}).get("panel_ids", [])
    if not panel_ids:
        raise KeyError("card must declare at least one panels.panel_ids entry")
    return str(panel_ids[0])


def get_n_draws_min(card: dict, default: int = 200) -> int:
    """Read the card's draw minimum from ``[scoring.params]``."""
    scoring = card.get("scoring", {})
    params = scoring.get("params", {}) if isinstance(scoring, dict) else {}
    return int(params.get("n_draws_min", default))


def _panel_candidates(unit_dir: pathlib.Path, panel_id: str) -> list[pathlib.Path]:
    """List supported panel locations, including the common unit-root layout."""
    return [
        unit_dir / f"{panel_id}.parquet",
        unit_dir / "panels" / f"{panel_id}.parquet",
    ]


def load_panel(unit_dir: pathlib.Path, panel_id: str, asof: str) -> pd.DataFrame:
    """Load the requested parquet panel and enforce the as-of cutoff.

    The authored cards store panel parquet files at the unit root. A ``panels/``
    subdirectory is also supported for compatibility with harness mounts.
    """
    unit_dir = pathlib.Path(unit_dir)
    panel_path = next(
        (path for path in _panel_candidates(unit_dir, panel_id) if path.is_file()),
        None,
    )
    if panel_path is None:
        matches = sorted(unit_dir.glob("*.parquet"))
        if len(matches) == 1:
            panel_path = matches[0]
        else:
            raise FileNotFoundError(
                f"could not locate {panel_id}.parquet under {unit_dir}"
            )

    panel = pd.read_parquet(panel_path)
    required = {"date", "asset", "value"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel {panel_path} is missing columns: {sorted(missing)}")

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    if panel["date"].isna().any():
        raise ValueError(f"panel {panel_path} contains invalid dates")
    panel = panel.loc[panel["date"] <= pd.Timestamp(asof)].copy()
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    panel = panel.dropna(subset=["asset", "value"])
    if panel.empty:
        raise ValueError(f"panel {panel_path} has no usable observations by {asof}")

    if "panel_id" not in panel.columns:
        panel["panel_id"] = panel_id
    return panel.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)


def load_text(text_dir: pathlib.Path, asof: str) -> list[dict]:
    """Load only corpus-indexed text documents dated on or before ``asof``.

    A corpus index is required so release dates (rather than dates embedded in
    document names or meeting dates) control the cutoff. Each returned item has
    ``name``, ``timestamp``, and ``text`` fields.
    """
    text_dir = pathlib.Path(text_dir)
    index_path = text_dir / "corpus_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"frozen text corpus index not found: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(asof)
    docs = []
    for record in index.get("documents", []):
        timestamp = pd.Timestamp(record["timestamp"])
        if timestamp > cutoff:
            continue

        name = record["file"]
        path = text_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"indexed corpus document is missing: {path}")
        docs.append({
            "name": name,
            "timestamp": timestamp.date().isoformat(),
            "text": path.read_text(encoding="utf-8", errors="replace"),
        })

    return docs
