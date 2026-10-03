"""Shared fixtures for the Phase 1 test suite.

The helpers here build the three inputs every Phase 1 test needs — a card, a
panel and a written submission — so each test states only the one thing it is
breaking.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.tools.card import CardSpec  # noqa: E402

#: Repository root, i.e. the directory holding `units/`.
REPO_ROOT = ROOT.parent

CARD_TEMPLATE = """\
schema_version = "2.0"

[task]
id    = "{unit_id}"
track = "forecasting"

[metadata]
category = "{family}"

[provenance]
data_cutoff = "{asof}"

[scoring.params]
representation = "samples"
{n_draws_line}

[targets]
asset_ids        = [{assets}]
horizons         = [{horizons}]
target_type      = "{target_type}"
value_unit       = "{value_unit}"
"""


def card_toml_text(
    unit_id: str = "t2-TEST-fixture",
    asof: str = "2024-01-02",
    assets: tuple[str, ...] = ("AAA",),
    horizons: tuple[int, ...] = (21,),
    target_type: str = "level",
    value_unit: str = "percent_per_annum",
    family: str = "T2-F1",
    n_draws_min: int | None = 200,
) -> str:
    """Render a minimal, valid `card.toml` as text."""
    return CARD_TEMPLATE.format(
        unit_id=unit_id,
        asof=asof,
        family=family,
        n_draws_line="" if n_draws_min is None else f"n_draws_min   = {n_draws_min}",
        assets=", ".join(f'"{a}"' for a in assets),
        horizons=", ".join(str(h) for h in horizons),
        target_type=target_type,
        value_unit=value_unit,
    )


def write_card(directory: pathlib.Path, text: str) -> pathlib.Path:
    """Write `text` to `<directory>/card.toml` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "card.toml"
    path.write_text(text, encoding="utf-8")
    return path


def make_spec(
    unit_id: str = "t2-TEST-fixture",
    asof: dt.date = dt.date(2024, 1, 2),
    asset_ids: tuple[str, ...] = ("AAA",),
    horizons: tuple[int, ...] = (21,),
    target_type: str = "level",
    value_unit: str = "percent_per_annum",
    n_draws_min: int = 200,
    family: str = "T2-F1",
) -> CardSpec:
    """Build a `CardSpec` directly, for tests that do not exercise parsing."""
    return CardSpec(
        unit_id=unit_id,
        asof=asof,
        asset_ids=asset_ids,
        horizons=horizons,
        target_type=target_type,
        value_unit=value_unit,
        n_draws_min=n_draws_min,
        family=family,
    )


def write_panel(
    directory: pathlib.Path,
    name: str,
    frames: dict[str, np.ndarray],
    start: dt.date = dt.date(2023, 1, 2),
    asset_column: str = "asset",
    dates: pd.DatetimeIndex | None = None,
) -> pathlib.Path:
    """Write a long-format panel parquet.

    Args:
        directory: Destination directory.
        name: File name, e.g. `panel.parquet`.
        frames: `{asset: values}`, each array one observation per date.
        start: First date, advanced by calendar day.
        asset_column: Spelling of the asset column, `asset` or `asset_id`.
        dates: Explicit date index; overrides `start` when given.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for asset, values in frames.items():
        if dates is None:
            index = pd.date_range(start, periods=len(values), freq="D")
        else:
            index = dates
        rows.append(pd.DataFrame({"date": index, asset_column: asset, "value": values}))
    frame = pd.concat(rows, ignore_index=True)
    path = directory / name
    frame.to_parquet(path, index=False)
    return path


def write_output(
    directory: pathlib.Path,
    card: CardSpec,
    n_draws: int = 200,
    frame: pd.DataFrame | None = None,
    meta: dict | None = None,
    rationale: str = "Fixture rationale.",
    extra_files: dict[str, str] | None = None,
) -> pathlib.Path:
    """Write a submission that passes every gate unless a test corrupts it.

    Args:
        directory: Destination directory.
        card: The card being answered.
        n_draws: Draw count, used to build both the frame and the sidecar.
        frame: Override the parquet's rows wholesale.
        meta: Override the sidecar wholesale.
        rationale: Contents of `forecast_rationale.md`.
        extra_files: Extra files to drop in the directory, e.g. to fail g0.

    Returns:
        The output directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if frame is None:
        frame = valid_frame(card, n_draws)
    frame.to_parquet(directory / "forecast.parquet", index=False)

    if meta is None:
        meta = valid_meta(card, n_draws)
    (directory / "forecast_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (directory / "forecast_rationale.md").write_text(rationale, encoding="utf-8")

    for name, body in (extra_files or {}).items():
        (directory / name).write_text(body, encoding="utf-8")
    return directory


def valid_meta(card: CardSpec, n_draws: int) -> dict:
    """The sidecar a correct pipeline would emit for `card`."""
    return {
        "unit_id": card.unit_id,
        "asof": card.asof.isoformat(),
        "representation": "samples",
        "asset_ids": list(card.asset_ids),
        "horizons": list(card.horizons),
        "n_draws": n_draws,
        "target": card.target_type,
    }


def valid_frame(card: CardSpec, n_draws: int, seed: int = 7) -> pd.DataFrame:
    """A parquet body that fills the card's grid exactly once."""
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(n_draws):
        for asset in card.asset_ids:
            for horizon in card.horizons:
                rows.append((draw, asset, horizon, float(rng.normal(1.0, 0.1))))
    return pd.DataFrame(rows, columns=["draw", "asset", "horizon", "value"])
