"""Card parsing — the entry point of the pipeline.

Role in the pipeline
--------------------
`card.toml` is the frozen specification of one Track 2 unit: what is being
forecast, over which horizons, in what units, and as of which date.  This module
turns that file into a `CardSpec`, and every later stage reads its bounds from
the result: the scored grid is `asset_ids x horizons`, the leakage cutoff is
`asof`, and the RNG seed is derived from `unit_id`.  Nothing downstream re-reads
the TOML.

Field provenance
----------------
The mapping below is fixed by the authored units in `units/*/card.toml` and is
not configurable::

    unit_id      <- [task].id
    asof         <- [provenance].data_cutoff
    asset_ids    <- [targets].asset_ids
    horizons     <- [targets].horizons
    target_type  <- [targets].target_type
    value_unit   <- [targets].value_unit
    n_draws_min  <- [scoring.params].n_draws_min   (omitted on some cards -> 200)
    family       <- [metadata].category

`n_draws_min` is the one field an authored card may omit: the worked exemplar
`units/t2-EXAMPLE-ust-curve-1m/card.toml` carries no such key, so its absence
means the scorer's own floor of 200 rather than an error.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_N_DRAWS_MIN",
    "FAMILIES",
    "TARGET_TYPES",
    "CardParseError",
    "CardSpec",
    "parse_card",
]

#: The only family strings a card may declare.  Locked by the platform: a card
#: outside this set is an organizer defect, not a new category.
FAMILIES: frozenset[str] = frozenset({"T2-F1", "T2-F2", "T2-F3", "T2-F4"})

#: The only target types the metric understands, per the card specification.
TARGET_TYPES: frozenset[str] = frozenset({"level", "log_return", "spread"})

#: Used when a card omits `[scoring.params].n_draws_min`.  Matches the scorer's
#: hard floor (`qfbench2_track_forecasting.limits.ParseLimits.min_draws`).
DEFAULT_N_DRAWS_MIN: int = 200


class CardParseError(ValueError):
    """Raised when a `card.toml` is missing, malformed, or under-specified.

    Subclasses `ValueError` so callers that only guard on "bad input" can catch
    the built-in, while callers that want to distinguish a card defect from a
    panel defect can catch this class by name.
    """


@dataclass(frozen=True, slots=True)
class CardSpec:
    """The validated contents of one `card.toml`.

    Instances are immutable and hashable, so they are safe to pass to the pure
    functions in `harness.tools.panel` without defensive copying.

    Attributes:
        unit_id: `[task].id`; the identity written into `forecast_meta.json`.
        asof: `[provenance].data_cutoff`; the last date that may be observed.
        asset_ids: `[targets].asset_ids`; the asset axis of the scored grid.
        horizons: `[targets].horizons`; business-day horizons, grid order kept.
        target_type: One of `TARGET_TYPES`.
        value_unit: Unit of the emitted draws, e.g. `percent_per_annum`.
        n_draws_min: Floor on draws; `DEFAULT_N_DRAWS_MIN` when omitted.
        family: One of `FAMILIES`.
    """

    unit_id: str
    asof: dt.date
    asset_ids: tuple[str, ...]
    horizons: tuple[int, ...]
    target_type: str
    value_unit: str
    n_draws_min: int
    family: str

    @property
    def cell_count(self) -> int:
        """Number of scored cells, i.e. `len(asset_ids) * len(horizons)`."""
        return len(self.asset_ids) * len(self.horizons)


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the `[name]` table, or raise if it is absent or not a table."""
    value = raw.get(name)
    if not isinstance(value, dict):
        raise CardParseError(f"card.toml has no [{name}] table")
    return value


def _text(table: dict[str, Any], key: str, where: str) -> str:
    """Return a required non-empty string field, or raise."""
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CardParseError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def _string_list(table: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    """Return a required non-empty list of distinct non-empty strings."""
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise CardParseError(f"{where}.{key} must be a non-empty array of strings")
    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise CardParseError(f"{where}.{key} must contain only non-empty strings")
        items.append(entry.strip())
    if len(set(items)) != len(items):
        raise CardParseError(f"{where}.{key} repeats an entry; a grid axis is a set of labels")
    return tuple(items)


def _parse_asof(text: str) -> dt.date:
    """Parse `[provenance].data_cutoff` as an ISO calendar date."""
    try:
        return dt.date.fromisoformat(text)
    except ValueError as exc:
        raise CardParseError(
            f"[provenance].data_cutoff must be an ISO date (YYYY-MM-DD), got {text!r}"
        ) from exc


def _parse_horizons(table: dict[str, Any], where: str) -> tuple[int, ...]:
    """Return positive, distinct, integral business-day horizons in declared order."""
    value = table.get("horizons")
    if not isinstance(value, list) or not value:
        raise CardParseError(f"{where}.horizons must be a non-empty array of integers")
    horizons: list[int] = []
    for entry in value:
        # `bool` is an `int` subclass; `True` is not a horizon.
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise CardParseError(f"{where}.horizons must contain only integers, got {entry!r}")
        if entry < 1:
            raise CardParseError(f"{where}.horizons must be positive business days, got {entry}")
        horizons.append(entry)
    if len(set(horizons)) != len(horizons):
        raise CardParseError(f"{where}.horizons repeats an entry; a grid axis is a set of labels")
    return tuple(horizons)


def _parse_n_draws_min(raw: dict[str, Any]) -> int:
    """Return `n_draws_min`, accepting it under `[scoring.params]` or `[scoring]`.

    Absence is not an error — see the module docstring.  A declared value must be
    a positive integer; the scorer applies its own floor of 200 separately, so a
    card that declares less is honoured here and clamped in the pipeline rather
    than refused, because no card may be skipped.
    """
    params = raw.get("scoring", {}).get("params", {}) if isinstance(raw.get("scoring"), dict) else {}
    declared = params.get("n_draws_min") if isinstance(params, dict) else None
    if declared is None and isinstance(raw.get("scoring"), dict):
        declared = raw["scoring"].get("n_draws_min")
    if declared is None:
        return DEFAULT_N_DRAWS_MIN
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise CardParseError(f"[scoring.params].n_draws_min must be a positive integer, got {declared!r}")
    return declared


def parse_card(path: Path) -> CardSpec:
    """Parse and validate a `card.toml`.

    Args:
        path: Path to the unit's `card.toml`.

    Returns:
        The validated `CardSpec` for that unit.

    Raises:
        CardParseError: the file is absent, is not valid TOML, or is missing or
            malformed in any required field.  The message always names the TOML
            path of the offending field.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CardParseError(f"card not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CardParseError(f"card is not valid TOML: {path}: {exc}") from exc

    task = _table(raw, "task")
    provenance = _table(raw, "provenance")
    metadata = _table(raw, "metadata")
    targets = _table(raw, "targets")

    unit_id = _text(task, "id", "[task]")
    asof = _parse_asof(_text(provenance, "data_cutoff", "[provenance]"))
    asset_ids = _string_list(targets, "asset_ids", "[targets]")
    horizons = _parse_horizons(targets, "[targets]")
    target_type = _text(targets, "target_type", "[targets]")
    value_unit = _text(targets, "value_unit", "[targets]")
    family = _text(metadata, "category", "[metadata]")

    if target_type not in TARGET_TYPES:
        raise CardParseError(
            f"[targets].target_type must be one of {sorted(TARGET_TYPES)}, got {target_type!r}"
        )
    if family not in FAMILIES:
        raise CardParseError(
            f"[metadata].category must be one of {sorted(FAMILIES)}, got {family!r}"
        )

    return CardSpec(
        unit_id=unit_id,
        asof=asof,
        asset_ids=asset_ids,
        horizons=horizons,
        target_type=target_type,
        value_unit=value_unit,
        n_draws_min=_parse_n_draws_min(raw),
        family=family,
    )
