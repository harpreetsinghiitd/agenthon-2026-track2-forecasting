"""Panel loading and the summary statistics that set the forecast's scale.

Role in the pipeline
--------------------
The numeric panel is the text-blind backbone.  Before any model is fitted, three
numbers are read off it and they bound everything that follows:

* the **anchor** — the last observation at or before `asof`, from which every
  simulated path starts;
* the **horizon spread** — the standard deviation of overlapping h-business-day
  changes, which is the width of an honest random walk and the yardstick the
  deterministic bridge caps its text-driven drift against (0.15 x this value);
* the **current volatility** — an EWMA of daily changes, which says whether the
  recent regime is calmer or more stressed than the full history.

Every read is filtered to `date <= asof` here, at the boundary, so no caller can
accidentally hand post-asof rows to a model or to a prompt.

Panel format
------------
Long format, one row per (date, asset): columns `date`, `asset`, `value`.  The
asset column is spelled `asset` in the authored units and `asset_id` in some
downstream exports, so both are accepted.  `date` and `value` are required as
named.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_EWMA_LAMBDA",
    "PanelError",
    "compute_ewma_vol",
    "compute_horizon_sd",
    "get_last_observation",
    "read_panel",
]

#: RiskMetrics daily decay.  0.94 is the published default for daily data and is
#: the value the playbook's "recent vol usually dominates" guidance assumes.
DEFAULT_EWMA_LAMBDA: float = 0.94

#: Accepted spellings of the asset column, in preference order.
_ASSET_COLUMNS: tuple[str, ...] = ("asset", "asset_id")

#: The canonical column order of every frame this module returns.
_OUTPUT_COLUMNS: tuple[str, ...] = ("date", "asset", "value")


class PanelError(ValueError):
    """Raised when a panel file is absent, unreadable, or has no usable history."""


def _asset_column(frame: pd.DataFrame) -> str | None:
    """Return the asset column's name, or `None` if the frame has no asset axis."""
    for name in _ASSET_COLUMNS:
        if name in frame.columns:
            return name
    return None


def _require_columns(frame: pd.DataFrame, path: Path) -> None:
    """Raise unless the frame carries the two columns every panel must have."""
    missing = [name for name in ("date", "value") if name not in frame.columns]
    if missing:
        raise PanelError(f"{path.name} is missing required column(s) {missing}")


def _canonicalise(
    subset: pd.DataFrame, asset: str, asof: dt.date, path: Path
) -> pd.DataFrame:
    """Filter one asset's rows to `date <= asof` and put them in canonical order.

    Args:
        subset: Rows already selected to a single asset.
        asset: The asset id, re-stamped so the column never varies in spelling.
        asof: Last observable date, inclusive.
        path: Source file, used only for error messages.

    Returns:
        Columns `date`, `asset`, `value`, sorted by date, one row per date.

    Raises:
        PanelError: no row survives the cutoff, or the cutoff leaves too little
            history to be usable.
    """
    dates = pd.to_datetime(subset["date"], errors="coerce")
    values = pd.to_numeric(subset["value"], errors="coerce")
    frame = pd.DataFrame({"date": dates, "value": values})
    frame = frame.dropna(subset=["date", "value"])
    # `dropna` removes NaN but not infinity, and an infinite observation would
    # poison every statistic downstream while still passing `isfinite` checks
    # that only look at the result.
    frame = frame[np.isfinite(frame["value"])]
    frame = frame[frame["date"] <= pd.Timestamp(asof)]
    if frame.empty:
        raise PanelError(f"{path.name} holds no observation of {asset!r} at or before {asof}")
    frame = frame.sort_values("date").drop_duplicates(subset="date", keep="last")
    frame["asset"] = asset
    return frame.loc[:, list(_OUTPUT_COLUMNS)].reset_index(drop=True)


def read_panel(unit_dir: Path, asset: str, asof: dt.date) -> pd.DataFrame:
    """Load the first panel in `unit_dir` that carries `asset`, filtered to `asof`.

    Files are tried in sorted order so the choice is deterministic across runs.
    A file is a match when it has an asset column containing `asset`; spellings
    `asset` and `asset_id` are both accepted.

    Args:
        unit_dir: Directory holding the unit's `*.parquet` panel files.
        asset: Asset id to select, exactly as spelled in `card.toml [targets]`.
        asof: Last observable date; rows after it are dropped, never seen.

    Returns:
        A frame with columns `date`, `asset`, `value`, ascending by date.

    Raises:
        PanelError: no `*.parquet` exists, none carries the asset, or the
            surviving history is empty.
    """
    candidates = sorted(unit_dir.glob("*.parquet"))
    if not candidates:
        raise PanelError(f"no *.parquet panel found in {unit_dir}")

    for path in candidates:
        frame = pd.read_parquet(path)
        column = _asset_column(frame)
        if column is None:
            continue
        _require_columns(frame, path)
        subset = frame[frame[column].astype(str) == asset]
        if subset.empty:
            continue
        return _canonicalise(subset, asset, asof, path)

    raise PanelError(f"no panel in {unit_dir} contains asset {asset!r}")


def get_last_observation(panel: pd.DataFrame) -> tuple[dt.date, float]:
    """Return the anchor `(date, value)`: the last row at or before `asof`.

    Args:
        panel: A frame from `read_panel`, already filtered and sorted.

    Returns:
        The date of the final observation and its value.

    Raises:
        PanelError: the panel is empty.
    """
    if panel.empty:
        raise PanelError("panel is empty; there is no observation to anchor on")
    row = panel.iloc[-1]
    return row["date"].date(), float(row["value"])


def _changes(series: pd.Series, target_type: str) -> pd.Series:
    """Return the per-step series whose h-fold aggregate is the h-day change.

    For a `level` or `spread` target the step is the first difference; for a
    `log_return` target each observation already *is* a daily change, so the
    step is the value itself.
    """
    if target_type == "log_return":
        return series.astype(float)
    return series.astype(float).diff().dropna()


def compute_horizon_sd(panel: pd.DataFrame, horizon: int, target_type: str = "level") -> float:
    """Return the standard deviation of overlapping `horizon`-step changes.

    This is the baseline spread: the width a random walk anchored on the last
    observation would honestly carry.  It is also the yardstick the deterministic
    bridge caps text-driven drift against.

    Args:
        panel: A frame from `read_panel`, filtered to `asof`.
        horizon: Horizon in business days, i.e. in panel steps.
        target_type: `level`, `spread` or `log_return`; see `_changes`.

    Returns:
        The sample standard deviation (ddof=1) of the h-step changes.

    Raises:
        PanelError: the panel holds fewer than `horizon + 2` steps, so no
            overlapping change exists.
    """
    if horizon < 1:
        raise PanelError(f"horizon must be a positive number of business days, got {horizon}")
    series = panel["value"]
    if len(series) < horizon + 2:
        raise PanelError(
            f"panel holds {len(series)} observations, too few for a {horizon}-step change"
        )
    if target_type == "log_return":
        changes = series.astype(float).rolling(window=horizon).sum().dropna()
    else:
        changes = series.astype(float).diff(periods=horizon).dropna()
    if len(changes) < 2:
        raise PanelError(f"panel yields {len(changes)} overlapping {horizon}-step changes; need 2")
    sd = float(changes.std(ddof=1))
    if not np.isfinite(sd):
        raise PanelError(f"horizon spread is not finite for horizon {horizon}")
    return sd


def compute_ewma_vol(
    panel: pd.DataFrame, lam: float = DEFAULT_EWMA_LAMBDA, target_type: str = "level"
) -> float:
    """Return the current per-step volatility from an EWMA of daily changes.

    The recursion is the standard RiskMetrics one,
    `var_t = lam * var_{t-1} + (1 - lam) * r_{t-1}^2`, seeded on the first squared
    change.  The seed's influence decays as `lam**n`, so with a few thousand
    observations the returned value is driven by recent data only.

    Args:
        panel: A frame from `read_panel`, filtered to `asof`.
        lam: Decay in `[0, 1)`; higher means a longer memory.
        target_type: `level`, `spread` or `log_return`; see `_changes`.

    Returns:
        The square root of the final variance, in the panel's own units.

    Raises:
        PanelError: the panel holds fewer than two steps, or `lam` is out of range.
    """
    if not 0.0 <= lam < 1.0:
        raise PanelError(f"EWMA lambda must be in [0, 1), got {lam}")
    changes = _changes(panel["value"], target_type).to_numpy(dtype=float)
    if len(changes) < 2:
        raise PanelError(f"panel yields {len(changes)} daily changes; need at least 2")
    variance = float(changes[0] ** 2)
    for change in changes[1:]:
        variance = lam * variance + (1.0 - lam) * float(change) ** 2
    if not np.isfinite(variance) or variance < 0.0:
        raise PanelError("EWMA variance is not a finite non-negative number")
    return float(np.sqrt(variance))
