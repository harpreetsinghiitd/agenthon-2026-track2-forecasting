"""Panel loader. Self-contained: globs parquet/csv under the panels dir.

Handles two layouts:
  (a) long  : columns include {date, asset, value}
  (b) wide  : first column is a date-like index, other columns are assets

Frames from multiple files are concatenated column-wise, de-duplicated,
sorted by date, and truncated to `asof`.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def _load_one(f: Path) -> pd.DataFrame | None:
    if f.suffix == ".parquet":
        df = pd.read_parquet(f)
    elif f.suffix == ".csv":
        df = pd.read_csv(f)
    else:
        return None

    if df.empty:
        return None

    lower = {c: c.lower() for c in df.columns}
    cols_lower = {c.lower() for c in df.columns}

    # Long format
    if {"date", "asset", "value"} <= cols_lower:
        df = df.rename(columns=lower)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[~df["date"].isna()]
        wide = df.pivot_table(index="date", columns="asset",
                              values="value", aggfunc="last")
        return wide

    # Wide format: identify a date-like first column
    first = df.columns[0]
    if any(tok in first.lower() for tok in ("date", "time", "timestamp")):
        df = df.set_index(first)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.select_dtypes(include="number")
    if df.shape[1] == 0:
        return None
    return df


def load_wide(panels_dir: Path, asof: str) -> pd.DataFrame:
    panels_dir = Path(panels_dir)
    if not panels_dir.exists():
        raise FileNotFoundError(f"panels dir not found: {panels_dir}")

    files = sorted([*panels_dir.rglob("*.parquet"), *panels_dir.rglob("*.csv")])
    if not files:
        raise FileNotFoundError(f"no parquet/csv under {panels_dir}")

    frames = []
    for f in files:
        try:
            w = _load_one(f)
        except Exception:
            continue
        if w is not None and w.shape[1] > 0:
            frames.append(w)

    if not frames:
        raise ValueError(f"no usable numeric panels under {panels_dir}")

    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()]
    wide = wide.sort_index()

    asof_ts = pd.to_datetime(asof)
    wide = wide.loc[wide.index <= asof_ts]
    wide = wide.dropna(how="all")
    wide = wide.ffill().dropna()

    if wide.empty:
        raise ValueError(f"panel empty at or before asof={asof}")
    return wide
