"""Load the card, panel, and text corpus."""
import pathlib
import tomllib

import pandas as pd


def find_card(start: pathlib.Path) -> pathlib.Path:
    for d in (start, start.parent, start.parent.parent):
        c = d / "card.toml"
        if c.exists():
            return c
    raise FileNotFoundError(f"no card.toml found near {start}")


def load_card(path: pathlib.Path) -> dict:
    return tomllib.loads(path.read_text())


def get_panel_id(card: dict) -> str:
    panels = card.get("panels", {})
    if isinstance(panels, dict) and "panel_ids" in panels:
        return panels["panel_ids"][0]
    if isinstance(panels, list) and panels:
        return panels[0]["panel_id"]
    raise KeyError(f"cannot determine panel_id from card; panels={panels!r}")


def get_n_draws_min(card: dict, default: int = 200) -> int:
    scoring = card.get("scoring", {})
    if not isinstance(scoring, dict):
        return default
    params = scoring.get("params", {})
    if isinstance(params, dict) and "n_draws_min" in params:
        return int(params["n_draws_min"])
    if "n_draws_min" in scoring:
        return int(scoring["n_draws_min"])
    return default


def load_panel(panels_dir: pathlib.Path, panel_id: str) -> pd.DataFrame:
    candidates = [
        panels_dir / f"{panel_id}.parquet",
        panels_dir / "panels" / f"{panel_id}.parquet",
    ]
    for c in candidates:
        if c.exists():
            return pd.read_parquet(c)

    parquets = sorted(panels_dir.glob("*.parquet"))
    if parquets:
        return pd.read_parquet(parquets[0])

    parquets = sorted(panels_dir.rglob("*.parquet"))
    if parquets:
        return pd.read_parquet(parquets[0])

    raise FileNotFoundError(f"no parquet found under {panels_dir}")


def load_text(text_dir: pathlib.Path, asof: str) -> list[dict]:
    """Return [{'name': str, 'text': str}, ...] for .txt files in text_dir."""
    if not text_dir.exists():
        return []
    docs = []
    for f in sorted(text_dir.glob("*.txt")):
        try:
            docs.append({"name": f.name, "text": f.read_text(encoding="utf-8", errors="ignore")})
        except Exception:
            continue
    return docs