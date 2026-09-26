"""CLI driver for Tier 2. Self-contained; no Tier 1 imports."""
from __future__ import annotations
import argparse
from pathlib import Path

from Forecaster.Tier2.loader import load_wide
from Forecaster.Tier2.forecaster_t2_copula import CopulaForecaster


def main():
    ap = argparse.ArgumentParser(description="Tier 2: ARIMA + Gaussian copula")
    ap.add_argument("--panels", required=True,
                    help="directory containing panel parquet/csv")
    ap.add_argument("--text", default=None,
                    help="accepted for CLI parity with other tiers; unused")
    ap.add_argument("--asof", required=True, help="ISO date, e.g. 2022-09-30")
    ap.add_argument("--out", required=True, help="output forecast.parquet path")
    ap.add_argument("--n-draws", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    wide = load_wide(Path(args.panels), args.asof)
    fc = CopulaForecaster(n_draws=args.n_draws, seed=args.seed).fit(wide)
    draws = fc.predict()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    draws.to_parquet(out, index=False)

    arr = draws.to_numpy()
    print(
        f"wrote {out}  shape={arr.shape}  "
        f"mean={arr.mean():.6f}  std={arr.std():.6f}  "
        f"assets={list(draws.columns)}"
    )


if __name__ == "__main__":
    main()
