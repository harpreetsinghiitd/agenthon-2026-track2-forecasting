"""CLI driver for Tier 3. Self-contained; no Tier 1 or Tier 2 imports."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from Forecaster.Tier3.loader import load_wide
from Forecaster.Tier3.forecaster_t3_garch import GarchCopulaForecaster


def main():
    ap = argparse.ArgumentParser(description="Tier 3: ARIMA + GARCH + copula")
    ap.add_argument("--panels", required=True,
                    help="directory containing panel parquet/csv")
    ap.add_argument("--text", default=None,
                    help="accepted for CLI parity with other tiers; unused")
    ap.add_argument("--asof", required=True, help="ISO date, e.g. 2008-10-15")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--horizons", default="1,5,22",
                    help="comma-separated horizons")
    ap.add_argument("--n-draws", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]

    wide = load_wide(Path(args.panels), args.asof)
    fc = GarchCopulaForecaster(n_draws=args.n_draws, seed=args.seed)
    fc.fit(wide, horizons)
    draws = fc.predict()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Flat wide output for local inspection: one column per (asset, horizon)
    arr = draws.reshape(draws.shape[0], -1)
    cols = [f"{a}_h{h}" for h in horizons for a in fc.assets_]
    pd.DataFrame(arr, columns=cols).to_parquet(out, index=False)

    print(
        f"wrote {out}  shape={draws.shape}  "
        f"mean={draws.mean():.6f}  std={draws.std():.6f}  "
        f"assets={fc.assets_}  horizons={horizons}"
    )


if __name__ == "__main__":
    main()
