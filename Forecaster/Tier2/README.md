# Tier 2 — Gaussian Copula on ARIMA Residuals

Closes the **cross-asset joint dependence** gap left by Tier 1. Target unit:
F3 `t2-F3-bear-flattener-2022`, where short and long rates move together by
construction and independent per-asset ARIMA is a modelling error.

Tier 2 is **self-contained**: it re-implements ARIMA, panel loading, and the
copula math inside `Forecaster/Tier2/`. It shares no code with Tier 1 or any
other tier.

---

## Why

Tier 1 fits one ARIMA per asset and samples each asset independently. That
implicitly assumes **zero residual correlation** across assets. On a
single-asset card that is fine. On F3 it is wrong: a bear flattener *is* the
joint move — short rates up, long rates up less, curve flattens. Sampling the
legs independently destroys the structure the scenario is named after.

The joint path is the forecast, not the marginals.

---

## Method

1. Fit ARIMA per asset (AIC order search over `p ∈ [0,3]`, `d ∈ [0,1]`,
   `q ∈ [0,3]`), record the in-sample residuals and the h-step conditional
   mean / std from `get_forecast`.
2. Align residuals across assets by truncating to the shortest series and
   dropping rows containing NaN.
3. Rank-transform each residual column to `u_i ∈ (0,1)` via the empirical
   CDF. No Gaussian assumption on the residuals themselves.
4. Map to normal scores `z = Φ⁻¹(u)`, estimate correlation `R = corr(z)`,
   ridge-regularise toward `I` if `R` is not positive definite.
5. Sample `z* ~ N(0, R)`, push back through `Φ` → `u*`, then invert the
   empirical CDF → joint residual draws.
6. Rescale each asset's residual draws by the ratio of the ARIMA h-step
   forecast std to the in-sample residual std, then add the ARIMA h-step
   conditional mean. This puts the copula draws into the correct target
   space per horizon.

Because marginals are rank-based, non-Gaussian assets (skew, fat tails) pass
through untouched. Only the **dependence structure** changes.

### Target types

The competition has exactly two target types and they are not interchangeable:

- **`level`** — panel stores the raw value (yield %, FX mid, index level).
  Forecast is that level at future horizons. Fit ARIMA directly on the level
  series.
- **`log_return`** — panel stores decimal daily simple returns. Forecast is
  the **cumulative log return** over the horizon, computed as
  `sum(log(1 + r))`. Transform before fitting: `log1p(r).cumsum()`, then fit
  ARIMA on the cumulative series. Without this transform the `log_return`
  units produce draws in the wrong space and score near zero.

`fit_and_sample` branches on `card["targets"]["target_type"]` at the top.

---

## Files

```
Forecaster/
  Tier2/
    __init__.py                    # package marker (Tier2 only)
    arima.py                       # per-asset ARIMA, AIC order search
    copula.py                      # rank transform + Gaussian copula math
    loader.py                      # panels → aligned wide DataFrame
    forecaster_t2_copula.py        # CopulaForecaster: fit + predict
    run_tier2.py                   # CLI driver for local runs
    README.md                      # this file
    output/                        # forecast.parquet per unit (gitignored)

my-agent-2/
  forecaster.py                    # REPLACED — self-contained Tier 2
```

Note: there is **no `Forecaster/__init__.py`**. `Forecaster/` is a PEP 420
namespace package so Tier 1, Tier 2, and future tiers can coexist without
touching each other's import semantics. Only `Forecaster/Tier2/__init__.py`
exists.

### File roles

- `arima.py` — `ArimaAsset.fit(y)` returns an object with `cond_mean_`,
  `resid_`, `fitted_`, `order_`, and a `.forecast(steps)` method returning
  `(means, std_errors)`. Self-contained; only needs statsmodels and numpy.
- `copula.py` — `to_pseudo_obs`, `from_pseudo_obs`, `fit_gaussian_copula`,
  `sample_copula`. Pure numpy/scipy; no project imports.
- `loader.py` — `load_wide(panels_dir, asof)` globs parquet/csv under the
  panels dir, handles long (`date, asset, value`) and wide layouts,
  truncates to `asof`, forward-fills. Self-contained; only needs pandas.
- `forecaster_t2_copula.py` — `CopulaForecaster.fit(wide)` then
  `.predict(n_draws)` returns a `DataFrame` of draws. Composes `arima.py`
  and `copula.py`.
- `run_tier2.py` — argparse over `--panels --text --asof --out --n-draws
  --seed`; writes `forecast.parquet`; prints shape / mean / std.

---

## Imports

Only two files import project code:

- `run_tier2.py`:
  ```python
  from Forecaster.Tier2.loader import load_wide
  from Forecaster.Tier2.forecaster_t2_copula import CopulaForecaster
  ```
- `forecaster_t2_copula.py`:
  ```python
  from Forecaster.Tier2.arima import ArimaAsset
  from Forecaster.Tier2.copula import fit_gaussian_copula, sample_copula
  ```

`arima.py`, `copula.py`, `loader.py`, and `__init__.py` import nothing from
the project. `my-agent-2/forecaster.py` is intentionally self-contained
because the Docker build context cannot see `Forecaster/`.

Imports resolve because scripts are run from the repo root, which puts the
cwd on `sys.path` and makes `Forecaster` importable as a namespace package.
If you run from anywhere else, prefix with `PYTHONPATH=$(pwd)`.

---

## Run

From the repo root:

```bash
cd /path/to/track2-starter-kit
REPO=$(pwd)

mkdir -p Forecaster/Tier2/output

# Read the as-of date from the card
grep -i asof units/t2-F3-bear-flattener-2022/card.toml

python3 Forecaster/Tier2/run_tier2.py \
  --panels units/t2-F3-bear-flattener-2022 \
  --text   units/t2-F3-bear-flattener-2022/text \
  --asof   <ASOF_FROM_CARD> \
  --out    Forecaster/Tier2/output/forecast.parquet

qfbench2-smoke units/t2-F3-bear-flattener-2022/ \
  Forecaster/Tier2/output/ --track forecasting
```

---

## Sanity checks

Run all of these before accepting a Tier 2 result.

1. **Shape & admissibility** — `(500, K)`, `admissible=True` on F3.
2. **Marginal preservation** — per-asset mean and std of Tier 2 draws sit
   within Monte Carlo noise (`~1/√500 ≈ 4.5%`) of a plain independent ARIMA
   draw. Bigger drift means `to_pseudo_obs` and `from_pseudo_obs` are out of
   sync.
3. **Dependence restored** — off-diagonal `np.corrcoef` on Tier 2 residual
   draws should match the correlation of aligned in-sample ARIMA residuals
   within noise. This is the entire point of Tier 2; if it fails, the copula
   is a no-op.
4. **Rank vs linear** — Gaussian copula preserves **rank** dependence exactly
   and linear dependence only approximately. Check Spearman as well as
   Pearson if residuals look non-Gaussian.
5. **No regression on F1** — run on `t2-F1-ai-mom-2024`. With weak dependence
   the copula should be near-identity and F1 should not get worse.

A quick standalone check (no benchmark needed):

```python
import numpy as np
from pathlib import Path
from Forecaster.Tier2.loader import load_wide
from Forecaster.Tier2.forecaster_t2_copula import CopulaForecaster

wide = load_wide(Path("units/t2-F3-bear-flattener-2022"), "<ASOF>")
fc = CopulaForecaster(n_draws=500, seed=42).fit(wide)
draws = fc.predict().to_numpy()

print("shape:", draws.shape)
print("cross-asset corr:\n", np.corrcoef(draws, rowvar=False))
print("per-asset mean:", draws.mean(axis=0))
print("per-asset std :", draws.std(axis=0))
```

The off-diagonal correlations must be non-zero and roughly match the
correlation of in-sample ARIMA residuals.

---

## Numerical notes

- **Ridge regularisation** — `fit_gaussian_copula` nudges `R` toward `I`
  along its smallest eigenvalue if not PD (happens with near-collinear
  assets). Default `ε = 1e-6`; raise if sampling emits NaNs.
- **Ties** — `to_pseudo_obs` uses stable argsort, so tied residuals get
  deterministic ranks. Fine for continuous financial series.
- **Seed** — `run_tier2.py` defaults to `seed=42`. Keep it fixed across
  tiers so you compare models, not Monte Carlo noise.
- **Order search cost** — `p ∈ [0,3]`, `d ∈ [0,1]`, `q ∈ [0,3]` is 15 fits
  per asset per unit. Fast on 4 assets × 4 units; shrink the grid for speed.
- **Horizons are sampled independently.** The copula captures cross-asset
  dependence at each horizon but not cross-horizon path coherence. That
  matches how the scoring is evaluated (marginal CRPS and joint variogram
  per horizon). If path coherence is ever needed, sample one joint residual
  per path and scale across horizons.
- **Fallback preserved.** If an asset has `<20` observations, is missing from
  the panel, or ARIMA fails, `fit_and_sample` falls back to the old
  normal-sampling behaviour for that asset only. One bad unit cannot take
  down the whole run.

---

## Dependencies

- `numpy`, `scipy` — hard requirements for `copula.py`.
- `statsmodels` — hard requirement for `arima.py`.
- `pandas` — parquet output in the driver.
- `copulae` is installed but **not used**. The Gaussian copula is a few lines
  of `scipy.stats.norm`; keeping it inline avoids pinning that package's API.

---

## What Tier 2 deliberately does not do

- **No fat tails beyond empirical marginals.** GARCH + skewed-t is Tier 3,
  target F4 `t2-F4-aud-gfc-2008`.
- **No regime awareness.** Regime-conditional ARIMA is Tier 4, target F2
  `t2-F2-abenomics-2012`.
- **No text.** `llm.py` / `combiner.py` layer on top of any tier's draws.
- **No shared code with Tier 1.** Intentional — Tier 2 is standalone so it
  drops into the Docker image without Tier 1's dependencies or assumptions.
