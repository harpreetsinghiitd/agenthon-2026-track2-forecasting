# Tier 3 — GARCH + Skewed-t + Gaussian Copula

Closes the **fat tails and volatility clustering** gap left by Tier 2. Target
unit: F4 `t2-F4-aud-gfc-2008`, where residuals have crash-scale tails and
vol that spikes in regimes that the training window barely samples.

Tier 3 is **self-contained**: it re-implements ARIMA, GARCH, and copula math
inside `Forecaster/Tier3/`. It shares no code with Tier 1 or Tier 2.

---

## Why Tier 3 exists

Tier 2 does two things well: ARIMA for the conditional mean, and a Gaussian
copula on empirical residuals for cross-asset dependence. But it fails on a
tail-penalised card like F4 for two reasons:

**1. The empirical sample cannot contain the tail event.**
Tier 2 rank-transforms residuals to pseudo-observations and inverts the
**empirical** CDF. That means the worst draw Tier 2 can ever produce is the
worst residual in the training window. If the training window predates or
only partially overlaps the GFC, the maximum draw is nowhere near the
realized crash. On a card scored with a 20% tail penalty, this is fatal.

**2. Volatility is treated as constant.**
Tier 2 uses ARIMA's `se_mean` at each horizon, which comes from the
unconditional residual variance. On a card where vol triples in a crisis,
that produces draws with the wrong scale in exactly the period that matters
most. GARCH is the standard fix: the h-step variance forecast is
conditional on today's vol, which is estimated from the recent regime.

Tier 3 replaces the two weak pieces — the marginal tail shape and the
marginal volatility — while keeping everything Tier 2 got right: ARIMA
conditional means, and the Gaussian copula for cross-asset dependence.

---

## What each component does

### ARIMA — the conditional mean

Same as Tier 2. Per-asset ARIMA(p,d,q) with AIC order search over
`p ∈ [0,3]`, `d ∈ [0,1]`, `q ∈ [0,3]`. Produces:

- `resid_` — in-sample residuals, the input to GARCH
- `forecast(h)` → `(means, ses)`, the h-step conditional mean and standard
  error from statsmodels

The mean path is unchanged from Tier 2. Tier 3 only replaces what happens
*around* that mean.

### GARCH(1,1) — the volatility model

Fit on the ARIMA residuals, not on the raw series. GARCH models the
conditional variance of a mean-zero series, so the ARIMA fit removes the
predictable mean first.

The model is:

```
σ²_t = ω + α · e²_{t-1} + β · σ²_{t-1}
```

- `e_t = y_t − μ_t` is the ARIMA residual
- `σ_t` is the conditional standard deviation at time `t`
- `ω, α, β` are fitted by maximum likelihood

Two things come out of this fit:

- `σ_t` — the **conditional volatility series**. It spikes when the model
  sees a large shock and then decays back toward the long-run average. This
  is the "volatility clustering" behaviour you see in every financial series.
- `z_t = e_t / σ_t` — the **standardized residuals**. By construction these
  are approximately iid with unit variance (and, if GARCH is correct, no
  remaining volatility structure). These are the inputs to the copula,
  because they are the part of the residual that is neither predictable
  mean nor predictable variance.

For forecasting, `res_.forecast(horizon=h)` gives the per-step variance
`Var(e_{T+i} | F_T)` for `i = 1..h`. Cumulative variance over the horizon is
`sum_i Var(e_{T+i})`, and the h-step cumulative vol is its square root. This
is the right scaling for a random-walk level forecast and for a cumulative
log-return target alike.

### Skewed-t — the tail shape

GARCH tells us *when* vol is high. Skewed-t tells us *what the standardized
residuals look like*, so we can extrapolate beyond the sample.

The skewed-t (Hansen 1994) has two parameters:

- `η` (eta) — degrees of freedom. Lower = fatter tails. `η → ∞` recovers
  the normal. Financial residuals typically land at `η ∈ [3, 8]`.
- `λ` (lambda) — skewness in `(-1, 1)`. Negative = left tail fatter than
  right tail. AUD/USD residuals in a crisis card typically give `λ < 0`,
  which is exactly the crash asymmetry a tail penalty rewards.

`arch` fits both by maximum likelihood alongside the GARCH parameters. After
fitting, `SkewStudent.ppf(u, [eta, lambda])` gives us the inverse CDF — this
is what Tier 3 uses to convert a uniform draw into a standardized residual.

The key property: `ppf` **extrapolates**. If `u = 0.001` and `η = 4`, the
returned `z` is far beyond anything in the empirical sample. That is the
whole point of Tier 3.

### Gaussian copula — the dependence

Same construction as Tier 2, but applied to a different object. The copula
is fit on the **standardized residuals** `z_t`, not on the raw ARIMA
residuals. This is the correct choice because after GARCH filtering, `z_t`
is the iid component, and dependence is a property of the iid component,
not of the jointly heteroskedastic residuals.

The pipeline:

1. Rank-transform each column of `z` to pseudo-observations `u ∈ (0,1)`.
2. Map to normal scores `Φ⁻¹(u)`.
3. Estimate the correlation matrix `R = corr(Φ⁻¹(u))`.
4. Ridge-regularise toward identity if `R` is not positive definite.

`R` is the joint dependence structure. Sampling `N(0, R)` and mapping
through `Φ` gives correlated uniforms, which then invert per-asset through
the skewed-t to produce joint standardized residual draws.

---

## Full pipeline, step by step

For each unit:

**Fit stage**

1. Load panel, filter to `asof`, pivot to wide.
2. For each target asset:
   a. Transform the series to the target space (`level` unchanged,
      `log_return` via `log1p(r).cumsum()`).
   b. Fit ARIMA → residuals and h-step mean/SE.
   c. Fit GARCH(1,1) + skewed-t on the residuals → `σ_t`, `z_t`, `η`, `λ`,
      h-step cumulative vol.
   d. Record `mean[k, h]` and `vol[k, h]` for each horizon.
3. Align `z` across assets (truncate to shortest, drop NaN rows).
4. Rank-transform, normal-score, estimate `R`.

**Sample stage**

For each horizon `h`:

1. Draw `Z ~ N(0, R)` with `n_draws` rows.
2. Map to correlated uniforms `U = Φ(Z)`.
3. For each asset `k`, invert through its skewed-t: `z*_k = ppf(U_k; η_k, λ_k)`.
4. Scale by `vol[k, h]` and add `mean[k, h]`.
5. Write into `draws[:, k, h]`.

Output: `(n_draws, n_assets, n_horizons)` — the same shape the Docker
agent expects.

---

## Why fit the copula on standardized residuals

This is the copula-GARCH construction and it is not optional. Consider two
assets whose raw residuals are correlated only because both series have
high-vol regimes at the same time. If you fit the copula on raw residuals,
you measure that co-movement as dependence. But it is **not** dependence
between the iid components — it is dependence between the volatility
processes, which GARCH has already modelled. The correct split is:

- GARCH (per asset) captures the vol dynamics.
- Copula (across assets) captures the iid co-movement.

Standardizing first is what makes this split work. It is also why the
per-asset `σ_t` series can differ wildly across assets without destabilizing
the copula — you are correlating `z = e/σ`, which is scale-free.

A practical consequence: if you skip standardization, the fitted `R` will
be biased upward and the joint variogram will overshoot. Sanity check #5
below is designed to catch exactly that.

---

## Files

```
Forecaster/
  Tier3/
    __init__.py                  # package marker
    arima.py                     # copy of Tier 2's ArimaAsset
    copula.py                    # rank transform + correlation + sampling
    garch.py                     # GarchAsset: fit, forecast_cum_vol, skewt_ppf
    loader.py                    # copy of Tier 2's load_wide
    forecaster_t3_garch.py       # GarchCopulaForecaster: fit / predict
    run_tier3.py                 # CLI driver for local runs
    README.md                    # this file
    output/                      # forecast.parquet (gitignored)

my-agent-2/
  forecaster.py                  # REPLACED — self-contained Tier 3
```

No `Forecaster/__init__.py`. `Forecaster/` is a PEP 420 namespace package,
so Tier 1, Tier 2, Tier 3 coexist without touching each other.

### File roles

- **`arima.py`** — `ArimaAsset.fit(y)` scans `(p,d,q)` by AIC, exposes
  `resid_` and a `forecast(steps)` method returning `(means, ses)`.
  Self-contained; needs only statsmodels and numpy.

- **`copula.py`** — `to_pseudo_obs`, `fit_copula_R`, `sample_copula_uniform`.
  Pure numpy/scipy. Difference vs Tier 2: no empirical inverse — the
  inversion step lives in `garch.py` because it goes through the skewed-t
  PPF, not the empirical CDF.

- **`garch.py`** — `GarchAsset`. Wraps `arch`. Exposes `sigma_` (conditional
  vol), `z_` (standardized residuals), `eta_`, `lambda_` (skewed-t params),
  `forecast_cum_vol(horizons)` (h-step cumulative vol), `skewt_ppf(u)`
  (inverse skewed-t CDF).

- **`loader.py`** — `load_wide(panels_dir, asof)`. Globs parquet/csv,
  handles long (`date, asset, value`) and wide layouts, truncates to
  `asof`, forward-fills. Self-contained; only pandas.

- **`forecaster_t3_garch.py`** — `GarchCopulaForecaster.fit(wide, horizons)`
  then `.predict(n_draws)` returns `(n_draws, K, H)`. `.predict_df()` returns
  a flat `DataFrame` at the first horizon for local inspection.

- **`run_tier3.py`** — CLI over `--panels --text --asof --out --horizons
  --n-draws --seed`.

---

## Imports

Only two files import project code:

- `run_tier3.py`:
  ```python
  from Forecaster.Tier3.loader import load_wide
  from Forecaster.Tier3.forecaster_t3_garch import GarchCopulaForecaster
  ```
- `forecaster_t3_garch.py`:
  ```python
  from Forecaster.Tier3.arima import ArimaAsset
  from Forecaster.Tier3.garch import GarchAsset
  from Forecaster.Tier3.copula import fit_copula_R
  ```

`arima.py`, `copula.py`, `garch.py`, `loader.py`, and `__init__.py` import
nothing from the project. `my-agent-2/forecaster.py` is intentionally
self-contained because the Docker build context cannot see `Forecaster/`.

Imports resolve because scripts are run from the repo root, which puts the
cwd on `sys.path` and makes `Forecaster` importable as a namespace package.
If you run from anywhere else, prefix with `PYTHONPATH=$(pwd)`.

---

## Target types

Same two as Tier 2, and the same branch:

- **`level`** — panel stores the raw value. Fit ARIMA directly on the level
  series. GARCH is fit on the ARIMA residuals of that level series.
- **`log_return`** — panel stores decimal daily simple returns. Forecast
  target is the **cumulative log return** `sum(log(1+r))`. Transform before
  fitting: `log1p(r).cumsum()`, then fit ARIMA on the cumulative series.

`fit_and_sample` branches on `card["targets"]["target_type"]` at the top.

---

## Run

From the repo root:

```bash
cd /path/to/track2-starter-kit
REPO=$(pwd)

mkdir -p Forecaster/Tier3/output

# Read horizons and as-of from the card
grep -iE 'asof|horizon' units/t2-F4-aud-gfc-2008/card.toml

python3 Forecaster/Tier3/run_tier3.py \
  --panels units/t2-F4-aud-gfc-2008 \
  --asof   <ASOF_FROM_CARD> \
  --horizons 1,5,22 \
  --out    Forecaster/Tier3/output/forecast.parquet

qfbench2-smoke units/t2-F4-aud-gfc-2008/ \
  Forecaster/Tier3/output/ --track forecasting
```

---

## Sanity checks

Run all six. Each one guards a specific failure mode.

**1. Shape & admissibility**
Output is `(500, K, H)` and `qfbench2-smoke` returns `admissible=True` on
F4. Catches schema errors early.

**2. Kurtosis beats Tier 2**
Draw kurtosis at `h = 1` must exceed Tier 2's on F4.
```python
from scipy.stats import kurtosis
k3 = kurtosis(draws_t3[:, :, 0], axis=0)
k2 = kurtosis(draws_t2[:, :, 0], axis=0)
print(k3 - k2)   # must be positive
```
If not, skewed-t is not biting. Check `eta_` — it should be `< 10` on F4.
`eta_ ≈ 30` means the fit collapsed to normal.

**3. Skew sign matches intuition**
For AUD/USD in a crisis card, `lambda_` should be negative.
```python
for a in fc.assets_:
    print(a, fc.garch_[a].eta_, fc.garch_[a].lambda_)
```
If `lambda_ = 0` for every asset, the skew parameter is unidentified — the
fit converged to a degenerate point. Raise `maxiter` and re-check.

**4. Vol clustering survives**
Conditional vol `sigma_` should spike around the crisis window. Plot if
unsure:
```python
import matplotlib.pyplot as plt
plt.plot(fc.garch_[fc.assets_[0]].sigma_)
```
Flat `sigma_` means GARCH collapsed to constant variance. Check that the
ARIMA residuals were actually the input, not the raw series.

**5. Dependence is preserved on standardized residuals**
```python
import numpy as np
Z_hat = np.column_stack([fc.garch_[a].z_ for a in fc._garch_assets])
print(np.corrcoef(Z_hat, rowvar=False))
```
Off-diagonals should match the copula matrix `fc.R_` within noise. If
`R_` is much larger, the copula is picking up residual vol dependence that
GARCH failed to filter. Check `sigma_` convergence.

**6. No regression on F1 / F2 / F3**
Run Tier 3 on all four units. On F1 (single asset, well-behaved) it should
be roughly equal to Tier 2. If F1 gets worse, the skewed-t is over-fitting
a thin tail into a series that doesn't have one. Guard: raise `maxiter` and
verify `eta_` is not dropping below 3.

---

## Numerical notes

**GARCH non-convergence.**
`arch` occasionally fails on short or well-behaved residual series. Tier 3
catches this per asset: if GARCH fails, that asset falls back to ARIMA
`se_mean` for vol and standard normal for shape. The copula still uses
whatever GARCH residuals exist. One bad asset cannot take down the unit.

**Cumulative h-step vol.**
`sqrt(sum_{i=1}^{h} σ²_i)` where `σ²_i` is the GARCH variance forecast at
step `i`. Correct for random-walk levels and for cumulative log returns.
Do not use `σ_T · √h` — that ignores mean reversion in the vol forecast and
understates the h-step vol when today's vol is high.

**Ridge regularisation of `R`.**
`fit_copula_R` nudges `R` toward `I` along its smallest eigenvalue if `R` is
not positive definite. Default `ε = 1e-6`. Near-collinear assets (a curve
trio, say) can produce singular `R`; the ridge keeps sampling stable.

**Seeding.**
`run_tier3.py` defaults to `seed=42`. Keep it fixed across tiers so you
compare models, not Monte Carlo noise. The Docker agent also seeds
`np.random.default_rng(seed=42)` inside `fit_and_sample`.

**Horizons are sampled independently.**
The copula captures cross-asset dependence at each horizon but not
cross-horizon path coherence. That matches how the scoring is evaluated
(marginal CRPS and joint variogram per horizon). If path coherence is ever
needed, sample one joint `z` per path and scale across horizons by the
GARCH vol forecast at each step.

**Order of `arch` parameters.**
`SkewStudent.ppf(u, [eta, lambda])` — eta first, lambda second. Swapping
them silently produces draws in the wrong tail regime with no error.
Confirm once:
```python
from arch.univariate.distribution import SkewStudent
import numpy as np
d = SkewStudent()
print(d.ppf(np.array([0.05, 0.5, 0.95]), [5.0, -0.2]))
```
The 5th percentile should be a large negative number (~ -2), the median
near 0, the 95th a smaller positive number (~ 1.5). If the asymmetry is
flipped, the parameter order is wrong for your `arch` version.

---

## Dependencies

- `numpy`, `scipy` — copula math and normal quantiles.
- `statsmodels` — ARIMA.
- `arch` — GARCH(1,1) and `SkewStudent`. Already installed.
- `pandas` — panel handling and parquet output.

---

## What Tier 3 deliberately does not do

- **No regime switching in the mean.** ARIMA parameters are constant over
  the sample. Regime-conditional ARIMA is Tier 4, target F2
  `t2-F2-abenomics-2012`.
- **No text conditioning.** `llm.py` / `combiner.py` layer on top of any
  tier's draws, applied after `forecaster.py` returns.
- **No path coherence across horizons.** Each horizon is drawn
  independently. See numerical notes above.
- **No shared code with Tier 1 or Tier 2.** Intentional — Tier 3 is
  standalone so it drops into the Docker image without pulling in other
  tiers' dependencies or assumptions.
