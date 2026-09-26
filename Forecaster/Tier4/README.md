# Tier 4 — Regime-Conditional ARIMA

Closes the **regime-shift** gap left by Tier 3. Target unit: F2
`t2-F2-abenomics-2012`, where Japanese policy changed in 2012 (QE, negative
rates, fiscal stimulus) and the pre-2012 and post-2012 behaviour of the
target series are effectively two different worlds.

Tier 4 is **self-contained**: it re-implements ARIMA, GARCH, the Gaussian
copula, and its own regime detector inside `Forecaster/Tier4/`. It shares no
code with Tier 1, Tier 2, or Tier 3.

---

## Why Tier 4 exists

Tiers 1–3 fit **one** set of ARIMA + GARCH parameters over the whole history.
On a card with a regime shift, that is wrong in two directions:

**1. The parameters are an average of two worlds.**
If the pre-2012 series had different dynamics from the post-2012 series, the
single ARIMA fit sees both and reports the compromise. Neither regime's
behaviour is captured correctly. The conditional mean is mis-centred for
whichever regime we are actually in at the forecast date, and the
conditional variance is inflated by the regime difference.

**2. The uncertainty is understated.**
Even if the average were acceptable, the true forecast distribution should
include the uncertainty about **which regime we are in**. A single-regime
model collapses that into one wide Gaussian, which is the wrong shape — it
is symmetric and unimodal, whereas a genuine regime mixture is a blend of
two different centres and two different scales.

Tier 4 fixes both by:

- Detecting regimes from the data (Markov switching on a vol proxy, with a
  threshold fallback).
- Fitting ARIMA + GARCH + copula **per regime**.
- Forecasting as a **mixture**: sample per regime weighted by the forecast
  probability of each regime at each horizon.

---

## What each component does

### Regime detector — which world are we in?

Tier 4 fits a **2-regime Markov switching model with switching variance**
(`statsmodels.tsa.regime_switching.MarkovRegression`) on a common signal —
the cross-asset mean absolute return, used as a portfolio vol proxy. The
model estimates:

- `assignments_` — a regime label per time step (0 or 1)
- `transition_` — the 2×2 row-stochastic transition matrix `P`, where
  `P[i, j]` is the probability of moving from regime `i` to regime `j`
- `last_prob_` — the **filtered** probability of each regime at the final
  observed time step, which is the correct starting point for forecasting
- `counts_` — observations per regime, used to gate whether a regime has
  enough data to fit its own models

For forecasting, `predict_probs(steps)` iterates the transition matrix
forward: `p_{t+1} = p_t @ P`. This gives a `(steps, n_regimes)` matrix of
forecast regime probabilities. If the model expects to stay in the current
regime, the probabilities stay peaked. If it expects a switch, they move.

**Fallback.** If MarkovRegression fails to converge (short sample, degenerate
fit, or fewer than `min_obs * n_regimes` observations), Tier 4 falls back
to a median-threshold split on the same signal with a smoothed transition
matrix (`0.9 * I + 0.1 * uniform`). The forecast is still a mixture; only
the regime assignment method changes.

### Per-regime ARIMA — the conditional mean per world

For each regime `r` with enough observations, Tier 4 re-fits ARIMA on the
subset of the series where `assign == r`. Two parameters matter:

- **Mean path per regime** — `means_r_[r]` is a `(K, H)` array giving the
  ARIMA conditional mean for each asset at each horizon, **fitted only on
  regime-r data**.
- **Residuals per regime** — the input to per-regime GARCH.

If a regime has fewer than `min_regime_obs` observations, Tier 4 reuses the
global (all-data) ARIMA for that regime. This keeps the mixture well-defined
even when one regime is rare.

### Per-regime GARCH — volatility per world

Same GARCH(1,1) + skewed-t as Tier 3, but fit on the **regime-restricted
ARIMA residuals**. The result is a per-regime `(K, H)` vol matrix
`vols_r_[r]`. In practice the high-vol regime gets larger vols and the
low-vol regime gets smaller vols, which is exactly the information a
single-regime model throws away.

### Per-regime copula — joint dependence per world

Same Gaussian copula on standardized residuals as Tier 3, but fit
**regime by regime**. The dependence structure can differ between calm and
crisis periods — this is well documented in the literature and is one
reason a single copula undershoots joint tail risk. Tier 4 lets each
regime have its own correlation matrix `R_r_[r]`.

### Mixture sampling — blending the worlds

Sampling is the new step. For each horizon `h`:

1. Get the regime forecast probabilities `p_h` from `predict_probs(h)`.
2. Draw a regime label per path: `r_choice ~ Categorical(p_h)`.
3. For each regime `r`, take the subset of paths assigned to it and sample
   from **that regime's** ARIMA mean, GARCH vol, and copula.
4. Write into the corresponding rows of the output array.

The result is a **mixture distribution**: a blend of the per-regime
distributions weighted by the forecast probability of each regime. Its
spread is naturally larger than any single-regime spread because it
includes both within-regime uncertainty and between-regime uncertainty.

---

## Full pipeline, step by step

For each unit:

**Fit stage**

1. Load panel, filter to `asof`, pivot to wide.
2. Compute the common vol signal: cross-asset mean `|Δ|`.
3. Fit `RegimeDetector` on the signal → assignments, transition matrix,
   last filtered probabilities, counts.
4. For each asset, fit a **global** ARIMA as a fallback (used if a regime
   is too sparse or a per-regime fit fails).
5. For each regime `r`:
   a. Mask the wide panel to rows where `assign == r`.
   b. For each asset, fit ARIMA on the regime-restricted series.
   c. Fit GARCH(1,1) skewed-t on the ARIMA residuals → per-regime vol.
   d. Align standardized residuals across assets, fit copula → `R_r_[r]`.
   e. Store `means_r_[r]` and `vols_r_[r]` as `(K, H)` arrays.

**Sample stage**

For each horizon `h`:

1. Compute `p_h = predict_probs(h)[h]` and normalise.
2. Draw `r_choice ~ Categorical(p_h)` for `n_draws` paths.
3. For each regime `r`:
   - Take paths where `r_choice == r`.
   - If `R_r_[r]` exists, sample correlated uniforms, invert through each
     asset's skewed-t PPF, scale by `vols_r_[r][:, h]`, add
     `means_r_[r][:, h]`.
   - Otherwise sample independent standard normals, scale, and shift.
4. Write into `draws[idx, :, h]`.

Output shape: `(n_draws, n_assets, n_horizons)` — identical to Tiers 1–3.

---

## Files

```
Forecaster/
  Tier4/
    __init__.py                    # package marker
    arima.py                       # self-contained ArimaAsset
    garch.py                       # self-contained GarchAsset (skewed-t)
    copula.py                      # rank transform + Gaussian copula
    regime.py                      # NEW: RegimeDetector
    loader.py                      # self-contained panel loader
    forecaster_t4_regime.py        # RegimeCopulaForecaster: fit / predict
    run_tier4.py                   # CLI driver
    README.md                      # this file
    output/                        # forecast.parquet (gitignored)

my-agent-2/
  forecaster.py                    # REPLACED — self-contained Tier 4
```

No `Forecaster/__init__.py`. `Forecaster/` is a PEP 420 namespace package,
so Tier 1–4 coexist without touching each other's import semantics.

### File roles

- **`arima.py`** — `ArimaAsset.fit(y)` scans `(p, d, q)` by AIC, exposes
  `resid_` and `.forecast(steps)`. Self-contained; statsmodels + numpy.

- **`garch.py`** — `GarchAsset`. Wraps `arch`. Exposes `sigma_`, `z_`,
  `eta_`, `lambda_`, `forecast_cum_vol(horizons)`, `skewt_ppf(u)`.

- **`copula.py`** — `to_pseudo_obs`, `fit_copula_R`, `sample_copula_uniform`.
  Pure numpy/scipy.

- **`regime.py`** — `RegimeDetector`. Fits a 2-regime Markov switching model
  with switching variance, or a threshold fallback. Exposes `assignments_`,
  `transition_`, `last_prob_`, `counts_`, and `predict_probs(steps)`.

- **`loader.py`** — `load_wide(panels_dir, asof)`. Same behaviour as Tiers
  2 and 3; handles long and wide layouts, truncates to `asof`.

- **`forecaster_t4_regime.py`** — `RegimeCopulaForecaster.fit(wide, horizons)`
  then `.predict(n_draws)` returns `(n_draws, K, H)`. `.predict_df()` returns
  a flat DataFrame at the first horizon for local inspection.

- **`run_tier4.py`** — CLI over `--panels --text --asof --out --horizons
  --n-draws --seed --n-regimes`.

---

## Imports

Only two files import project code:

- `run_tier4.py`:
  ```python
  from Forecaster.Tier4.loader import load_wide
  from Forecaster.Tier4.forecaster_t4_regime import RegimeCopulaForecaster
  ```
- `forecaster_t4_regime.py`:
  ```python
  from Forecaster.Tier4.arima import ArimaAsset
  from Forecaster.Tier4.garch import GarchAsset
  from Forecaster.Tier4.copula import fit_copula_R, sample_copula_uniform
  from Forecaster.Tier4.regime import RegimeDetector
  ```

`arima.py`, `garch.py`, `copula.py`, `regime.py`, `loader.py`, and
`__init__.py` import nothing from the project. `my-agent-2/forecaster.py`
is intentionally self-contained because the Docker build context cannot see
`Forecaster/`.

Imports resolve because scripts are run from the repo root, which puts the
cwd on `sys.path` and makes `Forecaster` importable as a namespace package.
If you run from anywhere else, prefix with `PYTHONPATH=$(pwd)`.

---

## Target types

Same two as Tiers 2 and 3, and the same branch:

- **`level`** — panel stores the raw value. Fit ARIMA directly on the level
  series. Regime detection is unaffected; it operates on the same `|Δ|`
  signal.
- **`log_return`** — panel stores decimal daily simple returns. Forecast
  target is the **cumulative log return** `sum(log(1 + r))`. Transform
  before fitting: `log1p(r).cumsum()`. Regime detection still uses the
  raw `|Δ|` of the panel because that is already the correct return-space
  signal.

`fit_and_sample` branches on `card["targets"]["target_type"]` at the top.

---

## Run

From the repo root:

```bash
cd /path/to/track2-starter-kit
REPO=$(pwd)

mkdir -p Forecaster/Tier4/output

# Read as-of and horizons from the F2 card
grep -iE 'asof|horizon' units/t2-F2-abenomics-2012/card.toml

python3 Forecaster/Tier4/run_tier4.py \
  --panels units/t2-F2-abenomics-2012 \
  --asof   <ASOF_FROM_CARD> \
  --horizons 1,5,22 \
  --out    Forecaster/Tier4/output/forecast.parquet

qfbench2-smoke units/t2-F2-abenomics-2012/ \
  Forecaster/Tier4/output/ --track forecasting
```

---

## Sanity checks

Run all six. Each guards a specific failure mode unique to Tier 4.

**1. Shape & admissibility.**
Output is `(500, K, H)` and `qfbench2-smoke` returns `admissible=True` on
F2. Same shape contract as Tiers 1–3.

**2. Regime detection is non-degenerate.**
The driver prints `counts=[n0, n1]`. Both should be at least
`min_regime_obs` (default 40). If one regime has 0 or 1 observations, the
detector fell back to `threshold` and the regime split is trivial — inspect
the signal or raise `--n-regimes 1` to disable the mixture entirely.

**3. Per-regime means actually differ.**
```python
for r in range(fc.n_regimes):
    print(r, fc.means_r_[r][:, 0])   # h=1 means per asset in regime r
```
The arrays should differ substantially. If they are identical, the
per-regime ARIMA fits failed and fell back to the global fit — check the
regime counts from check #2.

**4. Mixture spread exceeds single-regime spread.**
```python
import numpy as np
std_t4 = draws_t4[:, :, -1].std(axis=0)
std_t3 = draws_t3[:, :, -1].std(axis=0)
print(std_t4 - std_t3)   # should be positive on F2
```
If not, the regime probabilities are collapsing to a single regime at every
horizon. Check the transition matrix diagonals — a near-identity `P` with a
peaked `last_prob_` means the model expects no switching, in which case the
mixture degenerates to a single regime and Tier 4 loses its advantage.

**5. Regime assignment matches known events.**
On F2, plot the signal with the regime assignments overlaid. The 2012 policy
shift should be visible in the assignment series or in the fitted regime
probabilities. If assignments look like white noise, the Markov switching
fit failed and fell back to threshold; check `det._method`.

**6. No regression on F1 / F3 / F4.**
Tier 4 is dispatched to F2 only, but for local testing run it on F1, F3,
F4 to confirm it does not blow up. On F1 (single asset, well-behaved), Tier
4 should be roughly equal to Tier 1. If F1 degrades sharply, the regime
detector is splitting a stationary series into spurious regimes — consider
disabling regime detection when only one asset is present.

---

## Numerical notes

**Markov switching convergence.**
`MarkovRegression` occasionally fails on short, trending, or low-signal
series. Tier 4 catches the exception and falls back to a median threshold
on the same signal. The fallback uses a smoothed transition matrix
(`0.9 * I + 0.1 * uniform`) rather than a pure identity so the mixture is
non-degenerate even when the current regime is confident.

**Regime count.**
Default is 2. A 3-regime model is possible by passing `--n-regimes 3`, but
each additional regime needs `min_regime_obs` observations to be fit
properly, and on a short card one regime will collapse to the fallback. Two
is the safe default for F2-scale data.

**Transition matrix shape.**
`statsmodels` returns `regime_transition` as a `(k, k, 1)` array. Tier 4
squeezes the trailing axis and re-normalises rows so they sum to one, which
is robust to small numerical drift in the fit.

**Per-regime ARIMA failure.**
If a per-regime ARIMA fit fails (too few observations after masking, or
optimiser non-convergence), that regime's `means_r_[r]` and `vols_r_[r]`
default to the **global** all-data fit. The mixture still works; only the
within-regime precision is degraded.

**Per-regime copula failure.**
If a regime has fewer than 30 aligned standardized residuals, or `R_r_[r]`
is singular after ridge regularisation, the sampling for that regime falls
back to independent normals. The mixture is still correct in mean and
variance; only cross-asset dependence within that regime is dropped.

**Alignment of regime assignments to the panel index.**
The signal is built from `wide.diff().abs()`, which loses the first row.
Tier 4 aligns assignments back to the panel by truncating to the shorter
length (`L = min(len(assign), len(wide) - 1)`), then masking the wide panel
with the aligned assignment. This avoids an off-by-one that would
contaminate the regime masks.

**Mixture sampling order.**
The regime labels are drawn **once per horizon**, not once for the whole
path. This matches how the scoring is evaluated: marginal CRPS and joint
variogram are per-horizon, so per-horizon regime draws are the correct
choice. It does mean paths are not coherent across horizons in their
regime identity — same trade-off as Tiers 1–3 with respect to cross-horizon
dependence.

**Seeding.**
`run_tier4.py` and the Docker `fit_and_sample` both default to `seed=42`.
Keep it fixed across tiers so you compare models, not Monte Carlo noise.

---

## Dependencies

- `numpy`, `scipy` — copula math and normal quantiles.
- `statsmodels` — ARIMA and Markov switching.
- `arch` — GARCH(1,1) and `SkewStudent`.
- `pandas` — panel handling and parquet output.

---

## What Tier 4 deliberately does not do

- **No Student-t copula.** The within-regime dependence is Gaussian. A
  Student-t copula would add joint tail dependence, at the cost of one more
  parameter per regime. Consider as a later upgrade if joint tail penalty
  on F2 is the dominant error.
- **No EVT tails.** Skewed-t is the tail model. Peaks-over-threshold GPD
  would be a further upgrade, again optional.
- **No text conditioning.** `llm.py` / `combiner.py` layer on top of any
  tier's draws, applied after `forecaster.py` returns.
- **No path coherence across horizons.** Each horizon is drawn
  independently, including its regime label. Consistent with Tiers 1–3.
- **No shared code with Tiers 1–3.** Intentional — Tier 4 is standalone so
  it drops into the Docker image without pulling in other tiers'
  dependencies or assumptions.

---

## How Tier 4 differs from Tiers 1–3 in one table

| Aspect | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---|---|---|---|
| Conditional mean | ARIMA | ARIMA | ARIMA | ARIMA per regime |
| Volatility | constant | constant | GARCH(1,1) | GARCH(1,1) per regime |
| Tail shape | Gaussian | empirical | skewed-t | skewed-t per regime |
| Cross-asset dependence | none | Gaussian copula | Gaussian copula on z | Gaussian copula per regime |
| Regime awareness | no | no | no | **yes, mixture** |
| Target unit | F1 | F3 | F4 | F2 |
