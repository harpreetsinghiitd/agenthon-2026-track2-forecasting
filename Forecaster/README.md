# Statistical Forecasting Models for Agenthon 2026 Track 2

All models below are **fitted, non-neural statistical models**. None require organizer approval
under the Track 2 artifact policy. All take the panel (numeric) as input and produce Monte Carlo
draws of shape `[n_draws, n_assets, n_horizons]` as output.

---

## Part 1 — Tier-Wise List of Statistical Models

### Tier 1 — Classical Univariate Time-Series Models

| Model | How it works |
|---|---|
| **AR (Autoregressive)** | Predicts the current value as a linear combination of its own past values. Order `p` = how many lags. |
| **MA (Moving Average)** | Predicts the current value as a linear combination of past forecast errors. Order `q` = how many error terms. |
| **ARMA** | Combines AR and MA. Suitable for stationary series. |
| **ARIMA** | ARMA on a differenced series. The `I` (Integrated) component handles non-stationarity (trends). Order `(p, d, q)`. |
| **SARIMA** | ARIMA with seasonal AR, I, and MA terms. Order `(p, d, q)(P, D, Q, s)` where `s` is the seasonal period (e.g., 4 for quarterly). |
| **AutoARIMA** | Automated search over `(p, d, q)` combinations, selecting the best by AIC/BIC. Removes manual order selection. |
| **FARIMA** | ARIMA with fractional differencing. Captures long-memory dependence. |
| **ETS (Exponential Smoothing)** | Forecast is a weighted average of past observations, with weights decaying exponentially. Components: Error, Trend, Seasonality. |
| **Holt's Linear** | ETS with trend but no seasonality. |
| **Holt-Winters** | ETS with trend and seasonality. |
| **Theta** | Decomposes a series into "theta lines" (linear trend + short-term curvature) and combines them. Equivalent to Simple Exponential Smoothing with drift. |
| **Naive / Random Walk** | Forecast = last observed value. Baseline only. |
| **Drift** | Forecast = last value + average historical drift × horizon. Baseline only. |

### Tier 2 — Classical Multivariate Time-Series Models

| Model | How it works |
|---|---|
| **VAR (Vector Autoregression)** | Each asset's future value is a linear function of its own lags and the lags of every other asset. Captures linear cross-asset dependence natively. |
| **VARX** | VAR with exogenous regressors. Allows outside signals (e.g., text-derived features) to influence the forecast. |
| **VECM (Vector Error Correction Model)** | VAR for cointegrated series. Captures long-run equilibrium relationships between assets. |
| **Bayesian VAR (BVAR)** | VAR with prior distributions on coefficients. Regularizes when the number of assets is large. |
| **FAVAR (Factor-Augmented VAR)** | VAR on latent factors extracted from many assets, then map back to assets. |
| **Structural VAR (SVAR)** | VAR with economic identifying restrictions. Rarely needed for pure forecasting. |

### Tier 3 — Volatility Models

| Model | How it works |
|---|---|
| **ARCH(q)** | Models the conditional variance as a function of past squared residuals. |
| **GARCH(p,q)** | Extends ARCH: variance depends on past variances and past squared residuals. Captures volatility clustering. |
| **EGARCH** | GARCH with asymmetric response. Negative shocks increase volatility more than positive ones. |
| **GJR-GARCH** | Another asymmetric GARCH variant with a leverage term. |
| **TGARCH** | Threshold GARCH. Different volatility dynamics above and below a threshold. |
| **IGARCH** | Integrated GARCH. Shocks to variance are permanent. |
| **FIGARCH** | Fractionally integrated GARCH. Long-memory volatility. |
| **DCC-GARCH** | Multivariate GARCH with time-varying correlation matrix. Models joint volatility across assets. |
| **BEKK-GARCH** | Multivariate GARCH with a specific parameterization of the covariance matrix. |
| **Stochastic Volatility (SV)** | Volatility follows its own latent stochastic process. Estimated via Kalman filter or MCMC. |

### Tier 4 — Copula Models

| Model | How it works |
|---|---|
| **Gaussian Copula** | Couples marginals via a multivariate normal dependence structure. Linear correlation only. |
| **Student-t Copula** | Same but with a t-distribution. Captures joint tail dependence (assets crash together). |
| **Clayton Copula** | Strong lower-tail dependence. Good for crash scenarios. |
| **Gumbel Copula** | Strong upper-tail dependence. Good for rally scenarios. |
| **Frank Copula** | Symmetric tail dependence. |
| **Joe Copula** | Upper-tail dependence, sharper than Gumbel. |
| **Vine Copulas (C, D, R)** | Hierarchical pair-copula constructions. Flexible dependence for many assets. |
| **Elliptical Copulas** | Generalized family including Gaussian and Student-t. |

### Tier 5 — Linear and Tree-Based Regression Models

| Model | How it works |
|---|---|
| **OLS (Ordinary Least Squares)** | Fits a linear model minimizing squared error. |
| **Ridge (L2)** | OLS with an L2 penalty on coefficients. Handles multicollinearity. |
| **Lasso (L1)** | OLS with an L1 penalty. Performs feature selection. |
| **ElasticNet** | Combines L1 and L2 penalties. |
| **Quantile Regression** | Fits specific quantiles (e.g., 1st, 5th, 95th, 99th) rather than the mean. Directly targets tail behavior. |
| **Decision Tree** | Recursive binary splits on features to predict the target. |
| **Random Forest** | Ensemble of decision trees, each on a bootstrap sample. Reduces variance. |
| **Gradient Boosted Trees (GBT)** | Sequential ensemble where each tree corrects the residual errors of the previous. |
| **XGBoost / LightGBM / CatBoost** | Optimized GBT implementations. Fast, accurate, handle missing data. |
| **Quantile Regression Forest** | Random forest that predicts conditional quantiles. |

### Tier 6 — Factor and Dimensionality Reduction Models

| Model | How it works |
|---|---|
| **PCA (Principal Component Analysis)** | Finds orthogonal linear combinations of assets that explain the most variance. |
| **Statistical Factor Model** | Regress each asset's returns on PCA factors plus idiosyncratic noise. |
| **Dynamic Factor Model (DFM)** | Factors themselves follow a time-series process (e.g., VAR). Estimated via Kalman filter. |
| **Sparse PCA** | PCA with sparsity constraints. Factors load on fewer assets. |
| **Independent Component Analysis (ICA)** | Finds statistically independent components, not just uncorrelated ones. |

### Tier 7 — Non-Parametric and Bootstrap Models

| Model | How it works |
|---|---|
| **Historical Simulation** | Use actual past returns as scenarios. No model fitted. |
| **Block Bootstrap** | Resample blocks of consecutive historical days. Preserves intra-block dependence. |
| **Stationary Bootstrap** | Random block lengths (geometric). More flexible than fixed-length blocks. |
| **Moving Block Bootstrap** | Fixed-length blocks, all overlapping positions. |
| **Wild Bootstrap** | Resample residuals with random signs. Preserves volatility structure. |
| **Filtered Historical Simulation** | Combine GARCH filtering with bootstrap on standardized residuals. |

### Tier 8 — Tail-Focused and Extreme Value Models

| Model | How it works |
|---|---|
| **Extreme Value Theory (EVT)** | Fit a Generalized Pareto Distribution (GPD) to exceedances over a high threshold. Models tails separately from the body. |
| **Peaks Over Threshold (POT)** | Apply EVT to all observations above a threshold. |
| **Block Maxima** | Apply EVT to the maximum of each block (e.g., monthly max loss). |
| **Generalized Extreme Value (GEV)** | Fit the limiting distribution of block maxima. |
| **Conditional EVT** | EVT with time-varying or covariate-dependent parameters. Useful for text-conditioned tail forecasting. |

### Tier 9 — State-Space and Filtering Models

| Model | How it works |
|---|---|
| **Local Level Model** | State = level. Observed = level + noise. Recursive updates via Kalman filter. |
| **Local Linear Trend** | State = (level, slope). Allows the trend to change over time. |
| **Structural Time Series (STS)** | Explicit components: trend, seasonality, cycle, regression, irregular. |
| **Unobserved Components Model (UCM)** | Decomposes a series into interpretable components with their own dynamics. |
| **Kalman Filter / Smoother** | General recursive algorithm for estimating latent states. Underlies many of the above. |
| **Dynamic Linear Model (DLM)** | Bayesian version of state-space models. |

---

Which tier to use for what
Tier 1 (ARIMA) — for a single asset, single forecast. Start here.
Tier 2 (VAR) — for many assets that move together linearly.
Tier 3 (GARCH) — when markets are wild and calm at different times.
Tier 4 (Copula) — when you need to combine individual forecasts into a joint picture.
Tier 5 (Quantile) — when you care about the tails specifically.
Tier 6 (Factor) — when you have many assets and few observations.
Tier 7 (Bootstrap) — quick baseline, no fitting.
Tier 8 (EVT) — extreme events only.
Tier 9 (State-space) — when you think there is a hidden underlying truth.

---

## Part 2 — Card Families: What Each Tests and Best Models

### F1 — Baseline Forecasting

**What it tests:** The accuracy and calibration of a forecast's marginal distribution.
Ordinary time-series forecasting, often single-asset, single-horizon. For single-number cards,
the effective scoring weights shift to **71.4% marginal CRPS + 28.6% tail penalty** because the
joint variogram is structurally zero.

**What matters most:** A well-calibrated central tendency with realistic spread.

**Best models:**

1. **ARIMA / AutoARIMA** — the standard workhorse for univariate forecasting. Captures
   autocorrelation and trends. Start here.
2. **ETS (Holt-Winters)** — strong alternative when the series has smooth trend or seasonality.
   Often beats ARIMA on macro/quarterly series.
3. **Theta** — simple, robust, hard to beat on benchmark datasets. Good as a sanity-check baseline.
4. **Quantile Regression** — if you want direct control over the marginal distribution's tails.

### F2 — Text-Cued Regime Shift with Transfer

**What it tests:** The ability to use text to detect a regime shift, and to generalize that
detection from assets you have data for (G10 FX) to assets you do not (EM FX). The numerical
model must be flexible enough to have its dynamics altered by external signals.

**What matters most:** A flexible model that can be conditioned on regime information, and
that can transfer across assets.

**Best models:**

1. **ARIMA / ETS with regime-conditional parameters** — fit separate models per regime, or use
   a state-space model where the state is influenced by an external signal. When the LLM
   signals a regime shift, switch to the appropriate parameter set.
2. **VARX (VAR with exogenous regressors)** — allows the forecast for unseen assets to be
   influenced by the past values of seen assets and by a text-derived exogenous feature.
3. **Threshold Autoregressive (TAR) / Smooth Transition AR (STAR)** — parameters change when a
   threshold variable crosses a boundary. Adaptable to allow LLM signals as the trigger.

### F3 — Cross-Asset Reasoning

**What it tests:** Accurate modeling of the joint distribution across multiple assets.
The **Joint Variogram is the dominant term (30% weight)**. Independent per-asset sampling
will be severely penalized.

**What matters most:** Capturing the true dependence structure — correlations and tail
co-movement — between assets.

**Best models:**

1. **Copula Models (Gaussian or Student-t)** — the most direct approach. Fit univariate
   marginals (ARIMA, ETS, GARCH) to each asset, extract standardized residuals, fit a copula
   to the residuals. Sample from the copula and transform through inverse marginal CDFs.
   **Student-t copula** is particularly effective because it captures tail dependence.
2. **VAR (Vector Autoregression)** — each asset depends on lags of itself and all others.
   Naturally captures linear dependence. `statsmodels` provides robust implementation.
3. **DCC-GARCH** — combines univariate GARCH with a time-varying correlation matrix.
   Captures both volatility clustering and changing cross-asset dependence.
4. **Dynamic Factor Model** — if the number of assets is large, extract a few factors,
   model their dynamics with VAR, reconstruct asset paths.

### F4 — Tail / Shock-from-Text

**What it tests:** Extreme events driven by textual signals. The **Tail Penalty (20% weight)**
is the most important component. Your model must generate a significant number of samples in
the tails. Recommend **1,000+ draws** for reliable tail quantile estimates.

**What matters most:** Fat tails, asymmetry (crash risk), and joint tail dependence.

**Best models:**

1. **GARCH with Skewed-t Innovations** — GARCH models time-varying volatility; the skewed-t
   distribution controls the fatness and asymmetry of the tails. Simulate future volatility
   paths and draw returns from the conditional skewed-t. `arch` package supports this.
2. **Quantile Regression** — directly models the quantiles of interest (1st, 5th, 95th, 99th).
   Gives precise control over tail behavior. Fit on returns with lagged values or text
   signals as predictors.
3. **Extreme Value Theory (EVT) with Copula** — EVT models the tails separately from the body.
   Fit a Generalized Pareto Distribution to extreme losses and gains for each asset. Combine
   with a copula to ensure tail events occur jointly across assets.
4. **Filtered Historical Simulation with GARCH** — filter returns through a GARCH model,
   then bootstrap the standardized residuals. Preserves volatility clustering and tail shape.

---

# Model Selection by Card Family

| Family | What it tests | Start with | Upgrade to | Scoring focus |
|---|---|---|---|---|
| **F1** | Single-asset accuracy | ARIMA | ETS or Theta | 71.4% CRPS + 28.6% tail (single-number cards) |
| **F2** | Regime shift + transfer | ARIMA per regime | VARX or TAR/STAR | Marginal CRPS with regime awareness |
| **F3** | Cross-asset dependence | Gaussian copula + ARIMA marginals | Student-t copula or VAR | 30% joint variogram |
| **F4** | Extreme events | GARCH + skewed-t | EVT + Student-t copula | 20% tail penalty, 1000+ draws |

## Build Order

| Step | Add | Covers | Effort |
|---|---|---|---|
| 1 | ARIMA per asset | F1 | Small |
| 2 | Gaussian copula on residuals | F3 | Medium |
| 3 | GARCH + skewed-t innovations | F4 | Medium |
| 4 | Regime-conditional parameters | F2 | Large |

## Libraries

| Purpose | Package |
|---|---|
| ARIMA, ETS, VAR, state-space | `statsmodels`, `pmdarima` |
| GARCH, DCC | `arch` |
| Copulas | `copulae`, `pyvinecopulib` |
| Quantile regression, trees | `statsmodels`, `scikit-learn` |
| Bootstrap | `arch.bootstrap` |
