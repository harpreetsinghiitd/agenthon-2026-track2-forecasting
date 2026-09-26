# ARIMA — Why We Use It and How It Works

## What is ARIMA?

ARIMA stands for **AutoRegressive Integrated Moving Average**. It is a classical
statistical model for forecasting a single time series. It has three components,
each controlled by a parameter:

| Component | Parameter | What it does |
|---|---|---|
| **AR** — AutoRegressive | `p` | Predicts the next value using the previous `p` values |
| **I** — Integrated | `d` | Differences the series `d` times to remove trend before fitting |
| **MA** — Moving Average | `q` | Predicts the next value using the previous `q` forecast errors |

The model is written as **ARIMA(p, d, q)**.

### The three components in plain English

**AR (AutoRegressive) — "the past predicts the future"**

If a stock went up yesterday, it is slightly more likely to go up today.
AR captures this tendency. It looks back `p` days and uses a weighted
average of those values to predict today.

Example: `ARIMA(2, 0, 0)` uses the last two days to predict tomorrow.

**I (Integrated) — "remove the trend first"**

Most financial series are not stationary — they drift upward or downward over
time. ARIMA does not like that. So it differences the series: instead of
modeling the price, it models the **change** in price. If one difference is
not enough, it differences again (rare in practice for daily data).

Example: prices drift up over 24 years. ARIMA with `d = 1` models daily
returns instead.

**MA (Moving Average) — "the past errors predict the future"**

If yesterday's forecast was too high, today's forecast should probably be
adjusted down. MA captures this. It looks at the last `q` forecast errors
and uses them to correct the next prediction.

Example: `ARIMA(0, 0, 2)` uses the last two residuals to adjust tomorrow's
prediction.

### How ARIMA makes a forecast

1. **Fit** — find the values of `p`, `d`, `q`, and the coefficient weights that
   best explain the historical data.
2. **Predict** — apply the fitted model to produce a forecast for the next
   `h` steps.
3. **Uncertainty** — along with the mean forecast, ARIMA gives a standard
   error at each step. That is what we use to generate draws.

---

## Why ARIMA for this competition?

### 1. It fits the scoring

The competition scores each card on:
- **50% marginal CRPS** — needs a well-calibrated marginal distribution
- **30% joint variogram** — needs correct cross-asset dependence
- **20% tail penalty** — needs realistic tail behavior

ARIMA directly addresses the first 50%. It produces both a point forecast
**and** a standard error at each horizon, which is exactly what you need to
generate Monte Carlo draws. For F1 cards (single asset, single horizon),
ARIMA alone gets you the 71.4% effective CRPS score.

### 2. It fits the signature

The forecaster must take a panel and return an array of draws of shape
`[n_draws, n_assets, n_horizons]`. ARIMA fits cleanly:

```python
model = ARIMA(series, order=(p, d, q)).fit()
fc = model.get_forecast(steps=max_h)
mu = fc.predicted_mean           # mean at each step
se = fc.se_mean                  # std error at each step
draws = rng.normal(mu, se, (n_draws, max_h))
