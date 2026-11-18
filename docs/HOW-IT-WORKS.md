# How It Works — a Plain-English Walkthrough

This page explains the final agent (`my_agent_3/`) step by step for a reader who
doesn't work in quantitative finance. Technical terms are in **bold** the first time
they appear and are collected in the [glossary](#glossary) at the end.

---

## 1. What a "forecast" means here

A weather app that says "22°C tomorrow" gives a **point forecast**. One that says
"most likely 20–24°C, with a small chance of 28°C" gives a **probabilistic forecast**.
That is far more useful, because you can plan for the risk.

This competition asks for the second kind. For each task our agent produces **500
simulated futures** (1,000 for crash-risk tasks). Each simulated future is a complete
path: for example, "the 10-year yield is 3.9% in one week, 4.0% in a month, 4.2% in a
quarter". Taken together, the paths describe what is likely, what is possible, and what
is rare.

The judges score these simulated futures against what really happened. The score
combines three questions:

| Part of the score | Weight | The question it asks |
|---|---|---|
| **CRPS** | 50% | Was reality close to where we put most of our forecasts, and was our range sensibly sized? |
| **Variogram** | 30% | When we forecast several assets or dates together, did they move together the way reality did? |
| **Tail penalty** | 20% | Did we leave enough room for extreme moves? |

---

## 2. The rule that keeps the LLM honest

Language models are excellent readers but unreliable at producing numbers. So the
whole design follows one rule:

> **The LLM reads the documents and suggests adjustments. It never writes forecast
> numbers.**

Concretely, the LLM fills in a short, fixed form (see [`llm/signals.py`](../my_agent_3/llm/signals.py)):

| Signal | What it means | Hard limits |
|---|---|---|
| Mean shift | "The text suggests the centre should move up or down" | at most ±0.5 standard deviations |
| Variance scale | "The text suggests more (or less) uncertainty than usual" | 0.75× to 1.75× |
| Tail direction and strength | "The text hints at a crash (left tail) or spike (right tail)" | strength 0–1 |
| Correlation scale | "The text suggests markets will move more tightly together" | 1.0× to 1.5× |
| Basis | Is this **stated** in the documents or merely **inferred**? | Only stated evidence may *reduce* uncertainty |
| Evidence | A short explanation of what in the documents led to these signals | written into the rationale file |

Whatever the LLM says, the numbers are clamped to these limits. If the LLM is
unreachable, times out, or returns something malformed, every signal falls back to
"neutral" and the forecast simply proceeds on statistics alone. Responses are cached,
so re-running the same task gives exactly the same output.

---

## 3. The four layers

### Layer 1 — The statistician: a baseline from price history

[`statistical_layer/`](../my_agent_3/statistical_layer) contains several classic
models. Each one learns patterns from past data and simulates plausible futures:

- **Random walk with drift** — "tomorrow ≈ today, plus the recent trend". It sounds
  naive, but in finance it is famously hard to beat. Our version estimates the trend
  over the last year (252 trading days) and dampens it so that one noisy year can't
  produce a runaway forecast.
- **ARIMA** — learns how a series relates to its own recent past (momentum,
  mean-reversion). It picks its own settings by comparing candidates.
- **Exponential smoothing (ETS)** — a weighted average that trusts recent data more.
- **VAR** — like ARIMA, but for several series at once, so one market's moves can
  inform another's.
- **Regime-switching AR** — assumes the world flips between "states" (e.g. calm vs
  crisis) with different behaviour in each.
- **Historical analogs** — finds past moments that looked like today and replays what
  happened next.
- **Model selector** — tests candidates on recent *past* windows (never the future)
  and blends the ones that earn their place.

Every family has an ordered **fallback chain**. If the first model fails to fit, the
next is tried, all the way down to a plain random walk, which always works.

### Layer 2 — The risk manager: shaping the uncertainty

[`transformation_layer/`](../my_agent_3/transformation_layer) adjusts the *spread*
of the simulated futures:

- **GARCH** captures volatility clustering: turbulent days tend to follow turbulent
  days. We blend it with the long-run average and cap how far it can move, because
  GARCH on its own can overreact.
- **Fat tails (Student-t / skewed-t)** — real markets have more extreme days than the
  textbook bell curve predicts. This step stretches the tails while keeping the order
  of every path intact, so it doesn't scramble the paths.
- **Extreme-value theory** and **regime-switching GARCH** are also implemented for
  further experiments.

### Layer 3 — The connector: making assets move together

If we forecast the 2-year and 10-year Treasury yields separately, the simulation might
produce a future where one jumps and the other falls, even though historically they
move together. [`joint_draws_layer/`](../my_agent_3/joint_draws_layer) fixes this with
**copulas**:

1. Measure how the assets moved together historically, weighting recent data more
   and shrinking noisy estimates towards a safer default.
2. Re-pair the simulated paths so that "high" outcomes for one asset tend to coincide
   with "high" outcomes for its partners, without changing any single asset's own
   forecast.

The **Student-t copula** goes further: it makes assets more likely to crash
*together*, which is what actually happens in a crisis.

### Layer 4 — The reader: adjusting for the news

[`llm/`](../my_agent_3/llm) turns the dated documents into the bounded signals
described in section 2, then applies them:

- [`mean_shift.py`](../my_agent_3/llm/mean_shift.py) moves the whole forecast up or down.
- [`variance_scale.py`](../my_agent_3/llm/variance_scale.py) widens or narrows it.
- [`tail_skew.py`](../my_agent_3/llm/tail_skew.py) adds weight to one tail.
- [`copula_adjustment.py`](../my_agent_3/llm/copula_adjustment.py) tightens cross-asset links.

---

## 4. One task, start to finish

Take an F2 task: *"Japan, late 2012. Forecast the yen."* (`t2-F2-abenomics-2012`)

1. **Read safely.** [`input.py`](../my_agent_3/input.py) loads the task card, the yen
   price history, and the text library, then discards anything dated after the
   cut-off (30 November 2012).
2. **Baseline.** ARIMA simulates 500 future yen paths from the price history alone.
3. **Read the news.** The LLM reads the pre-cut-off documents. It might report that
   the text clearly signals aggressive monetary easing, meaning upward pressure on
   USD/JPY and more uncertainty than usual.
4. **Adjust.** The centre shifts within its ±0.5σ limit, and the range widens.
5. **Write up.** [`output.py`](../my_agent_3/output.py) writes the 500 paths, a
   metadata file, and a rationale that names every model used and summarises the
   text evidence behind the adjustments.

---

## 5. Why we trust the design choices

Every recipe in [`main.py`](../my_agent_3/main.py) was chosen by **back-testing**:
pretending it is a past date, forecasting with only the data available then, and
comparing with what actually happened. The rules we followed:

- **No look-ahead.** Models only ever see data up to the simulated "today".
- **Compare fairly.** The same tasks, dates and random seeds for every candidate.
- **Prefer robust over flashy.** A method that wins on average but occasionally fails
  badly loses to one that is slightly worse on average but never blows up.
- **Write down what didn't work.** Several ideas were tried and rejected, and those
  reports are kept in [`my_agent_3/testing/`](../my_agent_3/testing).

See [RESULTS.md](RESULTS.md) for the evidence.

---

## Glossary

| Term | Plain meaning |
|---|---|
| **Probabilistic forecast** | A forecast that gives a range of outcomes with likelihoods, not one number. |
| **Draws / simulated paths** | The individual simulated futures that together make up the forecast. |
| **As-of date / cut-off** | The "today" of a task. Nothing after it may be used. |
| **CRPS** | Continuous Ranked Probability Score: the main accuracy score for probabilistic forecasts. Lower is better. |
| **Variogram score** | Checks whether forecasts of several assets or horizons move together realistically. |
| **Tail / tail risk** | The extreme ends of the range: crashes and spikes. |
| **Random walk** | "Tomorrow equals today plus random noise." The standard benchmark in finance. |
| **Drift** | A steady trend added to a random walk. |
| **Shrinkage** | Pulling a noisy estimate towards a safer default to avoid overreacting. |
| **ARIMA** | A classic model that forecasts a series from its own recent values and recent surprises. |
| **ETS** | Exponential smoothing: a forecast that weights recent data more heavily. |
| **VAR** | Vector autoregression: ARIMA-style forecasting for several related series at once. |
| **Regime switching** | A model that assumes the market alternates between distinct states, such as calm and crisis. |
| **GARCH** | A model of changing volatility: storms cluster together. |
| **Fat tails / Student-t** | Distributions that allow more extreme outcomes than the normal bell curve. |
| **Copula** | A tool that glues separate forecasts together so they move jointly in a realistic way. |
| **Back-test** | Testing a method on the past as if it were live, to see how it would have done. |
| **Fallback chain** | An ordered list of backup models used if the preferred one fails. |
| **Oracle signal** | In an experiment, a "perfect" signal built from the known answer, used to measure the best-case ceiling of a method. |
