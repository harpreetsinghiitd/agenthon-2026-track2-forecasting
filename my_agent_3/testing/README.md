## Executive summary (read this first)

This folder records rolling-origin backtests of the numerical forecasting pipeline in `my_agent_3`. In the latest comparison, the combined pipeline beat a standalone random walk on median CRPS for F1, F2, and F3 cards, while F4 was nearly even. F1's multi-horizon variogram results were a concern: marginal accuracy often improved while cross-horizon dependence worsened. These are local historical comparisons, not official normalized competition scores.

## Backtest design

- **Scope:** 104 authored cards were considered. Ninety cards had usable target rows and forecast windows; 14 were excluded. The valid set contained 21 F1, 23 F2, 17 F3, and 29 F4 cards.
- **Origins:** Four historical forecast origins were selected per valid card, spread across the available valid origin range. Each origin used at least 100 observations for fitting and had enough later panel observations to score the card's authored horizons.
- **Forecasting:** Each origin produced 500 draws from the current numerical family pipeline and 500 draws from the standalone random-walk model. The same deterministic seed was used for the paired model forecasts.
- **Information cutoff:** Models saw only observations on or before each simulated origin. Later observations already present in the public panel were used only as historical backtest targets.
- **Models:** F1 log-return cards tried the 10-step block bootstrap before ARIMA; F1 level cards started with ARIMA. F2 used ARIMA with its regime-AR fallback. F3 used its numerical forecaster with GARCH scaling and a Student-t copula where available. F4 used its numerical forecaster with GARCH scaling and skewed-t draws where available. LLM conditioning was neutral and did not affect these results.
- **Metrics:** Marginal CRPS, variogram loss, and tail pinball loss at levels 0.01, 0.05, 0.95, and 0.99 were computed with the project scoring toolkit. Lower is better.

This comparison tests the current combined pipeline against a random-walk-only forecast. It does not isolate the contribution of every individual layer. A component-specific ablation is needed to measure the effect of the copula, GARCH, block bootstrap, and tail transformation separately.

## Results by family

The percentage changes compare combined-model loss with random-walk loss. Negative values favor the combined model. Median and mean changes are calculated across each card's four-origin average; win counts use card averages. Origin win rates count paired forecast origins individually.

| Family | Cards | CRPS median change | CRPS mean change | Cards with lower CRPS | Origin CRPS wins | Tail median change | Cards with lower tail loss | Variogram median change | Cards with lower variogram |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | 21 | -15.91% | -10.02% | 14/21 (66.7%) | 53/84 (63.1%) | -0.97% | 11/21 (52.4%) | +115.24% | 2/16 (12.5%) |
| F2 | 23 | -8.47% | -12.23% | 19/23 (82.6%) | 52/92 (56.5%) | -0.84% | 12/23 (52.2%) | -12.62%* | 1/1* |
| F3 | 17 | -9.05% | -7.20% | 14/17 (82.4%) | 43/68 (63.2%) | -3.36% | 10/17 (58.8%) | -3.89% | 10/17 (58.8%) |
| F4 | 29 | -0.44% | -5.96% | 15/29 (51.7%) | 65/116 (56.0%) | -0.86% | 17/29 (58.6%) | +7.03%* | 1/2* |

`*` Variogram comparisons are informative only when a card has a nonzero variogram term. Most F2 and F4 cards had one target cell, for which the variogram is zero by construction. The F1 table includes only the 16 cards with a measurable variogram; single-cell or otherwise zero-denominator comparisons are omitted. F3 had 17 measurable cards.

There is no single official composite score in this report. The local practice cards do not provide the per-card reference scales needed to normalize CRPS, variogram, and tail loss as the official composite does. Raw component values also have different units across cards, so the analysis compares each model to random walk within the same card instead of averaging raw losses across cards.

## Selected card results

Values below are the mean loss across the four origins for that card. The percentage change is relative to random walk; negative favors the combined model.

| Card | Metric | Combined | Random walk | Change |
|---|---|---:|---:|---:|
| `t2-F1-ai-mom-2024` | CRPS | 0.079012 | 0.104957 | -24.7% |
|  | Tail pinball | 0.008764 | 0.007752 | +13.1% |
| `t2-F1-pause-2006` | CRPS | 0.761127 | 0.797865 | -4.6% |
|  | Variogram | 1.34721 | 1.48133 | -9.1% |
|  | Tail pinball | 0.129524 | 0.113137 | +14.5% |
| `t2-F2-abenomics-2012` | CRPS | 4.60973 | 4.67479 | -1.4% |
|  | Tail pinball | 0.823298 | 0.885258 | -7.0% |
| `t2-F3-bear-flattener-2022` | CRPS | 0.643263 | 0.665482 | -3.3% |
|  | Variogram | 8.56051 | 8.89575 | -3.8% |
|  | Tail pinball | 0.134839 | 0.127492 | +5.8% |
| `t2-F4-aud-gfc-2008` | CRPS | 0.016092 | 0.019970 | -19.4% |
|  | Tail pinball | 0.001914 | 0.002382 | -19.7% |

## Interpretation and next experiments

1. **Improve multi-horizon path coherence.** F1 CRPS usually improved, but variogram loss worsened on 14 of 16 cards with measurable variograms. Test coherent path sampling for every horizon, rather than drawing each horizon independently, and score the path structure directly.
2. **Test the F1 block bootstrap by target type and card.** It improved central CRPS on many return cards, but tail loss was close to even overall and varied materially by card. Compare several block lengths using earlier origins only, then evaluate a held-out date range.
3. **Ablate F3 layers.** Compare ARIMA alone, ARIMA plus copula, ARIMA plus GARCH, and the full combination against the same random-walk origins. This will show whether GARCH scaling helps or offsets dependence gains.
4. **Tune F4 tail treatment on held-out origins.** Tail loss improved on 17 of 29 cards, while median CRPS was almost unchanged. Compare GARCH and skewed-t draws separately and together; select settings on training origins, not the final holdout.
5. **Expand origin coverage for tail conclusions.** Four origins per card are not enough to establish reliable 1% tail calibration. More origins can improve the evidence, though they cannot create additional independent extreme events.

The largest F1 regressions in per-card average CRPS were `t2-F1-cad-boc-2017` (+35.7%), `t2-F1-conflicting-texts-2024` (+19.5%), and `t2-F1-normalization-2017` (+14.1%). This reinforces the need for card-level robustness rather than selecting a model from the average alone.

## Excluded cards

Fourteen of the 104 considered cards did not produce a valid backtest:

- **Insufficient valid origin/target windows (9):** `t2-F1-cpi-glidepath-2023`, `t2-F1-sahm-watch-2024`, `t2-F3-boj-ust-channel-2023`, `t2-F3-divergence-2014`, `t2-F3-dollar-vortex-2022`, `t2-F3-election-2024-joint`, `t2-F3-safe-haven-paradox-2011`, `t2-F4-covid-nfp-2020`, `t2-F4-cpi-vintage-2022`.
- **No panel rows for declared target assets (5):** `t2-F2-cnh-stress-2015`, `t2-F2-cnh-tradewar-2018`, `t2-F2-fragile-five-brl-2013`, `t2-F2-fragile-five-inr-2013`, `t2-F3-fragile-five-joint-2013`.

## Limitations

- Four origins per card give a broad screening comparison, not a precise estimate of future performance. Forecast windows can overlap, so origin-level wins are not independent statistical trials.
- Results are specific to the public panels, chosen origins, current implementations, seed, and 500 draws. Changing any of these can change the measured losses.
- Historical backtests cannot reproduce the official sealed outcomes or their normalized reference scales. Only the CodaBench score can establish official relative performance.
- The raw origin-level output from this run was saved outside the repository at `/tmp/agenthon_rolling_rows.csv`; the exclusion log was saved at `/tmp/agenthon_rolling_skips.json`.
