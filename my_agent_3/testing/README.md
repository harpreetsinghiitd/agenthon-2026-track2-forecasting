## Executive summary (read this first)

This folder records rolling-origin backtests of the numerical forecasting pipeline in `my_agent_3`. The original broad comparison showed lower median CRPS than random walk on F1, F2, and F3, with F4 nearly even, but raised concerns about multi-horizon dependence. A focused comparison after path-coherence changes improved variogram loss on two matched cards with nearly unchanged CRPS; that small sample is promising but not conclusive. All results are local historical comparisons, not official normalized competition scores.

The latest all-family, document-free backtest is in [`layer_backtest/REPORT.md`](layer_backtest/REPORT.md). It covers 90 cards with two non-overlapping forecast origins per card and includes family-level actual-versus-predictive-mean plots.

The broad 90-card results below are a snapshot from **before** the path-coherence changes documented later in this file. They should not be read as scores for the current implementation.

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

## Path-coherence update and focused backtest

The forecasters were updated so each draw is a complete multi-horizon path: ARIMA uses state-space simulation, ETS resamples short contiguous residual blocks, copulas reassign whole asset paths using terminal-horizon ranks, and the skewed-t transformation preserves each marginal's ranks. This reduces the risk of transformations independently shuffling forecast horizons.

I reran five multi-horizon cards and one F4 single-horizon card at the **same four origins, seeds, and 500 draws** used in the earlier full run. The F4 card exercises the updated tail transform; its variogram is zero by construction. Random-walk losses match the previous run for overlapping cards, so those comparisons are paired. Changes below compare the updated full pipeline with random walk; negative means lower loss.

| Card | CRPS: combined vs RW | CRPS change | Variogram: combined vs RW | Variogram change | Tail: combined vs RW | Tail change |
|---|---:|---:|---:|---:|---:|---:|
| `t2-F1-aud-on-hold-2016` | 0.031995 vs 0.031611 | +1.2% | 0.007046 vs 0.015125 | -53.4% | 0.003271 vs 0.003671 | -10.9% |
| `t2-F1-conflicting-texts-2024` | 0.425581 vs 0.356539 | +19.4% | 0.200105 vs 0.186020 | +7.6% | 0.040935 vs 0.038773 | +5.6% |
| `t2-F1-pause-2006` | 0.752256 vs 0.797865 | -5.7% | 1.121588 vs 1.481332 | -24.3% | 0.127725 vs 0.113137 | +12.9% |
| `t2-F3-bear-flattener-2022` | 0.644344 vs 0.665482 | -3.2% | 7.342118 vs 8.895749 | -17.5% | 0.137039 vs 0.127492 | +7.5% |
| `t2-F3-conundrum-joint-2005` | 0.352532 vs 0.519024 | -32.1% | 3.055292 vs 5.905725 | -48.3% | 0.061301 vs 0.051491 | +19.1% |
| `t2-F4-aud-gfc-2008` | 0.015868 vs 0.019970 | -20.5% | 0.000000 vs 0.000000 | n/a | 0.001873 vs 0.002382 | -21.4% |

For the two cards with saved pre-change combined results, path coherence changed the tradeoff as follows:

| Card | Pre-change CRPS / variogram / tail | Updated CRPS / variogram / tail | Change in those losses |
|---|---:|---:|---:|
| `t2-F1-pause-2006` | 0.761127 / 1.34721 / 0.129524 | 0.752256 / 1.121588 / 0.127725 | CRPS -1.2%; variogram -16.8%; tail -1.4% |
| `t2-F3-bear-flattener-2022` | 0.643263 / 8.56051 / 0.134839 | 0.644344 / 7.342118 / 0.137039 | CRPS +0.2%; variogram -14.2%; tail +1.6% |

This focused check supports the intended path-coherence improvement: variogram loss fell on both matched cards, while CRPS stayed nearly flat. Tail loss worsened on the F3 card and improved slightly on F1 pause. Five cards are not enough to establish general gains, so extend this paired comparison before treating the new path sampler as a validated overall improvement.

## Limited check: one card from each family

I also ran a small family-level check on one representative card per family, using four historical origins and 500 draws per forecast. The combined pipeline was compared with random walk. Negative changes favor the combined pipeline; origin wins count the four individual dates.

| Family / card | CRPS change; wins | Tail-loss change; wins | Variogram change; wins |
|---|---:|---:|---:|
| F1 `t2-F1-pause-2006` | -5.7%; 3/4 | +12.9%; 1/4 | -24.3%; 2/4 |
| F2 `t2-F2-abenomics-2012` | -3.5%; 2/4 | -2.2%; 3/4 | Zero for both models (single target cell) |
| F3 `t2-F3-bear-flattener-2022` | -3.2%; 2/4 | +7.5%; 2/4 | -17.5%; 3/4 |
| F4 `t2-F4-aud-gfc-2008` | -20.5%; 3/4 | -21.4%; 3/4 | Zero for both models (single target cell) |

The limited check suggests CRPS improved on all four selected cards, but it is only one card per family. Tail performance remains mixed for F1 and F3, while the F2 and F4 cards improved on both reported marginal metrics. Treat this as a smoke check of behavior after the path changes, not as a family-wide performance estimate.

## Interpretation and next experiments

1. **Improve multi-horizon path coherence.** F1 CRPS usually improved, but variogram loss worsened on 14 of 16 cards with measurable variograms. Test coherent path sampling for every horizon, rather than drawing each horizon independently, and score the path structure directly.
2. **Test the F1 block bootstrap by target type and card.** It improved central CRPS on many return cards, but tail loss was close to even overall and varied materially by card. Compare several block lengths using earlier origins only, then evaluate a held-out date range.
3. **Ablate F3 layers.** Compare ARIMA alone, ARIMA plus copula, ARIMA plus GARCH, and the full combination against the same random-walk origins. This will show whether GARCH scaling helps or offsets dependence gains.
4. **Tune F4 tail treatment on held-out origins.** Tail loss improved on 17 of 29 cards, while median CRPS was almost unchanged. Compare GARCH and skewed-t draws separately and together; select settings on training origins, not the final holdout.
5. **Expand origin coverage for tail conclusions.** Four origins per card are not enough to establish reliable 1% tail calibration. More origins can improve the evidence, though they cannot create additional independent extreme events.

The largest F1 regressions in per-card average CRPS were `t2-F1-cad-boc-2017` (+35.7%), `t2-F1-conflicting-texts-2024` (+19.5%), and `t2-F1-normalization-2017` (+14.1%). This reinforces the need for card-level robustness rather than selecting a model from the average alone.

## F1 analogs versus ARIMA

I compared the analog forecaster directly with ARIMA on every F1 card with enough history and a scorable future window. Both models received the same as-of history, four evenly spaced rolling origins per card, 500 draws, and the same seed at each origin. A model's later observations were used only for scoring. This is an ARIMA-only comparison; it does not include the block-bootstrap fallback or text signals.

There were 23 authored F1 cards. Twenty-one had four valid rolling origins each (84 paired ARIMA forecasts); `t2-F1-cpi-glidepath-2023` and `t2-F1-sahm-watch-2024` had no valid origins under the minimum-history and horizon requirements. Analog sampling could find at least 20 complete donor paths at 57/84 origins (67.9%). The other 27 origins are excluded from the direct model comparison because the analog model was unavailable there; in the production pipeline those cases fall back to the next F1 model.

For the 57 matched origins, changes are `100 × (analog loss − ARIMA loss) / ARIMA loss`; positive means analogs are worse. Percent changes are averaged across paired origins, and card win rates use the mean percentage change within each card.

| Metric | Mean paired change | Median paired change | Analog wins |
|---|---:|---:|---:|
| CRPS | +6.68% | -5.83% | 31/57 origins; 8/21 cards |
| Tail pinball | +12.23% | -1.00% | 29/57 origins; 10/21 cards |
| Single-cell composite | +2.77% | -3.83% | 33/57 origins; 9/21 cards |

The composite uses the single-cell renormalized weights: 71.4% CRPS and 28.6% tail pinball. Across all 57 matched forecasts, the raw mean composite was 0.2905 for analogs and 0.2556 for ARIMA. The mean per-card percentage change in composite was +8.58%, with a median of +3.38%; analogs had lower card-average composite on 9 of 21 cards.

| F1 card | Matched origins | Mean composite change vs ARIMA | Analog wins |
|---|---:|---:|---:|
| `t2-F1-ai-mom-2024` | 3 | -8.61% | 2/3 |
| `t2-F1-aud-on-hold-2016` | 3 | +20.09% | 2/3 |
| `t2-F1-cad-boc-2017` | 3 | -11.59% | 2/3 |
| `t2-F1-chf-highly-valued-2021` | 3 | -42.64% | 3/3 |
| `t2-F1-conflicting-texts-2024` | 3 | +6.95% | 1/3 |
| `t2-F1-considerable-period-2003` | 1 | +37.47% | 0/1 |
| `t2-F1-conundrum-2005` | 2 | +12.73% | 1/2 |
| `t2-F1-dkk-peg-2019` | 3 | -10.32% | 2/3 |
| `t2-F1-eur-range-2017` | 3 | +2.51% | 1/3 |
| `t2-F1-fed-put-2019` | 3 | -9.01% | 3/3 |
| `t2-F1-greater-confidence-2024` | 3 | -8.69% | 2/3 |
| `t2-F1-hawkish-cut-2024` | 3 | +11.96% | 1/3 |
| `t2-F1-jpy-ycc-flex-2018` | 3 | +3.38% | 1/3 |
| `t2-F1-liftoff-telegraph-2015` | 3 | -3.92% | 2/3 |
| `t2-F1-measured-pace-2004` | 1 | +114.43% | 0/1 |
| `t2-F1-normalization-2017` | 3 | +6.52% | 1/3 |
| `t2-F1-patient-dropped-2015` | 3 | +4.01% | 2/3 |
| `t2-F1-pause-2006` | 2 | +65.60% | 0/2 |
| `t2-F1-qe3-end-2014` | 3 | -19.49% | 3/3 |
| `t2-F1-third-cut-2019` | 3 | +18.20% | 2/3 |
| `t2-F1-ust-positioning-2018` | 3 | -9.51% | 2/3 |

**Conclusion:** the evidence does not show that analogs are better than ARIMA overall. Analogs win on some cards, but ARIMA has lower mean composite loss on this paired sample, and analog donors are unavailable at about one-third of tested origins. This motivated the per-card selector: it validates analog, ARIMA, and ETS on recent past-only windows, then refits the selected model at the actual cutoff. Results are public-panel historical checks, not sealed official scores. The raw paired rows are saved outside the repository at `/tmp/agenthon_f1_arima_analog_rows.csv`.

## F1 per-card model selector

The F1 pipeline evaluates up to four recent non-overlapping pseudo-origins whose full target paths are already observed by the actual `asof`. It compares target-aware ARIMA variants, ARIMA with residual-block innovations, ETS, analog, and local-drift random walk using only each origin's available prefix. Candidate losses are normalized to random walk on the same fold. A challenger is eligible only with at least 10% mean relative gain over ARIMA and wins on at least 60% of paired folds; eligible challenger mass is capped at 25%, with ARIMA as the anchor. It then samples whole forecast paths from this mixture. The random-walk ratios are a local model-selection proxy, not the organizer's official normalization.

Integration smoke checks passed for `t2-F1-ai-mom-2024` (log-return, ARIMA-centered mixture from 4 folds) and `t2-F1-aud-on-hold-2016` (level, ARIMA only); both wrote 500 finite draws and passed the organizer smoke gate.

## Exhaustive F1 nested backtest

To test the updated selector, I ran a nested rolling-origin comparison across the full 23-card F1 roster. Twenty-one cards supported four outer forecast origins each, for 84 held-out forecasts; `t2-F1-cpi-glidepath-2023` and `t2-F1-sahm-watch-2024` lacked enough history for the outer horizon plus inner validation windows. At every outer origin, the selector saw only the panel through that date and used up to four earlier non-overlapping validation origins. Fixed target-aware candidates and baselines were also evaluated on the same outer windows. Each forecast used 500 draws and deterministic seeds. The run took about 23 minutes.

The comparison metric is each model's raw card-weighted composite divided by the random-walk composite for the same card and origin. Therefore random walk is 1.0; lower is better. This provides a within-card scale reference for pooling results, but it is not the organizer's signed reference-scale normalization.

| Model | Outer forecasts available | Mean relative composite | Median | Better than RW |
|---|---:|---:|---:|---:|
| Local-drift random walk | 84/84 | 0.9212 | 0.8969 | 54/84 |
| Historical analog | 60/84 | 0.9417 | 0.9883 | 31/60 |
| ARIMA-centered selector | 84/84 | 0.9482 | 0.9002 | 53/84 |
| ARIMA (AIC order search) | 84/84 | 0.9635 | 0.9116 | 56/84 |
| ARIMA(0,1,1) with drift | 75/84 | 1.0166 | 1.0059 | 37/75 |
| ARIMA(1,1,0) with drift | 71/84 | 1.0106 | 1.0091 | 31/71 |
| ARIMA residual-block innovations | 84/84 | 1.3985 | 1.2752 | 32/84 |
| ETS | 84/84 | 1.3490 | 1.2318 | 26/84 |
| Random walk | 84/84 | 1.0000 | 1.0000 | 0/84 |

Analog is available on only 60 origins because 24 fits lacked the 20 required donor paths. ARIMA(0,1,1) and ARIMA(1,1,0) drift variants failed on 9 and 13 origins, respectively. Fixed ARIMA(1,0,0) and block-bootstrap candidates were available on only 8 origins because they apply only to log-return cards. The selector used an ARIMA-centered mixture on 42 origins and ARIMA alone on 42.

On identical outer origins, the selector's mean relative composite was 0.0153 lower than fixed ARIMA; it had lower loss on 46/84 origins and on 11/21 card means. Its mean relative score (0.9482) improved on the ARIMA mean (0.9635) modestly. However, local-drift random walk was stronger overall at 0.9212, and beat the selector on 48/84 origins (the selector won 36/84). The selector beat ETS on 63/84 and analog on 34/60 matched origins. Candidate means: local-drift RW 0.9212, analog 0.9417 (availability-limited), ARIMA 0.9635, ETS 1.3490, and ARIMA residual-block 1.3985. These pooled averages are means of per-origin ratios, with each origin divided by its own RW score; they do not represent a common absolute loss scale.

**Conclusion:** the revised selector modestly improves on fixed ARIMA in this nested sample, but the simplest local-drift random walk performed best on mean relative composite. That points to drift estimation/shrinkage as the most promising Layer 1 area to refine; the selector's gains over ARIMA are modest and should be treated as historical evidence, not a guarantee on sealed cards. The CSV contains 764 rows of outer-origin model results, selector diagnostics, and scores: [f1_nested_backtest.csv](f1_nested_backtest.csv).

## Follow-up: selector-anchor and log-change experiment

To test whether the selector should choose any candidate as its anchor, I reran the same 84 outer origins across the 21 eligible F1 cards with 500 draws. The experiment added plain random walk and ARIMA-on-log-changes for positive level series, and let the lowest inner-validation mean become the anchor. It returned a mean relative composite of **1.0118** (median **0.9963**) versus the prior saved selector's **0.9482** (median **0.9002**) on the same outer origins; it beat the prior saved per-origin score on 38/84. The separate log-change candidate averaged **1.3606** (median **1.0059**) against random walk on its 76 applicable positive-level origins. The all-candidate anchor experiment therefore did not improve the historical results. These runs used different Monte Carlo seeds, so the comparison is indicative; score estimates are seed-sensitive.

The production selector was restored to its prior conservative ARIMA anchor, allowing local-drift random walk to take the anchor only after a material paired validation gain and at least a 60% fold win rate. The new [level_change_arima.py](../statistical_layer/level_change_arima.py) remains available for controlled research but is not in the production candidate set. The full experimental comparison is in [F1_SELECTOR_V2_BACKTEST.md](F1_SELECTOR_V2_BACKTEST.md), with per-origin rows in [f1_selector_v2_backtest.csv](f1_selector_v2_backtest.csv). This experiment argues against making the mean forecast wigglier by default; the next useful change is repeated-seed validation or a better horizon-specific drift estimator, tested on the same outer origins before promotion.

## Multi-seed drift grid and full-selector promotion check

I evaluated 15 drift-window/shrinkage settings on the same 84 outer origins, with three fixed seeds and 500 draws. Composite losses were pooled across origins within each card and then macro-averaged across the 21 cards; the 95% interval resampled cards as clusters. The best standalone setting was a 252-observation window with full shrinkage: macro card ratio **0.8962** against random walk, with **150/252** seed-origin wins and a **[0.0438, 0.1651]** confidence interval for relative gain. Grid results are in [F1_DRIFT_GRID_BACKTEST.md](F1_DRIFT_GRID_BACKTEST.md).

I then ran the complete selector with that setting on all 84 outer origins. It scored **0.9309** macro card ratio to random walk, with **56/84** origin wins, but its card-cluster interval was **[-0.0087, 0.1479]**, which includes no gain. Per the promotion rule, I **did not change the production drift setting**; it remains 252 observations with 0.5 shrinkage. The experimental override is available to the selector for further paired evaluation, and the full-pipeline result is in [F1_SELECTOR_DRIFT_PROMOTION.md](F1_SELECTOR_DRIFT_PROMOTION.md) with raw results in [f1_selector_drift_promo.csv](f1_selector_drift_promo.csv).

**Current deployment decision (2026-10-09):** to restore the most profitable F1 iteration, `main.py` now routes F1 directly to the best standalone grid model: 252-observation local-drift random walk with full shrinkage. The selector remains the fallback if that forecaster fails. This uses the grid's paired three-seed standalone result (macro card ratio 0.8962, card-cluster gain interval [0.0438, 0.1651]); it does not claim that the full selector passed its separate confidence check. The F1 daily charts were regenerated from the active direct-drift route.

## Excluded cards

Fourteen of the 104 considered cards did not produce a valid backtest:

- **Insufficient valid origin/target windows (9):** `t2-F1-cpi-glidepath-2023`, `t2-F1-sahm-watch-2024`, `t2-F3-boj-ust-channel-2023`, `t2-F3-divergence-2014`, `t2-F3-dollar-vortex-2022`, `t2-F3-election-2024-joint`, `t2-F3-safe-haven-paradox-2011`, `t2-F4-covid-nfp-2020`, `t2-F4-cpi-vintage-2022`.
- **No panel rows for declared target assets (5):** `t2-F2-cnh-stress-2015`, `t2-F2-cnh-tradewar-2018`, `t2-F2-fragile-five-brl-2013`, `t2-F2-fragile-five-inr-2013`, `t2-F3-fragile-five-joint-2013`.

## Limitations

- Four origins per card give a broad screening comparison, not a precise estimate of future performance. Forecast windows can overlap, so origin-level wins are not independent statistical trials.
- Results are specific to the public panels, chosen origins, current implementations, seed, and 500 draws. Changing any of these can change the measured losses.
- Historical backtests cannot reproduce the official sealed outcomes or their normalized reference scales. Only the CodaBench score can establish official relative performance.
- The six-card path-coherence origin-level output is temporarily saved outside the repository at `/tmp/agenthon_path_coherence_six.csv`; the four-family check is at `/tmp/agenthon_four_family_rows.csv`. The original 90-card run is summarized above; its temporary raw CSV was overwritten by focused follow-up runs.
