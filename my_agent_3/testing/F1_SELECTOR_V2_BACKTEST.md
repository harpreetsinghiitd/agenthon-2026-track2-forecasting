# Revised F1 selector experiment: nested outer backtest

## Method

Recomputed 84 previously defined outer origins across 21 F1 cards with 500 Monte Carlo draws. All selector validation and fitting used panel rows through each t1; later data was used only for scoring. No text was loaded. The metric is card-weighted composite relative to the same-origin plain random walk (RW = 1.0); lower is better. This experiment added plain RW and positive-level log-change ARIMA candidates and let any best-scoring inner-validation model anchor the mixture.

## Candidate results

| Model | Origins | Mean relative composite | Median | Wins vs RW |
|---|---:|---:|---:|---:|
| ARIMA | 84 | 1.2379 | 0.9404 | 53/84 |
| ARIMA_log_changes | 76 | 1.3606 | 1.0059 | 37/76 |
| ETS | 84 | 9.7298 | 2.1901 | 19/84 |
| local_drift_rw | 84 | 1.1176 | 0.9317 | 50/84 |
| random_walk | 84 | 1.0000 | 1.0000 | 0/84 |
| selector_v2 | 84 | 1.0118 | 0.9963 | 45/84 |

## Comparison with the previous selector

On the same 84 outer origins, the previous saved selector averaged 0.9482 (median 0.9002); this experiment averaged 1.0118 (median 0.9963) and beat the previous per-origin saved score on 38/84 origins. The runs used different Monte Carlo seeds, so this comparison is indicative rather than seed-matched.

The experiment did not improve aggregate results. Production remains on the prior conservative selector rule: ARIMA is the center unless local-drift random walk clears the paired-gain and win-rate thresholds. The log-change ARIMA implementation is retained as a research candidate but excluded from production selection. It was applicable to 76 positive-level origins; the other 8 were log-return targets.

These are public-panel historical backtests, not sealed competition outcomes. The full rows are in [f1_selector_v2_backtest.csv](f1_selector_v2_backtest.csv).
