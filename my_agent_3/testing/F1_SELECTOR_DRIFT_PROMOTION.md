# F1 selector after drift promotion

## Method

Ran the complete production F1 selector on the same 84 outer origins across 21 eligible F1 cards, with 500 draws and one fixed seed per origin. Each selector run used only data through t1; later observations were used only to score. No text was loaded. Candidate 252-step local drift uses full drift shrinkage, as chosen by the preceding 3-seed drift grid. Card-level ratios pool raw composite loss across origins within each card, then average cards equally. The interval resamples cards as clusters (4,000 replicates).

## Result

- Macro mean card ratio to random walk: **0.9309** (RW = 1.0).
- Median card ratio: **0.9057**; median outer-origin ratio: **0.9215**.
- Selector won on **56/84** outer origins.
- Card-cluster 95% interval for relative gain: **[-0.0087, 0.1479]**.

## Promotion decision

The selector did not pass the full-pipeline check. Revert the drift parameter change and keep the prior production selector.

The previous saved selector's 84-origin report used a different Monte Carlo seed and a per-origin ratio aggregate; it is not treated as a seed-matched comparison. The preceding drift grid is the multi-seed evidence for the setting. This is still a public-panel backtest, not sealed competition scoring.

## Later deployment decision

This report records the historical full-selector promotion check. On 2026-10-09, to restore the best-performing F1 iteration, the primary F1 route was changed to the standalone 252-step, full-shrinkage local-drift random walk, whose three-seed grid results are in [F1_DRIFT_GRID_BACKTEST.md](F1_DRIFT_GRID_BACKTEST.md). The model selector remains a fallback. Thus the full-selector confidence check above is not being represented as a pass.
