# F1 local-drift grid: multi-seed outer backtest

## Method

Evaluated 84 previously defined outer origins across 21 eligible F1 cards, using 3 fixed seeds and 500 draws per run. Grid: windows [63, 126, 252, 504, None] and shrinkage [0.0, 0.5, 1.0]. Each origin was fitted using only panel data through t1; post-t1 data was used only for scoring. No text was loaded. The metric is the card-weighted composite divided by same-origin, same-seed random-walk loss; RW = 1.0 and lower is better. Confidence intervals resample cards as clusters (4000 bootstrap replicates), preserving within-card dependence.

## Results

| Window | Shrink | Macro mean card ratio | Median card ratio | Median origin ratio | Wins vs RW | Mean gain vs RW | 95% card-cluster CI for gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| 252 | 1.00 | 0.8962 | 0.8872 | 0.9419 | 150/252 | 0.1038 | [0.0438, 0.1651] |
| 504 | 1.00 | 0.9168 | 0.9300 | 0.9324 | 153/252 | 0.0832 | [0.0271, 0.1416] |
| full | 1.00 | 0.9228 | 0.8627 | 0.9462 | 157/252 | 0.0772 | [0.0135, 0.1321] |
| 126 | 1.00 | 0.9245 | 0.9146 | 0.9086 | 156/252 | 0.0755 | [-0.0475, 0.1764] |
| full | 0.50 | 0.9464 | 0.9227 | 0.9567 | 160/252 | 0.0536 | [0.0218, 0.0825] |
| 252 | 0.50 | 0.9547 | 0.9742 | 0.9631 | 134/252 | 0.0453 | [-0.0131, 0.1042] |
| 504 | 0.50 | 0.9851 | 0.9894 | 0.9544 | 147/252 | 0.0149 | [-0.0431, 0.0742] |
| full | 0.00 | 1.0000 | 1.0000 | 1.0000 | 0/252 | 0.0000 | [0.0000, 0.0000] |
| 126 | 0.50 | 1.0424 | 0.9019 | 0.9471 | 146/252 | -0.0424 | [-0.2541, 0.1156] |
| 63 | 1.00 | 1.0579 | 0.9567 | 0.9440 | 147/252 | -0.0579 | [-0.3098, 0.1301] |
| 504 | 0.00 | 1.0951 | 1.0776 | 1.0000 | 80/252 | -0.0951 | [-0.1857, -0.0052] |
| 252 | 0.00 | 1.1265 | 1.1424 | 1.0589 | 118/252 | -0.1265 | [-0.2112, -0.0377] |
| 63 | 0.50 | 1.3282 | 1.1011 | 0.9788 | 132/252 | -0.3282 | [-0.7750, -0.0087] |
| 126 | 0.00 | 1.3402 | 1.0861 | 1.1407 | 94/252 | -0.3402 | [-0.6469, -0.0959] |
| 63 | 0.00 | 1.8995 | 1.5780 | 1.3537 | 92/252 | -0.8995 | [-1.4933, -0.4164] |

## Decision

**Grid candidate:** window=252, shrink=1.00. It had positive mean gain, a positive 95% card-cluster lower confidence bound, and won on more than half of seeded cases when compared as a standalone forecast against random walk.

The required full-selector follow-up did not clear the promotion rule: it achieved a 0.9309 macro card ratio to random walk on 84 origins with 56/84 wins, but the card-cluster 95% interval for gain was [-0.0087, 0.1479], including no gain. The production selector therefore remains at its prior 252-window, 0.5-shrinkage setting. See [F1_SELECTOR_DRIFT_PROMOTION.md](F1_SELECTOR_DRIFT_PROMOTION.md).

Forecast-generation failures: 0. See [f1_drift_grid_backtest.csv](f1_drift_grid_backtest.csv), [f1_drift_grid_summary.json](f1_drift_grid_summary.json), and the failures JSON for row-level details.
