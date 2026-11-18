# Agenthon Track 2 — `my_agent_3`

## Executive summary (read this first)

`my_agent_3` is our final Docker forecasting agent for Agenthon 2026 Track 2. It reads a numerical panel and a frozen, dated text corpus, produces probabilistic Monte Carlo forecast draws, and writes the competition's required output files. `main.py` owns the family-specific pipeline composition. Numerical implementations live in the statistical, transformation, and joint-draw layers; the `llm/` package turns the text corpus into bounded signals that adjust those draws.

For a non-technical overview, start with [`docs/HOW-IT-WORKS.md`](../docs/HOW-IT-WORKS.md) and [`docs/RESULTS.md`](../docs/RESULTS.md) at the repository root. This file is the detailed engineering reference.

The first goal is a reliable valid submission, then stronger family-specific models. Local gates can validate interface and data contracts, but realized outcomes are sealed; model score feedback comes from a CodaBench submission.

### Current implementation status

- Implemented: CLI wrapper, input loading/cutoff, output serialization, analog, ARIMA/ETS/random-walk and block-bootstrap forecasts, VAR, two-regime AR, GARCH scaling, skewed-t draws, EVT and regime-volatility transform modules, and Gaussian/Student-t/dependence modules.
- Wired into `main.py`: F1 now uses a local-drift random walk with a 252-observation window and full drift shrinkage, the best standalone setting in the three-seed F1 drift grid. Past-only model selection is the fallback if that forecast fails. GARCH scale is blended toward empirical horizon variance and bounded. Copula correlation uses recency-weighted estimates with sample-size-aware shrinkage. F2 uses ARIMA then regime-AR fallback; F3 uses ARIMA then VAR/ETS/random-walk fallback plus Student-t copula (Gaussian fallback); F4 uses the ARIMA fallback chain, GARCH scaling, then Student-t marginal tails. ARIMA and ETS simulate complete paths across horizons, and the copula and tail transform preserve those within-asset path ranks. The optional vine, regime-copula, EVT, and Markov-switching-GARCH modules are present but not selected by the dispatcher.
- LLM text layer: `llm/client.py` calls the organizer's model endpoint (stdlib only, cached by content hash so re-runs are identical); `llm/prompts.py` and `llm/signals.py` elicit and hard-bound a fixed signal vector (mean shift, variance scale, tail direction/strength, correlation scale, shock probability, basis, evidence). Only corpus-stated evidence may narrow variance, and any endpoint or validation failure falls back to neutral signals. `main.py` applies mean shift + variance scale for F2, a dependence adjustment for multi-asset F3, and variance scale + tail skew for F4; F1 stays text-free. The choice of conditioning mechanism is benchmarked in [`testing/posterior_benchmark/POSTERIOR_BENCHMARK.md`](testing/posterior_benchmark/POSTERIOR_BENCHMARK.md).
- The F1 entrypoint and organizer smoke gate passed on two practice units (one log-return card and one level card). The full Docker image and all-card contract still need validation before submission.
- A document-free, two-origin numerical backtest now covers 90 cards across all four families. It compares fixed ARIMA with local drift, tests shrunk EWMA volatility (GARCH when available), and tests Gaussian/Student-t copulas. See [`testing/layer_backtest/REPORT.md`](testing/layer_backtest/REPORT.md) for scores, exclusions, paired comparisons, and four family plots; these are local relative scores, not official CodaBench scores.

## Objective and scoring context

Track 2 evaluates probabilistic forecasts using a CRPS composite with marginal CRPS, a joint variogram component, and a tail penalty. The handover describes the composite weights as 50%, 30%, and 20%; for a single asset and single horizon, the variogram is zero and the remaining weights renormalize to 71.4% marginal CRPS and 28.6% tail penalty. Lower is better. A normalized score of 1.0 represents the organizer's text-blind baseline. A failed card receives worst-case score 4.0, so valid attempts matter.

The agent's deliverable is a Docker image that implements the `forecast` verb and is uploaded to CodaBench. It is not enough to deliver source code or a repository.

## Project structure

```text
my_agent_3/
├── README.md
├── forecast.sh
├── main.py
├── input.py
├── output.py
├── statistical_layer/
│   ├── __init__.py
│   ├── arima.py
│   ├── block_bootstrap.py
│   ├── ets.py
│   ├── random_walk.py
│   ├── var.py
│   └── markov_switching_ar.py
├── transformation_layer/
│   ├── __init__.py
│   ├── garch.py
│   ├── skewed_t.py
│   ├── markov_switching_garch.py
│   └── evt.py
├── joint_draws_layer/
│   ├── __init__.py
│   ├── gaussian_copula.py
│   ├── student_t_copula.py
│   ├── vine_copula.py
│   └── regime_copula.py
└── llm/
    ├── __init__.py
    ├── client.py
    ├── prompts.py
    ├── signals.py
    ├── mean_shift.py
    ├── variance_scale.py
    ├── tail_skew.py
    └── copula_adjustment.py
```

`forecast.sh` is the shell wrapper. `main.py` loads data, branches by card family, composes the layers, checks the final draws, and calls `output.py`. There is no separate `pipeline.py` or `guardrails.py` in this design. Add modules only when they contain reusable logic; keep family branches in `main.py` thin and readable.

## Inputs, shared contracts, and outputs

### Inputs

- **Card:** parsed `card.toml` dictionary. Important fields include `card["task"]["id"]`, `card["metadata"]["category"]`, `card["panels"]["panel_ids"]`, `card["targets"]["asset_ids"]`, `card["targets"]["horizons"]`, and `card["targets"]["target_type"]`.
- **Numerical panel:** pandas DataFrame with `date`, `asset`, `value`, and `panel_id` columns. Card panels are stored at the unit root in the practice units, not necessarily in a `panels/` subdirectory.
- **Frozen text:** dated documents provided with the unit. The agent may call the configured model endpoint, but must not fetch new text or use live internet data.
- **As-of date:** string in `YYYY-MM-DD` format. No numeric observation or text document after this cutoff may influence the forecast.

### Internal draw format

Statistical, transformation, and joint layers use a NumPy array with shape `[n_draws, n_assets, n_horizons]`. Asset and horizon axes preserve the order in `card["targets"]`. For joint forecasts, values at the same draw index form one scenario across assets. Use fixed, reproducible random seeds.

### Required output

Write exactly these three regular files to the requested output directory:

1. `forecast.parquet`, columns `draw:int32`, `asset:string`, `horizon:int32`, `value:float64`.
2. `forecast_meta.json`, including `unit_id` equal to `card["task"]["id"]`, `asof`, `representation: "samples"`, `asset_ids`, `horizons`, `n_draws`, and `target`.
3. `forecast_rationale.md`, non-blank free text; it is not scored.

Use 200 draws minimum, 500 as a practical default, and 1,000 or more for F4. Keep the output directory free of extra files, logs, caches, and `.DS_Store` artifacts.

## File responsibilities and interfaces

### Entrypoint and I/O

| File | Inputs | Outputs and role |
|---|---|---|
| `forecast.sh` | Harness `forecast` verb and its CLI arguments (confirm exact argument names from the repo's harness). | Invokes `main.py` and propagates its exit code. No model logic. |
| `main.py` | CLI arguments for panel/input path, text path, `asof`, and output path. | Calls `input.py`, identifies family, composes model layers, checks draws, and calls `output.py`. |
| `input.py` | Unit/panel path, text path, and `asof`. | Returns card dictionary, cutoff-filtered numerical DataFrame, and eligible dated documents. Reads panel ID from `card["panels"]["panel_ids"][0]`. |
| `output.py` | Draws, ordered assets/horizons, card, `asof`, draw count, and rationale/signals. | Writes the three required forecast artifacts and no others. |

### Layer 1 — Statistical forecasts

Models receive the filtered panel, card, `asof`, draw count, and seed; they return `[n_draws, n_assets, n_horizons]` draws.

| File | Inputs | Outputs and role |
|---|---|---|
| `statistical_layer/__init__.py` | None. | Package exports, if useful. |
| `statistical_layer/arima.py` | Numerical panel and forecast parameters. | Per-asset state-space ARIMA simulations; each draw is a complete path shared by all requested horizons. |
| `statistical_layer/analog.py` | Panel, card, `asof`, draw count, seed, analogue count. | Finds similar historical states using trend, volatility, and (for levels) relative-level features; samples complete observed future paths. Candidate model in F1 rolling validation; requires at least 20 complete donors. |
| `statistical_layer/model_selection.py` | Panel, card, `asof`, draw count, seed. | F1 safety fallback: backtests target-aware ARIMA variants, residual-block ARIMA, ETS, analog, and local-drift random walk at up to four past-only origins, then fits a conservative mixture through `asof`. |
| `statistical_layer/block_bootstrap.py` | Panel, card, `asof`, draw count, seed, block length. | Resamples contiguous transformed increments; fallback for F1 log-return targets, not used for level targets. |
| `statistical_layer/ets.py` | Same numerical inputs. | Exponential smoothing paths with bootstrapped residual blocks; fallback when ARIMA fitting fails. |
| `statistical_layer/random_walk.py` | Same numerical inputs. | Robust final fallback based on the latest level or historical increments. |
| `statistical_layer/var.py` | Aligned multi-asset panel and forecast parameters. | Joint VAR forecast paths/draws, capturing cross-asset effects in means; mainly F3. |
| `statistical_layer/markov_switching_ar.py` | Target history and forecast parameters. | Regime-dependent autoregressive draws; mainly F2 when enough data supports a stable fit. |

Fallback order should be explicit and robust: try the selected primary model, then ETS, then random walk. A failed optional model must not prevent a valid forecast.

### Layer 2 — Distribution transformations

Transformation modules reshape marginal volatility or tails. They preserve draw shape, target ordering, and reproducibility. If a transform fails, `main.py` should retain the valid incoming draws.

| File | Inputs | Outputs and role |
|---|---|---|
| `transformation_layer/__init__.py` | None. | Package exports, if useful. |
| `transformation_layer/garch.py` | History/residuals, base draws, card, seed, shrinkage/bounds. | Shrinks conditional volatility toward empirical horizon scale and bounds the resulting scale changes; uses EWMA if GARCH fitting is unavailable. |
| `transformation_layer/skewed_t.py` | Base draws or standardized innovations, scale and distribution parameters, seed. | Fat-tailed and asymmetric marginal draws; mainly F4. |
| `transformation_layer/markov_switching_garch.py` | History/residuals and base draws. | Regime-dependent volatility draws; later F4 enhancement. |
| `transformation_layer/evt.py` | Historical residuals and base draws. | Extreme-value tail adjustment; optional F4 refinement when enough tail data is available. |

GARCH plus skewed-t is the primary planned transformation for F4. It can also help F3's tail component. Markov-switching GARCH, EGARCH/GJR-GARCH variants, and peaks-over-threshold EVT are later options rather than first-build requirements.

### Layer 3 — Joint draws

Joint modules receive marginal draws and aligned historical observations or dependence estimates. They return the same draw shape and make each draw index a coherent multi-asset scenario. For a single asset, bypass this layer and pass the draws through.

| File | Inputs | Outputs and role |
|---|---|---|
| `joint_draws_layer/__init__.py` | Aligned panel, target assets/type, estimator settings. | Recency-weighted correlation with adaptive identity shrinkage and PSD projection. |
| `joint_draws_layer/gaussian_copula.py` | Marginal draws, aligned history/dependence matrix, seed. | Reassigns whole asset paths by terminal-horizon Gaussian copula ranks, preserving within-path horizon relationships. |
| `joint_draws_layer/student_t_copula.py` | Marginal draws, dependence estimates, degrees-of-freedom setting, seed. | Adds terminal cross-asset tail dependence while keeping each asset path intact. |
| `joint_draws_layer/vine_copula.py` | Marginal draws, aligned history, seed and settings. | Flexible pairwise dependence for larger or heterogeneous panels; optional later. |
| `joint_draws_layer/regime_copula.py` | Marginal draws, regime probabilities/labels, per-regime dependence, seed. | Regime-specific dependence; optional refinement. |

F3 requires dependent multi-asset draws. Start with a Gaussian copula and then evaluate a Student-t copula. F4 uses Student-t dependence only when there are multiple target assets. Clayton copula is a possible lower-tail-specific alternative; a vine copula is most relevant for larger mixed panels.

### Layer 4 — Text extraction and conditioning

The text layer reads only eligible documents and card context. The LLM returns bounded structured signals, not forecast values. Conditioning modules receive draws plus validated signals and return draws of the same shape. If the endpoint fails or the response is invalid, use neutral signals and preserve a valid numerical forecast.

| File | Inputs | Outputs and role |
|---|---|---|
| `llm/__init__.py` | None. | Package exports, if useful. |
| `llm/client.py` | Messages and `MODEL_ENDPOINT` configuration. | Raw endpoint response; handles timeouts/errors and does not expose credentials. |
| `llm/prompts.py` | Eligible dated documents, family/target context, `asof`, signal schema. | Prompt/messages requesting structured signals based only on provided text. |
| `llm/signals.py` | Raw response and expected schema. | Validated signal object with bounded numeric/categorical fields, or neutral fallback. |
| `llm/mean_shift.py` | Draws and a validated directional/mean signal. | Same-shape shifted draws; essential conditioning for F2. |
| `llm/variance_scale.py` | Draws and bounded uncertainty multiplier. | Same-shape draws with adjusted dispersion; F2 and F4. |
| `llm/tail_skew.py` | Draws and validated tail direction/strength. | Same-shape draws with additional left/right tail mass; mainly F4. |
| `llm/copula_adjustment.py` | Joint draws and cross-asset dependence signal. | Same-shape joint draws with adjusted dependence; optional F3 refinement. |

Define the signal schema before implementing prompts. Useful candidate fields include direction/mean bias, regime-change probability, uncertainty multiplier, tail direction, shock probability, and a brief rationale. The API client must use the permitted model endpoint only; no web search, retrieval, or live data fetch.

## Family-specific model plan

“Essential” means important for a strong result in that family. “Optional” means a later refinement. The build order below is staged so the first image can ship before every family-specific improvement is complete.

### F1 — Baseline forecasting

F1 usually tests one asset at one horizon. Its single-cell score emphasizes marginal CRPS (effective 71.4%) and tail penalty (28.6%); joint variogram is zero by construction.

- **Layer 1:** Primary forecast is local-drift random walk (252-observation window, full shrinkage), selected by the multi-seed standalone F1 backtest. Past-only ARIMA/ETS/analog model selection remains as a fallback if the primary fails. See the drift grid and selector follow-up in `testing/README.md`.
- **Layer 2:** Usually skip. GARCH/skewed-t is optional if leaderboard feedback points to tail weakness. EVT is generally not useful with little single-series tail data.
- **Layer 3:** Skip for single-cell cards.
- **Layer 4:** Optional text extraction, mean shift, and variance scaling where corpus signals are relevant.
- **Optional model:** Theta as a cheap ensemble member.

### F2 — Regime shift with transfer

F2 tests whether text can reveal a regime change and inform forecasts, sometimes for an asset absent from the input panel.

- **Layer 1:** ARIMA, ETS, random walk; Markov-switching AR is the key family-specific extension. TAR/STAR are optional alternatives.
- **Layer 2:** GARCH/skewed-t or Markov-switching GARCH are optional volatility extensions.
- **Layer 3:** Add a copula only for multi-asset cards; regime-conditional dependence is optional.
- **Layer 4:** LLM signal extraction, mean shift, and variance scaling are essential. Tail skew is optional for crash-like regimes.

### F3 — Cross-asset reasoning

F3 emphasizes joint dependence; the variogram component carries 30% of the composite. Independent marginal sampling is not sufficient.

- **Layer 1:** ARIMA/ETS/random walk for marginals; VAR is important because it captures cross-asset mean spillovers that a copula alone does not.
- **Layer 2:** GARCH/skewed-t is recommended for the tail component. DCC-GARCH is an optional alternative when modeling time-varying correlation directly.
- **Layer 3:** Gaussian copula first, Student-t copula next. Vine copula is optional for heterogeneous or larger panels; regime copula is optional when regime changes affect dependence.
- **Layer 4:** Optional cross-asset text themes and copula parameter adjustment.

### F4 — Tail / shock from text

F4 emphasizes extreme outcomes, with a 20% tail penalty and at least 1,000 draws recommended.

- **Layer 1:** ARIMA, ETS, random walk establish the forecast center.
- **Layer 2:** GARCH with skewed-t is essential. Markov-switching GARCH is a useful later extension; EGARCH/GJR-GARCH and EVT are optional refinements.
- **Layer 3:** Student-t copula is relevant for multi-asset joint tails. Gaussian copula lacks tail dependence; Clayton is an optional lower-tail-specific alternative.
- **Layer 4:** LLM extraction, text-directed tail skew, and variance scaling are essential. Use at least 1,000 draws for tail stability.

### Family summary

| Layer | F1 | F2 | F3 | F4 |
|---|---|---|---|---|
| **L1 — Statistical** | ARIMA, ETS, random walk | ARIMA, ETS, Markov-switching AR, random walk | ARIMA, ETS, VAR, random walk | ARIMA, ETS, random walk |
| **L2 — Transform** | Usually skip; GARCH optional | Optional GARCH | GARCH/skewed-t recommended | GARCH/skewed-t; later MS-GARCH/EVT |
| **L3 — Joint draws** | Skip single-cell | Copula if multi-asset | Gaussian then Student-t; vine optional | Student-t if multi-asset |
| **L4 — Text** | Optional | LLM + mean shift + variance | Optional cross-asset signal | LLM + variance scale + tail skew |

## Build priorities

The initial sequence follows the handover's ship-after-1–7 milestone. Later priorities can change based on leaderboard results.

| Rank | Component | Layer | Main benefit |
|---:|---|---|---|
| 1 | ARIMA | L1 | Core numerical forecast for all families |
| 2 | ETS fallback | L1 | Robustness when ARIMA fails |
| 3 | Random-walk fallback | L1 | Final never-fail numerical path |
| 4 | Gaussian copula | L3 | F3 cross-asset dependence |
| 5 | Student-t copula | L3 | Joint tail dependence for F3/multi-asset F4 |
| 6 | GARCH + skewed-t | L2 | F4 volatility and fat asymmetric tails |
| 7 | LLM extractor + mean-shift combiner | L4 | F2 text-driven direction/regime adjustment |
| 8 | Variance scaling | L4 | Text-driven uncertainty adjustment for F2/F4 |
| 9 | Tail skew | L4 | Text-directed asymmetric tail for F4 |
| 10 | VAR | L1 | Cross-asset mean effects for F3 |
| 11 | Markov-switching AR | L1 | Regime dynamics for F2 |
| 12 | Markov-switching GARCH | L2 | Regime-dependent volatility for F4 |
| 13 | Vine copula | L3 | Flexible dependence for larger F3 panels |
| 14 | EVT | L2 | Optional F4 extreme-tail refinement |

Ship a first image after components 1–7 pass the interface and local gate checks. Components 8–14 are refinements; the README's family tables describe the fuller target architecture.

## Practice units and local workflow

Practice cards identified in the handover:

| Unit | Family | Targets |
|---|---|---|
| `units/t2-F1-ai-mom-2024` | F1 | MOM, log return, one horizon |
| `units/t2-F2-abenomics-2012` | F2 | Level, one asset/horizon |
| `units/t2-F3-bear-flattener-2022` | F3 | Four assets, two horizons |
| `units/t2-F4-aud-gfc-2008` | F4 | Level, one asset/horizon |
| `units/t2-EXAMPLE-ust-curve-1m` | Exemplar | Four assets, one horizon |

The earlier agent used commands in this form; adapt paths and output location to the final `forecast.sh` contract:

```bash
REPO=$(pwd) bash my-agent-2/forecast.sh \
  --panels units/t2-F1-ai-mom-2024 \
  --text units/t2-F1-ai-mom-2024/text \
  --asof 2024-05-31 \
  --out out/forecast.parquet

qfbench2-smoke units/t2-F1-ai-mom-2024/ my-agent-2/out/ --track forecasting

python3 scoring/scoring.py score \
  --card units/t2-F1-ai-mom-2024/card.toml \
  --forecast my-agent-2/out/forecast.parquet
```

This is a reference workflow from the existing agent, not a claim that `my_agent_3` already implements the same CLI. Confirm the actual harness arguments and Docker contract before finalizing the wrapper. The local scorer can report that a card was not scored because outcomes are sealed; actual score feedback is on CodaBench.

## Runtime and project constraints

- Use the competition's `qfbench2-common` toolkit rather than reimplementing its scoring or gate math.
- Do not modify organizer-owned `baselines/` or `scoring/` files.
- Do not add realized outcomes, answer keys, or other private evaluation data to the public repository.
- No pretrained neural model unless the organizers approve it. The planned models here are fitted statistical models.
- No internet data fetching at runtime. Text comes from the supplied corpus; model reasoning uses the configured `MODEL_ENDPOINT`.
- Pin random seeds and package all dependencies in Docker. The handover lists Python 3.14, a 16 CPU / 128 GB RAM resource budget, and 1,800 seconds per unit.
- The handover notes installed packages including `statsmodels`, `pmdarima`, `arch`, `copulae`, `scikit-learn`, `pandas`, `pyarrow`, `numpy`, `scipy`, `qfbench2-common`, and `qfbench2_track_forecasting`; verify the Docker image includes the versions actually needed.
- The working branch at the time of inspection was `agent-dev`; preserve the active branch unless intentionally instructed otherwise.
- Only submit to CodaBench after the Docker image has been built and smoke-checked. Do not open pull requests to the upstream organizer repository; use the private fork if a PR is needed.

## Important failure checks

- Output directory must contain exactly the three required files.
- Metadata `unit_id` must equal `card["task"]["id"]`; do not use `card_id` in its place.
- `representation` must be `samples`; `n_draws` must satisfy the card minimum and scorer bounds (the handover gives 200–2,000).
- `asof` must match the card, and no panel or text data after that date may be used.
- Parquet must cover the expected asset/horizon grid with finite values and the exact expected shape.
- F3 rows must preserve joint draw indices across all assets and horizons.
- All random sampling should be reproducible from pinned seeds.

## Sources in this project

- `../docs/CONCEPTS.md` explains probabilistic forecasts, text cutoffs, and scoring concepts.
- `../docs/CATEGORIES.md` describes the four card families and their emphases.
- `../AGENTS.md` contains repository rules, including public/private data safety, toolkit use, family strings, horizon preservation, and text cutoff requirements.
- `ARCHITECTURE.md` contains the expanded per-file input/output contracts. This README is the consolidated entry point; update it and that file together when the architecture changes.
