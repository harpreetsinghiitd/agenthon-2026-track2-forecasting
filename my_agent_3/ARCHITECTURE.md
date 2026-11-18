# Agent Architecture and File Contracts

## Executive summary (read this first)

This document describes the files for `my_agent_3` and what data each file receives and returns. The agent loads the card, numerical panel, and frozen text corpus; composes the forecasting layers in `main.py`; validates the resulting draws; and writes the three required forecast files. Pipeline composition stays in `main.py`. There is no separate `guardrails.py`. The numerical layers and the `llm/` text layer are implemented; optional modules (vine and regime copulas, EVT, Markov-switching GARCH) are available but not selected by the dispatcher.

## Shared data contracts

The card is the parsed `card.toml` dictionary. Relevant fields include `task.id`, `metadata.category`, `panels.panel_ids`, `targets.asset_ids`, `targets.horizons`, and `targets.target_type`.

The numeric panel is a pandas DataFrame with columns `date`, `asset`, `value`, and `panel_id`. The loader must retain only observations on or before `asof`. The frozen text corpus is a collection of dated documents, also restricted to dates on or before `asof`.

All statistical and distribution layers use forecast draws shaped `[n_draws, n_assets, n_horizons]`. Asset and horizon axes follow the order in `card["targets"]`; draws at the same draw index represent one joint scenario across assets. Use deterministic random seeds. The output writer serializes the draws in the organizer's required long-form parquet schema.

## Project tree

```text
my_agent_3/
├── forecast.sh
├── main.py
├── input.py
├── output.py
├── statistical_layer/
│   ├── __init__.py
│   ├── arima.py
│   ├── analog.py
│   ├── model_selection.py
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

## Entrypoint and data I/O

| File | Inputs | Outputs / responsibility |
|---|---|---|
| `forecast.sh` | The competition's `forecast` verb and its arguments: panel/input path, text path, `asof`, and output path (using the exact harness CLI contract). | Starts `main.py` with the same arguments and propagates its exit status. It should not implement forecasting logic. |
| `main.py` | Parsed command-line arguments. | Calls `input.py`; selects a family pipeline from `card["metadata"]["category"]`; invokes model layers; validates shape and finite values; calls `output.py`. It owns the family-specific layer ordering. |
| `input.py` | Input paths and `asof`. | Returns `(card, panel, documents)`, where `card` is a dictionary, `panel` is a filtered DataFrame, and `documents` is a list of dated document records. It resolves the panel file from the card's panel ID and applies the hard cutoff. |
| `output.py` | Draw array, ordered asset IDs and horizons, card, `asof`, draw count, and optional rationale/signals. | Writes exactly `forecast.parquet`, `forecast_meta.json`, and non-blank `forecast_rationale.md`. The parquet columns are `draw:int32`, `asset:string`, `horizon:int32`, `value:float64`; metadata uses `unit_id = card["task"]["id"]` and `representation = "samples"`. |

`main.py` should keep the pipeline branches straightforward: F1 uses numerical marginals; F2 adds text conditioning; F3 couples numerical marginals; F4 applies tail transformation, dependence when applicable, and text conditioning. Single-asset cards can bypass the joint-draw layer.

## Layer 1 — Statistical forecasts

Each statistical model receives the filtered numeric panel, card, `asof`, draw count, and seed. It returns draws with shape `[n_draws, n_assets, n_horizons]` in target order. A model may expose a `fit_and_sample(...)` function. `main.py` or a small helper in the layer selects a model and falls back if fitting fails.

| File | Inputs | Outputs / responsibility |
|---|---|---|
| `statistical_layer/__init__.py` | None. | Makes the folder importable; may re-export the shared `run` interface. |
| `statistical_layer/analog.py` | Panel, card, `asof`, draw count, seed, analogue count. | Compares the latest trend/volatility state with historical states that have fully observed future paths; samples whole donor paths for F1. Requires at least 20 complete donors. |
| `statistical_layer/model_selection.py` | Panel, card, `asof`, draw count, seed. | F1 fallback: scores target-aware ARIMA variants, residual-block ARIMA, ETS, analog paths, and local-drift random walk at up to four past-only inner origins. Adds only challengers that clear the gain and consistency thresholds, then samples a conservative mixture fitted through `asof`. |
| `statistical_layer/arima.py` | Panel, card, `asof`, draw count, seed. | Simulates full state-space ARIMA paths per asset, then reads all requested horizons from each shared path. For log returns, output cumulative log returns; for levels, output the simulated level at each horizon. |
| `statistical_layer/block_bootstrap.py` | Panel, card, `asof`, draw count, seed, block length. | Resamples contiguous transformed increments; fallback for F1 log-return targets if the rolling selector's candidates fail. |
| `statistical_layer/ets.py` | Same numerical inputs. | Exponential smoothing paths with resampled contiguous residual blocks; fallback when ARIMA cannot fit or converge. |
| `statistical_layer/random_walk.py` | Same numerical inputs. | Random-walk or historical-increment draws as the final robust fallback. |
| `statistical_layer/var.py` | Aligned multivariate panel, card, `asof`, draw count, seed. | Joint VAR forecast paths/draws that model cross-asset effects in conditional means; primarily useful for F3. |
| `statistical_layer/markov_switching_ar.py` | A target series, card settings, `asof`, draw count, seed. | Regime-dependent autoregressive draws; candidate for F2 when sufficient history and a stable fit are available. |

## Layer 2 — Marginal distribution transformation

These functions receive draws and any fitted residual/volatility information they need, and return draws of the same shape. They should not change asset or horizon ordering. Keep this layer optional by family so a failed transform can fall back to the incoming draws.

| File | Inputs | Outputs / responsibility |
|---|---|---|
| `transformation_layer/__init__.py` | None. | Makes the folder importable; may re-export transform functions. |
| `transformation_layer/garch.py` | Historical target/residual series, base draws, card, seed, shrinkage and scale bounds. | GARCH volatility blended toward the empirical horizon variance; scale adjustments are bounded to reduce unstable fits. Mainly for F4 and potentially F3. |
| `transformation_layer/skewed_t.py` | Base draws or standardized innovations, estimated scale, skew/tail parameters, seed. | Same-shape draws with fat-tailed and asymmetric marginal distributions. Its monotone rank transform preserves dependence among horizons and assets. Parameters must be estimated from permitted numerical history, not future outcomes. |
| `transformation_layer/markov_switching_garch.py` | Historical series/residuals, base draws, card, seed. | Same-shape draws with regime-dependent volatility; later F4 refinement. |
| `transformation_layer/evt.py` | Historical residuals and base draws. | Tail-adjusted draws using an extreme-value model; optional later refinement when there is enough tail data. |

## Layer 3 — Joint draws

Joint-draw functions receive marginal draws and historical aligned observations (or fitted dependence parameters), plus card metadata and a seed. They return the same shape. Each draw row must remain a coherent scenario across assets and horizons.

| File | Inputs | Outputs / responsibility |
|---|---|---|
| `joint_draws_layer/__init__.py` | Aligned panel, target assets/type, optional half-life and shrinkage. | Estimates a recency-weighted correlation matrix with dimension/sample-aware shrinkage and PSD projection. |
| `joint_draws_layer/gaussian_copula.py` | Marginal draws, aligned history or dependence matrix, seed, optional estimator settings. | Reassigns complete asset paths by terminal-horizon Gaussian copula ranks, preserving each asset's within-path horizon sequence. |
| `joint_draws_layer/student_t_copula.py` | Marginal draws, aligned history or correlation and degrees-of-freedom estimates, seed, optional estimator settings. | Adds terminal cross-asset tail dependence while moving each asset's full path as one scenario. |
| `joint_draws_layer/vine_copula.py` | Marginal draws, historical aligned data, seed, and model settings. | Flexible pairwise dependence for larger or heterogeneous panels; optional after simpler copulas. |
| `joint_draws_layer/regime_copula.py` | Marginal draws, regime probabilities/labels, dependence estimates per regime, seed. | Same-shape draws whose dependence varies by regime; optional refinement for regime-aware cards. |

For one asset, the joint layer should return its input unchanged. For small samples or failed dependence fitting, use a documented safe fallback such as the Gaussian copula or unchanged draws.

## Layer 4 — Text extraction and conditioning

The LLM components receive only the provided frozen documents and card context, with the `asof` cutoff. The extractor returns a validated structured signal object. Conditioning functions receive draws plus that object and return same-shape draws. The LLM client must use the configured `MODEL_ENDPOINT`; it must not fetch web content or use unapproved tools.

Suggested signal fields include direction/mean bias, regime-change probability, uncertainty multiplier, tail direction, shock probability, and a short rationale. Define and validate the exact schema before implementing the prompt. Treat missing or invalid signals as neutral adjustments.

| File | Inputs | Outputs / responsibility |
|---|---|---|
| `llm/__init__.py` | None. | Makes the folder importable; may expose `extract_and_condition`. |
| `llm/client.py` | Prompt/messages, configured endpoint and credentials from environment. | Raw model response. Handles timeout and API errors; never embeds credentials in source or output. |
| `llm/prompts.py` | Dated documents, card family/targets, `asof`, signal schema. | Prompt/messages requesting constrained, structured signals based only on eligible corpus documents. |
| `llm/signals.py` | Raw model response and expected schema. | Validated signal dictionary with bounded numeric fields and allowed categorical values, or a neutral fallback signal. |
| `llm/mean_shift.py` | Draws and directional/standardized mean signal. | Same-shape draws shifted consistently with the extracted mean signal; mainly F2. |
| `llm/variance_scale.py` | Draws and bounded uncertainty multiplier. | Same-shape draws with adjusted dispersion; mainly F2 and F4. |
| `llm/tail_skew.py` | Draws and validated tail direction/strength. | Same-shape draws with additional left- or right-tail weight; mainly F4. |
| `llm/copula_adjustment.py` | Joint draws and validated cross-asset dependence signal. | Same-shape joint draws with adjusted dependence; optional F3 refinement. |

## Family execution order

| Family | Layer order in `main.py` | Initial draw guidance |
|---|---|---|
| F1 | Local-drift random walk (252-observation window, full shrinkage); past-only rolling model-selection mixture is the fallback. | At least 200; use 500 as a practical default. |
| F2 | Statistical forecast (Markov-switching AR when ready); extract text signals; mean shift; variance scale. | At least 200; use 500 as a practical default. |
| F3 | Statistical forecast (VAR when ready); marginal transformation when enabled; Gaussian copula, then Student-t copula as the upgrade. | At least 200; recommend 500 or more. |
| F4 | Statistical forecast; GARCH/skewed-t transformation; Student-t copula if multi-asset; extract text signals; variance scale and tail skew. | At least 200; recommend 1,000 or more. |

## Implementation order

1. Implement the entrypoint, input/output contracts, and deterministic draw interface.
2. Implement ARIMA, ETS, and random-walk fallback, then run practice cards through the local gates.
3. Add Gaussian copula and Student-t copula for joint draws.
4. Add GARCH and skewed-t transformation.
5. Add the LLM client, structured signal parsing, and mean-shift conditioning for F2.
6. Add variance scaling and tail skew for F4; use at least 1,000 draws there.
7. Add VAR and Markov-switching AR where practice behavior and leaderboard feedback justify them; consider vine/regime copulas and EVT later.
8. Build and smoke-check the Docker image before uploading it to CodaBench.

The output contract, as-of cutoff, and exact required file set take precedence over optional model complexity. Statistical failures should trigger a safe fallback so a difficult fit does not prevent a valid forecast from being produced.
