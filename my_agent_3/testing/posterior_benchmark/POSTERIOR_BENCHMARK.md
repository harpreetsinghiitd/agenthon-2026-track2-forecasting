# Posterior-generation mechanism benchmark

**Question:** given the elicited signal vector (the LLM's bounded, unitless
testimony), which mathematical mechanism for converting it into posterior
draws scores best — and how gracefully does each degrade when the signal is
noisy or wrong?

**Method.** Leakage-safe rolling origins over 90 valid cards (the same folds
as `layer_backtest`): each origin fits the family's text-blind pipeline on
panel data through t1 and is scored on the realized outcome in (t1, t2] with
the card's own composite weights (50% CRPS / 30% variogram / 20% tail). For
every (card, origin) the harness derives an **oracle signal** — the realized
outcome expressed in the production signal's own units and bounds, i.e. the
best signal the vocabulary could ever carry — then degrades it (±25% / ±50%
multiplicative noise, and an adversarial inversion: every knob wrong) and
applies **17 mechanisms** from `posterior_mechanisms.py`:

| Family | Mechanisms |
|---|---|
| baseline | `none` |
| parametric (production) | `shift`, `scale`, `tail`, `shift_then_scale`, `scale_then_shift`, `affine_mean_centred`, `full_param`, `full_param_joint` |
| tilting (entropy pooling) | `tilt_loc`, `tilt_loc_disp`, `tilt_loc_disp_ess` (ESS floor 100) |
| mixture | `mixture2`, `mixture3`, `mixture3_scale` |
| mapping | `quantile_warp`, `shock_inject` |

500 draws, 2 origins per card, 15,300 scored cells. Ratios are composite loss
relative to the **unconditioned prior** (1.000 = no effect, lower = better).
The oracle uses the realized outcome *by design* — this is ceiling analysis
for mechanism comparison, never a submission path.

Rerun: `PYTHONPATH=. python3 my_agent_3/testing/run_posterior_benchmark.py
--origins 2 --draws 500` (priors cached under `cache/`, gitignored).

## Headline results (mean composite ratio vs prior, all 90 cards)

| Mechanism | oracle | noisy25 | noisy50 | adversarial | win% (oracle) |
|---|---:|---:|---:|---:|---:|
| shift | **0.822** | 0.831 | **0.849** | 1.347 | **95.6** |
| shock_inject | 0.857 | 0.857 | 0.874 | 1.281 | 73.9 |
| mixture3 | 0.863 | 0.871 | 0.883 | **1.189** | 93.3 |
| mixture2 | 0.864 | 0.871 | 0.884 | 1.214 | 95.0 |
| affine_mean_centred | 0.886 | **0.821** | 0.884 | 1.608 | 89.4 |
| shift_then_scale | 0.887 | 0.822 | 0.884 | 1.607 | 88.3 |
| full_param | 0.887 | 0.825 | 0.884 | 1.628 | 88.9 |
| full_param_joint | 0.890 | 0.829 | 0.895 | 1.628 | 88.9 |
| scale_then_shift | 0.907 | 0.832 | 0.892 | 1.684 | 88.3 |
| mixture3_scale | 0.943 | 0.870 | 0.923 | 1.498 | 87.2 |
| tail | 0.980 | 0.981 | 0.987 | 1.011 | 20.0 |
| quantile_warp | 1.038 | 1.010 | 1.025 | 1.119 | 53.9 |
| scale | 1.104 | 1.016 | 1.049 | 1.324 | 73.3 |
| tilt_loc_disp_ess | 1.281 | 1.998 | 2.658 | 6.218 | 85.0 |
| tilt_loc_disp | 1.340 | 2.213 | 2.957 | 6.049 | 83.3 |
| tilt_loc | 3.214 | 3.243 | 2.986 | 3.284 | 76.1 |

Neutral signals produce exactly 1.000 for every mechanism — the production
no-op contract is verified.

## Per-family (mean composite ratio: oracle / noisy25 / adversarial)

| Mechanism | F1 | F2 | F3 | F4 |
|---|---|---|---|---|
| shift | **0.870** / 0.878 / 1.273 | 0.808 / 0.820 / 1.364 | 0.878 / 0.885 / 1.233 | 0.766 / 0.775 / 1.455 |
| full_param | 1.484 / 1.080 / 2.116 | **0.664** / 0.724 / 1.479 | **0.845** / 0.869 / 1.326 | 0.656 / 0.695 / 1.569 |
| affine_mean_centred | 1.492 / 1.071 / 2.079 | 0.665 / 0.725 / 1.464 | 0.854 / 0.874 / 1.319 | **0.642** / 0.686 / 1.552 |
| mixture3 | 0.902 / 0.907 / 1.151 | 0.843 / 0.858 / 1.204 | 0.916 / 0.919 / 1.111 | 0.820 / 0.827 / 1.251 |
| tilt_loc | 10.96 / 11.05 / 9.58 | 0.812 / 0.826 / 1.344 | 0.965 / 0.970 / 1.302 | 0.830 / 0.837 / 1.423 |
| tail | 0.993 / 0.994 / 1.008 | 0.969 / 0.972 / 1.015 | 0.985 / 0.986 / 1.001 | 0.978 / 0.977 / 1.016 |

## Findings

1. **The parametric family wins; entropy pooling loses.** The proposed replacement
   idea (Meucci-style reweighting) is *dominated* by the shipped parametric
   conditioners at these draw counts. Two measured causes: (a) the
   **resampling tax** — tilting reweights then resamples, and the resample's
   Monte-Carlo noise is pure loss wherever the prior is already good; the 14
   catastrophic cells (ratio up to 300×) are all F1 cards whose local-drift
   prior was near-perfect, so any movement was pure noise, amplified by tiny
   denominators; (b) **narrowing under wrong signals** — the dispersion view
   inverted under the adversarial signal (6.0–6.2× damage); the ESS floor
   limits movement, not damage. Medians tell the same story more gently
   (tilt medians 0.70–0.85 on F2–F4), but the tails are disqualifying for an
   untrusted signal source.

2. **Drift is the load-bearing knob.** `shift` alone wins the oracle mean
   (0.822) and the highest win rate (95.6%), and it is the most
   noise-tolerant mechanism measured (0.82 → 0.85 from perfect to 50% noise).
   The elicitation layer's mean_shift_sigma field carries most of the value
   in the vocabulary.

3. **Combining shift+scale is best when the signal is roughly right.**
   `shift_then_scale` / `affine_mean_centred` / `full_param` post the best
   realistic-case numbers (0.82 at noisy25) and the best medians (0.674 — a
   third off the composite on the typical card), but pay for it in the tail:
   1.61–1.68 under adversarial signals, and they actively damage F1 (1.48 on
   oracle — the coverage-derived variance knob mis-fires on a near-perfect
   prior).

4. **Mixtures are the graceful-degradation champions.** `mixture3` gives up
   ~4–8% of ceiling vs the parametric combos but takes the least damage when
   the signal is wrong (1.19 vs 1.63 adversarial) — the hedge branch caps the
   cost of a confidently wrong corpus read. If signal trust is uncertain,
   mixtures are the risk-adjusted choice.

5. **The tail knob is nearly free insurance**: +2% ceiling, +1.1% worst case.
   Keep it.

6. **Noise in the elicited magnitudes barely matters; direction errors are
   everything.** ±25% multiplicative noise on every knob costs almost nothing
   (most mechanisms: oracle ≈ noisy25 ≈ noisy50). The binding constraint is
   the signal being *pointed the wrong way* — which is exactly what the
   bounds, the basis discipline, and the widen-don't-average prompt rules
   defend against.

## Production recommendations (my_agent_3 wiring)

| Family | Current wiring | Benchmark verdict |
|---|---|---|
| F1 | text-blind | **Confirmed correct.** Every conditioning mechanism hurts the already-excellent local-drift prior; tilt catastrophically so. |
| F2 | mean shift + variance scale | **Validated** — best noisy-case family score (0.72). Keep. |
| F3 | dependence blend only | **Weakest wiring.** Shift and scale contribute more than dependence (0.87/0.85 vs oracle); consider adding mean shift to F3. |
| F4 | variance scale + tail skew | **Missing the load-bearing knob.** F4 gains most from shift (0.766 alone; affine combos 0.64) — add mean shift to the F4 conditioner. |
| all | parametric conditioners | **Keep.** Entropy pooling rejected: dominated on ceiling, catastrophic on wrong signals. |
| optional | — | If corpus-trust is low on event cards, `mixture3` is the robustness-optimal alternative (implementable as a scenario extension of the signal schema). |

The oracle numbers are ceilings, not expectations: a real LLM will land
between neutral and oracle. The decision rule the curves imply: with a
roughly-right signal the parametric combos dominate; with an often-wrong one
only `shift`, `tail`, and the mixtures stay net-positive.

*Local historical comparison on public-dev panels; not official CodaBench
scores. Oracle signals use realized outcomes and are for mechanism ranking
only.*
