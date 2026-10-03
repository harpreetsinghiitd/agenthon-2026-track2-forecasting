# my-agent-2 — Track 2 forecasting agent

A reasoning-augmented time-series forecaster for the QFBench 2.0 Track 2 card
families (F1–F4). It reads a frozen text corpus and a numeric panel as of a fixed
date, and emits a probabilistic forecast as a set of draws.

## The one rule

> The LLM reads the frozen corpus and writes code. It does not write numbers that
> go into `forecast.parquet`.

Every number in a submission is traceable to deterministic computation: a
statistical model, a seeded RNG, and text-derived features. The LLM's output is a
*vector*; a deterministic function in `harness/bridge/` maps that vector to
forecast parameters. `harness/tools/write.py` is the only module that touches the
output directory.

## Layout

```
my-agent-2/
├── agent/            the non-deterministic policy (Phases 2, 3, 6)
├── harness/
│   ├── tools/        card.py, panel.py, write.py        (Phase 1)
│   │                 corpus.py, stats.py, simulate.py   (Phases 2–5)
│   ├── bridge/       vector_to_adjustment, scenario_mix (Phase 3)
│   ├── llm/          client, cache, extract             (Phase 3)
│   └── validate/     preflight.py                       (Phase 1)
├── tests/            pytest suite
├── requirements.txt
└── README.md
```

Phase 1 delivers `harness/tools/card.py`, `harness/tools/panel.py`,
`harness/tools/write.py` and `harness/validate/preflight.py`, with their tests.
Phase 2 adds `harness/tools/stats.py` (block bootstrap + ARIMA),
`harness/tools/simulate.py` (deterministic mixture) and `agent/main.py`
(entrypoint + per-asset fallback). `forecast.py` is the CLI shim the harness
calls; Phase 3 adds the text layer (`harness/llm`, `harness/bridge`).

## How to run

```bash
python forecast.py --panels <unit_dir> --text <unit_dir>/text \
  --asof <YYYY-MM-DD> --out out/forecast.parquet
```

`--asof` must equal the card's `[provenance].data_cutoff`; the card is
authoritative. On any per-asset failure the fallback random walk
(Student-t(5), empirical spread) emits instead and the trigger is logged.

## How to test

From this directory, using the shared venv at `../../.venv`:

```bash
../.venv/bin/python -m pytest tests/ -q
```

The suite includes an integration test that parses every authored
`units/*/card.toml` in the repository, so a drift between this parser and the
real cards fails locally rather than at scoring time.

## The output contract

Three files, and exactly three — gate `g0_integrity` refuses anything else:

| File | Role |
|---|---|
| `forecast.parquet` | The scored artifact. Columns *exactly* `draw, asset, horizon, value`. |
| `forecast_meta.json` | Sidecar: `unit_id, asof, representation, asset_ids, horizons, n_draws, target`. |
| `forecast_rationale.md` | The derivation. Required, never scored. |

`n_draws` must lie in `[200, 20000]`. Draw ids are contiguous from zero. The
parquet must fill the card's `assets × horizons` grid exactly once — no duplicate
cell, no missing cell, no NaN.

### Deviations from `docs/TRACK-2-FORECASTING-DESIGN.md`

The design document is authoritative on architecture, method and rationale. Its
illustrative snippets disagree with the organizer's scorer on three mechanical
points, and the scorer wins, because the scorer is what grades the card:

| Point | Design document | This implementation |
|---|---|---|
| Parquet `unit_id` column | `assert "unit_id" in df.columns` | **No such column.** The scorer's `FORECAST_COLUMNS` is exact and refuses extras (`qfbench2_track_forecasting/grid.py`). |
| Gate names | `g0`, `g1`, `g2`, `g3` | `g0_integrity`, `g1_schema`, `g2_cutoff_resource`, `g3_domain_semantics`, as declared in every `units/*/card.toml`. |
| `canary_guid` location | `[metadata]` | `[contamination]`. Not read by this parser. |

Two further points are implementation decisions, not conflicts:

- The rationale file is `forecast_rationale.md`. `SOLVER-PLAYBOOK.md` calls it
  `rationale.md`; the scorer and the submission CLI both require the longer name.
- `compute_horizon_sd` and `compute_ewma_vol` take a `target_type`. A
  `log_return` target's observations already are daily changes, so differencing
  them again would be wrong; the parameter makes that explicit rather than
  assumed.

## Provenance

`ARTIFACT_PROVENANCE.md` records every dependency, its licence, and its
first-availability date, per `docs/ARTIFACT-POLICY.md`. It is written in Phase 6.
