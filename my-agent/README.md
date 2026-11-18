# v1: Baseline Agent

The first working version of our agent. Its job was to prove the end-to-end plumbing
before any modelling work began: read a task, produce a valid forecast, and pass the
competition's automatic checks.

## What it does

1. **Finds and reads the task card** (`card.toml`), which supports both card layouts
   used by the organizers.
2. **Loads the price panel** named on the card.
3. **Produces 500 simulated futures** (or more, if the card requires it) using the
   organizers' reference statistical baseline.
4. **Writes the three required files:**
   - `forecast.parquet`: the simulated paths, in the exact column format the scorer
     expects
   - `forecast_meta.json`: task ID, cut-off date, assets, horizons and draw count
   - `forecast_rationale.md`: a short human-readable explanation of the method

## Why it matters

A task that fails to produce valid output receives the worst possible score. v1 gave us
a guaranteed-valid submission to fall back on, and it fixed the input/output contract
that every later version reuses.

## What came next

v1 ignores the text corpus completely. [v2](../my-agent-2) added an LLM that reads the
documents, and [v3](../my_agent_3), the final agent, built a layered statistical engine
with text adjustments on top.

## Running it

From the root of the organizers' Track 2 starter kit:

```bash
REPO=$(pwd) my-agent/forecast.sh --panels units/<task> --text units/<task>/text --asof <YYYY-MM-DD> --out out/forecast.parquet
```
