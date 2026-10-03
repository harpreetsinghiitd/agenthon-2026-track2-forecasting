"""Deterministic tool harness for the Track 2 forecasting agent.

Everything under `harness/` is deterministic: given the same card, the same
panel, the same corpus and the same LLM cache, it produces the same bytes.  The
non-deterministic policy lives in `agent/` and reaches this package only through
the tools defined here.
"""
