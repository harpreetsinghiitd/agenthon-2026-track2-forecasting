"""Self-check for the LLM signal contract and conditioning functions.

Run:  python3 my_agent_3/testing/test_llm_signals.py
No endpoint is contacted; the extraction test asserts the neutral fallback
that must hold whenever MODEL_ENDPOINT is unconfigured or the call fails.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from llm import copula_adjustment, mean_shift, signals, tail_skew, variance_scale


def _draws(seed: int = 0, n_assets: int = 2, n_horizons: int = 3, n: int = 500):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, n_assets, n_horizons))


def test_parse_clamps_out_of_range():
    raw = json.dumps({
        "mean_shift_sigma": 5.0, "variance_scale": 9.0, "tail_strength": 2.0,
        "correlation_scale": 3.0, "shock_probability": -1.0,
        "tail_direction": "UP", "basis": "GUESS", "evidence": "x",
    })
    s = signals.parse_signals(raw)
    assert s["mean_shift_sigma"] == 0.5, s
    assert s["variance_scale"] == 1.75, s
    assert s["tail_strength"] == 1.0, s
    assert s["correlation_scale"] == 1.5, s
    assert s["shock_probability"] == 0.0, s
    assert s["tail_direction"] == "none", s
    assert s["basis"] == "inferred", s


def test_parse_inferred_cannot_narrow():
    s = signals.parse_signals(json.dumps({
        "variance_scale": 0.8, "basis": "inferred", "evidence": "tone read",
    }))
    assert s["variance_scale"] == 1.0, s
    s = signals.parse_signals(json.dumps({
        "variance_scale": 0.8, "basis": "established", "evidence": "print",
    }))
    assert s["variance_scale"] == 0.8, s


def test_parse_tolerates_fenced_json():
    raw = "```json\n" + json.dumps({"mean_shift_sigma": -0.3}) + "\n```"
    assert signals.parse_signals(raw)["mean_shift_sigma"] == -0.3


def test_parse_rejects_garbage():
    for bad in ("no json here", "{", json.dumps([1, 2])):
        try:
            signals.parse_signals(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_signals accepted garbage: {bad!r}")


def test_neutral_is_a_no_op():
    draws = _draws()
    for fn in (lambda d: mean_shift.apply(d, signals.NEUTRAL),
               lambda d: variance_scale.apply(d, signals.NEUTRAL),
               lambda d: tail_skew.apply(d, signals.NEUTRAL, seed=71),
               lambda d: copula_adjustment.apply(d, None, {}, signals.NEUTRAL)):
        out = fn(draws)
        assert np.array_equal(out, draws), fn


def test_mean_shift_moves_centre_by_sigma_sd():
    draws = _draws()
    s = dict(signals.NEUTRAL, mean_shift_sigma=0.25)
    out = mean_shift.apply(draws, s)
    sd = draws.std(axis=0)
    shift = np.median(out, axis=0) - np.median(draws, axis=0)
    assert np.allclose(shift, 0.25 * sd, atol=1e-9), shift


def test_variance_scale_multiplies_spread():
    draws = _draws()
    out = variance_scale.apply(draws, dict(signals.NEUTRAL, variance_scale=1.5))
    ratio = (out - np.median(out, axis=0)) / (draws - np.median(draws, axis=0))
    assert np.allclose(ratio, 1.5, atol=1e-9), ratio


def test_tail_skew_widens_only_stated_tail():
    draws = _draws()
    s = dict(signals.NEUTRAL, tail_direction="left", tail_strength=0.5)
    out = tail_skew.apply(draws, s)
    lo_in = np.quantile(draws, 0.05, axis=0)
    lo_out = np.quantile(out, 0.05, axis=0)
    assert np.all(lo_out < lo_in), "left tail must widen"
    assert np.all(np.quantile(out, 0.95, axis=0) == np.quantile(draws, 0.95, axis=0)), \
        "right tail must be untouched"


def test_copula_adjustment_raises_correlation():
    rng = np.random.default_rng(7)
    draws = np.stack([rng.normal(0, 1, (400, 3)), rng.normal(0, 1, (400, 3))], axis=1)
    s = dict(signals.NEUTRAL, correlation_scale=1.5)
    out = copula_adjustment.apply(draws, None, {}, s)
    before = np.corrcoef(draws[:, 0, 0], draws[:, 1, 0])[0, 1]
    after = np.corrcoef(out[:, 0, 0], out[:, 1, 0])[0, 1]
    assert after > before + 0.2, (before, after)
    single = copula_adjustment.apply(draws[:, :1], None, {}, s)
    assert np.array_equal(single, draws[:, :1])


def test_extract_signals_neutral_without_endpoint():
    import os
    saved = os.environ.pop("MODEL_ENDPOINT", None)
    try:
        s = signals.extract_signals([{"name": "d", "timestamp": "2020-01-01",
                                      "text": "hi"}], "2020-01-01", {})
        assert s["mean_shift_sigma"] == 0.0 and s["variance_scale"] == 1.0, s
        assert "No text adjustment" in s["evidence"], s
    finally:
        if saved is not None:
            os.environ["MODEL_ENDPOINT"] = saved


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
