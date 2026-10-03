"""Tests for `harness.bridge.scenario_mix` — deterministic scenario mixing."""

from __future__ import annotations

import numpy as np
import pytest

from harness.bridge.scenario_mix import scenario_mix


def test_scenario_mix_basic() -> None:
    scenarios = [(0.1, 1.0, 0.0), (-0.1, 1.5, 0.2)]
    weights = [0.6, 0.4]
    drifts, widths, skews = scenario_mix(scenarios, weights, n_draws=1000, seed=42)
    assert len(drifts) == 1000
    assert len(widths) == 1000
    assert len(skews) == 1000
    # Roughly 60/40 split
    sc0 = sum(1 for d in drifts if abs(d - 0.1) < 1e-9)
    sc1 = sum(1 for d in drifts if abs(d + 0.1) < 1e-9)
    assert 0.5 < sc0 / 1000 < 0.7
    assert 0.3 < sc1 / 1000 < 0.5


def test_scenario_mix_deterministic_same_seed() -> None:
    scenarios = [(0.1, 1.0, 0.0), (-0.1, 1.5, 0.2)]
    weights = [0.6, 0.4]
    a = scenario_mix(scenarios, weights, n_draws=100, seed=99)
    b = scenario_mix(scenarios, weights, n_draws=100, seed=99)
    assert a == b
    assert a[0] == b[0]


def test_scenario_mix_different_seed_different_assignment() -> None:
    scenarios = [(0.1, 1.0, 0.0), (-0.1, 1.5, 0.2)]
    weights = [0.6, 0.4]
    a = scenario_mix(scenarios, weights, n_draws=100, seed=1)
    b = scenario_mix(scenarios, weights, n_draws=100, seed=2)
    # Very unlikely to be identical (but possible); just check they're lists
    assert len(a[0]) == len(b[0])


def test_scenario_mix_renormalises_weights() -> None:
    scenarios = [(0.1, 1.0, 0.0), (-0.1, 1.0, 0.0)]
    weights = [2.0, 1.0]  # will be renormalised to 2/3, 1/3
    drifts, _, _ = scenario_mix(scenarios, weights, n_draws=3000, seed=0)
    sc0 = sum(1 for d in drifts if d == 0.1)
    assert 0.6 < sc0 / 3000 < 0.7


def test_scenario_mix_single_scenario() -> None:
    scenarios = [(0.05, 1.2, -0.1)]
    weights = [1.0]
    drifts, widths, skews = scenario_mix(scenarios, weights, n_draws=50, seed=0)
    assert all(d == 0.05 for d in drifts)
    assert all(w == 1.2 for w in widths)
    assert all(s == -0.1 for s in skews)


def test_scenario_mix_invalid_n_draws() -> None:
    with pytest.raises(ValueError, match="n_draws"):
        scenario_mix([(0.0, 1.0, 0.0)], [1.0], n_draws=0, seed=0)


def test_scenario_mix_length_mismatch() -> None:
    with pytest.raises(ValueError, match="scenarios.*!= weights"):
        scenario_mix([(0.0, 1.0, 0.0), (0.1, 1.0, 0.0)], [1.0], n_draws=10, seed=0)


def test_scenario_mix_empty_scenarios() -> None:
    with pytest.raises(ValueError, match="at least one"):
        scenario_mix([], [], n_draws=10, seed=0)


def test_scenario_mix_output_types() -> None:
    scenarios = [(0.1, 1.0, 0.0)]
    weights = [1.0]
    drifts, widths, skews = scenario_mix(scenarios, weights, n_draws=5, seed=0)
    assert isinstance(drifts, list)
    assert all(isinstance(d, float) for d in drifts)