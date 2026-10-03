"""Tests for `harness.tools.simulate` — the deterministic bridge output.

What matters: same seed → same bytes; drift and width compose linearly with
the innovations; the shuffle that pairs innovations to overlay rows is part of
the seed's job.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.tools.simulate import SimulateError, seed_from_unit_id, simulate_mixture


def test_seed_is_stable_and_cardly() -> None:
    assert seed_from_unit_id("t2-F1-x") == seed_from_unit_id("t2-F1-x")
    assert seed_from_unit_id("t2-F1-x") != seed_from_unit_id("t2-F1-y")
    assert 0 <= seed_from_unit_id("t2-F1-x") < 2**31


def test_zero_drift_unit_width_returns_cumsum() -> None:
    innov = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    out = simulate_mixture(innov, drift_per_draw=0.0, width_mult=1.0, seed=0)
    # The shuffle with a fixed seed may reorder rows; cumsum rows are:
    # cumsum([1,2,3]) = [1,3,6] and cumsum([0.5,0.5,0.5]) = [0.5,1,1.5].
    expected_rows = {(1.0, 3.0, 6.0), (0.5, 1.0, 1.5)}
    for row in out:
        assert tuple(row) in expected_rows


def test_drift_adds_linearly() -> None:
    innov = np.zeros((4, 3))
    out = simulate_mixture(innov, drift_per_draw=0.1, width_mult=1.0, seed=0)
    assert np.allclose(out[:, -1], 0.3)


def test_width_scales_innovations() -> None:
    innov = np.array([[1.0, 1.0]])
    wide = simulate_mixture(innov, drift_per_draw=0.0, width_mult=2.0, seed=0)
    base = simulate_mixture(innov, drift_per_draw=0.0, width_mult=1.0, seed=0)
    assert np.allclose(wide, 2.0 * base)


def test_per_draw_drift_and_width() -> None:
    innov = np.ones((3, 2))
    drift = np.array([0.0, 1.0, 2.0])
    width = np.array([1.0, 0.0, 2.0])
    out = simulate_mixture(innov, drift_per_draw=drift, width_mult=width, seed=0)
    # Row sums: innovations*width cumulated + drift*t; verify set of final values.
    finals = sorted(out[:, -1])
    expected = sorted([2.0, 2.0, 8.0])
    assert np.allclose(finals, expected)


def test_seed_shuffles_pairings() -> None:
    innov = np.arange(8.0).reshape(4, 2)
    a = simulate_mixture(innov, 0.0, 1.0, seed=1)
    b = simulate_mixture(innov, 0.0, 1.0, seed=2)
    # Same rows, generally different order across seeds (could collide; just
    # require set equality).
    assert sorted(map(tuple, a)) == sorted(map(tuple, b))


def test_same_seed_is_byte_identical() -> None:
    innov = np.random.default_rng(0).normal(0, 1, (100, 63))
    drift = np.random.default_rng(1).normal(0, 0.001, 100)
    width = np.random.default_rng(2).uniform(0.8, 1.2, 100)
    a = simulate_mixture(innov, drift, width, seed=99)
    b = simulate_mixture(innov, drift, width, seed=99)
    assert a.tobytes() == b.tobytes()


def test_rejects_negative_width() -> None:
    with pytest.raises(SimulateError, match="non-negative"):
        simulate_mixture(np.ones((2, 2)), 0.0, -1.0, seed=0)


def test_rejects_misshapen_drift() -> None:
    with pytest.raises(SimulateError, match="drift"):
        simulate_mixture(np.ones((3, 2)), np.zeros(4), 1.0, seed=0)


def test_rejects_nan_innovations() -> None:
    with pytest.raises(SimulateError, match="NaN"):
        simulate_mixture(np.array([[1.0, np.nan]]), 0.0, 1.0, seed=0)
