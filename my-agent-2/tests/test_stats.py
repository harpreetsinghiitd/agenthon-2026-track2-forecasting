"""Tests for `harness.tools.stats` — bootstrap shape, ARIMA contract.

The decisions that matter here: demeaning happens at the caller, block
bootstrap preserves short-run autocorrelation, and a too-short series is a
loud error rather than a silent garbage fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.tools.stats import (
    DEFAULT_ARIMA_ORDER,
    DEFAULT_BLOCK_SIZE,
    MIN_ARIMA_OBS,
    StatsError,
    bootstrap_paths,
    fit_arima,
)


def test_bootstrap_shape() -> None:
    rng = np.random.default_rng(0)
    out = bootstrap_paths(np.random.default_rng(1).normal(0, 0.01, 200), 500, 63, rng=rng)
    assert out.shape == (500, 63)


def test_bootstrap_is_deterministic_per_seed() -> None:
    changes = np.random.default_rng(2).normal(0, 0.02, 300)
    a = bootstrap_paths(changes, 50, 21, rng=np.random.default_rng(42))
    b = bootstrap_paths(changes, 50, 21, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_bootstrap_preserves_block_structure() -> None:
    # With block_size == n_obs every path is a rotation of a contiguous slice
    # of the source: consecutive differences within a block are source
    # differences, which i.i.d. resampling would destroy.
    src = np.sin(np.linspace(0, 6 * np.pi, 120))
    demeaned = src - src.mean()
    out = bootstrap_paths(demeaned, 40, 20, block_size=120, rng=np.random.default_rng(3))
    # Autocorrelation of one-step-ahead values within each path should be high.
    corrs = [np.corrcoef(out[i, :-1], out[i, 1:])[0, 1] for i in range(10)]
    assert np.mean(corrs) > 0.8


def test_bootstrap_rejects_tiny_series() -> None:
    with pytest.raises(StatsError, match="at least 2"):
        bootstrap_paths(np.array([0.5]), 10, 5, rng=np.random.default_rng(0))


def test_bootstrap_rejects_nans() -> None:
    with pytest.raises(StatsError, match="NaN"):
        bootstrap_paths(np.array([0.1, np.nan, 0.2]), 10, 5, rng=np.random.default_rng(0))


def test_fit_arima_returns_forecastable_result() -> None:
    series = np.random.default_rng(4).normal(0, 0.01, 500)
    fitted = fit_arima(series)
    fc = np.asarray(fitted.forecast(steps=3))
    assert fc.shape == (3,)
    assert np.isfinite(fc).all()


def test_fit_arima_accepts_exog() -> None:
    series = np.random.default_rng(5).normal(0, 0.01, 400)
    exog = np.random.default_rng(6).normal(0, 1, 400)
    fitted = fit_arima(series, exog=exog)
    assert fitted is not None


def test_fit_arima_rejects_short_series() -> None:
    with pytest.raises(StatsError, match="at least"):
        fit_arima(np.arange(MIN_ARIMA_OBS - 1, dtype=float) * 0.001)


def test_fit_arima_rejects_misaligned_exog() -> None:
    series = np.random.default_rng(7).normal(0, 0.01, 300)
    with pytest.raises(StatsError, match="exog"):
        fit_arima(series, exog=np.zeros(10))


def test_defaults_are_the_spec_ones() -> None:
    assert DEFAULT_BLOCK_SIZE == 7
    assert DEFAULT_ARIMA_ORDER == (1, 0, 0)
