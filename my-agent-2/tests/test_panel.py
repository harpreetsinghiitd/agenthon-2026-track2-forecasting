"""Tests for the panel tools: loading, the anchor, and the two width estimates.

The interesting cases are the ones the pipeline actually depends on — that the
as-of filter is inclusive and applied before anything else, that the file chosen
is deterministic when several match, and that the two width estimates agree with
hand-computed values rather than merely with themselves.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import pandas as pd
import pytest

from conftest import write_panel
from harness.tools.panel import (
    DEFAULT_EWMA_LAMBDA,
    PanelError,
    compute_ewma_vol,
    compute_horizon_sd,
    get_last_observation,
    read_panel,
)

ASOF = dt.date(2023, 1, 11)


def test_read_panel_returns_canonical_columns(tmp_path: pathlib.Path) -> None:
    """A read yields exactly `date`, `asset`, `value`, ascending by date."""
    write_panel(tmp_path, "p.parquet", {"AAA": np.arange(20.0)})

    frame = read_panel(tmp_path, "AAA", ASOF)

    assert list(frame.columns) == ["date", "asset", "value"]
    assert frame["date"].is_monotonic_increasing
    assert set(frame["asset"]) == {"AAA"}


def test_read_panel_filters_at_the_asof_inclusive(tmp_path: pathlib.Path) -> None:
    """Rows after `asof` are dropped; the as-of row itself is kept."""
    write_panel(tmp_path, "p.parquet", {"AAA": np.arange(30.0)})

    frame = read_panel(tmp_path, "AAA", ASOF)

    assert frame["date"].max() == pd.Timestamp(ASOF)
    assert len(frame) == 10  # 2024-01-02 .. 2024-01-11 inclusive


def test_read_panel_accepts_the_asset_id_spelling(tmp_path: pathlib.Path) -> None:
    """Some exports spell the asset axis `asset_id`; both are understood."""
    write_panel(tmp_path, "p.parquet", {"AAA": np.arange(20.0)}, asset_column="asset_id")

    frame = read_panel(tmp_path, "AAA", ASOF)

    assert list(frame.columns) == ["date", "asset", "value"]
    assert set(frame["asset"]) == {"AAA"}


def test_read_panel_picks_the_first_file_that_holds_the_asset(tmp_path: pathlib.Path) -> None:
    """Files are tried in sorted order and the first match wins, deterministically."""
    write_panel(tmp_path, "a_first.parquet", {"ZZZ": np.arange(20.0)})
    write_panel(tmp_path, "b_second.parquet", {"AAA": np.arange(20.0) + 5.0})

    frame = read_panel(tmp_path, "AAA", ASOF)

    assert frame["value"].iloc[0] == 5.0


def test_read_panel_is_indifferent_to_the_other_assets_in_the_file(tmp_path: pathlib.Path) -> None:
    """A shared panel file is narrowed to the requested asset alone."""
    write_panel(tmp_path, "p.parquet", {"AAA": np.arange(20.0), "BBB": np.arange(20.0) * 2.0})

    frame = read_panel(tmp_path, "BBB", ASOF)

    assert set(frame["asset"]) == {"BBB"}
    assert frame["value"].iloc[0] == 0.0
    assert len(frame) == 10


def test_read_panel_raises_when_no_parquet_exists(tmp_path: pathlib.Path) -> None:
    """An empty unit directory is an error, not an empty frame."""
    with pytest.raises(PanelError, match="no .*parquet"):
        read_panel(tmp_path, "AAA", ASOF)


def test_read_panel_raises_when_the_asset_is_absent(tmp_path: pathlib.Path) -> None:
    """The TRANSFER cards have no own-history; the caller must know that here."""
    write_panel(tmp_path, "p.parquet", {"ZZZ": np.arange(20.0)})

    with pytest.raises(PanelError, match="contains asset"):
        read_panel(tmp_path, "AAA", ASOF)


def test_read_panel_raises_when_the_asof_predates_all_history(tmp_path: pathlib.Path) -> None:
    """An asset present only after the cutoff has no history to anchor on."""
    write_panel(tmp_path, "p.parquet", {"AAA": np.arange(20.0)})

    with pytest.raises(PanelError, match="at or before"):
        read_panel(tmp_path, "AAA", dt.date(2020, 1, 1))


def test_read_panel_raises_on_a_malformed_panel(tmp_path: pathlib.Path) -> None:
    """A file with an asset axis but no `value` column is refused loudly."""
    tmp_path.mkdir(exist_ok=True)
    pd.DataFrame({"date": pd.date_range("2024-01-02", periods=3), "asset": ["AAA"] * 3}).to_parquet(
        tmp_path / "p.parquet", index=False
    )

    with pytest.raises(PanelError, match="missing required column"):
        read_panel(tmp_path, "AAA", ASOF)


def test_read_panel_drops_non_finite_values(tmp_path: pathlib.Path) -> None:
    """Holiday gaps and NaNs never reach the estimators."""
    write_panel(tmp_path, "p.parquet", {"AAA": np.array([1.0, np.nan, 3.0, np.inf])})

    frame = read_panel(tmp_path, "AAA", ASOF)

    assert np.isfinite(frame["value"]).all()
    assert list(frame["value"]) == [1.0, 3.0]


def test_get_last_observation_returns_the_anchor(tmp_path: pathlib.Path) -> None:
    """The anchor is the final row at or before the cutoff."""
    write_panel(tmp_path, "p.parquet", {"AAA": np.arange(30.0)})
    panel = read_panel(tmp_path, "AAA", ASOF)

    date, value = get_last_observation(panel)

    assert date == ASOF
    assert value == 9.0


def test_get_last_observation_rejects_an_empty_panel() -> None:
    """An empty panel has no anchor, and saying so is better than returning NaNs."""
    empty = pd.DataFrame({"date": [], "asset": [], "value": []})
    with pytest.raises(PanelError, match="empty"):
        get_last_observation(empty)


def test_horizon_sd_matches_a_hand_computed_value() -> None:
    """Five points, horizon two: changes [3, 5, 7], sample sd 2.0."""
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=5),
            "asset": ["AAA"] * 5,
            "value": [0.0, 1.0, 3.0, 6.0, 10.0],
        }
    )

    assert compute_horizon_sd(panel, 2) == pytest.approx(2.0)


def test_horizon_sd_is_zero_for_a_constant_drift() -> None:
    """A perfectly linear ramp has no spread at any horizon."""
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=40),
            "asset": ["AAA"] * 40,
            "value": np.arange(40.0) * 0.5,
        }
    )

    assert compute_horizon_sd(panel, 5) == pytest.approx(0.0, abs=1e-12)


def test_horizon_sd_grows_with_the_horizon() -> None:
    """Under a random walk the spread scales roughly with sqrt(h)."""
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=2000),
            "asset": ["AAA"] * 2000,
            "value": 100 + np.cumsum(rng.normal(0, 1, 2000)),
        }
    )

    short = compute_horizon_sd(panel, 10)
    long = compute_horizon_sd(panel, 40)

    assert long > short
    assert 1.5 < long / short < 2.5  # sqrt(4) = 2, with sampling slack


def test_horizon_sd_uses_cumulative_returns_for_log_return_targets() -> None:
    """For `log_return` the values already are the steps; they are not differenced."""
    values = np.array([0.01, -0.005, 0.02, 0.004, -0.01, 0.003])
    panel = pd.DataFrame(
        {"date": pd.date_range("2024-01-02", periods=6), "asset": ["AAA"] * 6, "value": values}
    )

    got = compute_horizon_sd(panel, 2, target_type="log_return")
    expected = float(pd.Series(values).rolling(2).sum().dropna().std(ddof=1))

    assert got == pytest.approx(expected)


def test_horizon_sd_refuses_a_panel_shorter_than_the_horizon() -> None:
    """No overlapping change exists, so the estimate is refused rather than guessed."""
    panel = pd.DataFrame(
        {"date": pd.date_range("2024-01-02", periods=4), "asset": ["A"] * 4, "value": [1.0, 2.0, 3.0, 4.0]}
    )

    with pytest.raises(PanelError, match="too few"):
        compute_horizon_sd(panel, 10)


def test_ewma_vol_of_constant_changes_is_the_change_size() -> None:
    """With every daily change equal to c the EWMA converges to |c| immediately."""
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=50),
            "asset": ["AAA"] * 50,
            "value": np.arange(50.0) * 0.5,
        }
    )

    assert compute_ewma_vol(panel) == pytest.approx(0.5)


def test_ewma_vol_tracks_the_recent_regime() -> None:
    """A calm tail inside a violent history pulls the estimate down toward the tail."""
    calm = np.concatenate([np.cumsum(np.full(200, 1.0)), np.cumsum(np.full(200, 0.01))])
    panel = pd.DataFrame(
        {"date": pd.date_range("2024-01-02", periods=400), "asset": ["A"] * 400, "value": calm}
    )

    vol = compute_ewma_vol(panel)

    assert vol < 0.2  # far below the 1.0 the violent stretch alone implies


def test_ewma_vol_rejects_an_out_of_range_lambda() -> None:
    """A decay outside [0, 1) is meaningless and is refused."""
    panel = pd.DataFrame(
        {"date": pd.date_range("2024-01-02", periods=5), "asset": ["A"] * 5, "value": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )

    with pytest.raises(PanelError, match="lambda"):
        compute_ewma_vol(panel, lam=1.0)


def test_ewma_default_lambda_is_the_riskmetrics_one() -> None:
    """The published daily default is 0.94; changing it changes every width."""
    assert DEFAULT_EWMA_LAMBDA == 0.94
