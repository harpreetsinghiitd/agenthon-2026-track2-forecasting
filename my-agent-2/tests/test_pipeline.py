"""End-to-end Phase 2 test: synthetic F1 card → gate-passing, byte-identical output.

Also forces the fallback path by breaking the primary one, proving the
pipeline always emits a valid submission.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import pandas as pd
import pytest

from conftest import card_toml_text, write_card, write_panel
from agent.main import run
from harness.tools import stats as stats_mod


def _synthetic_unit(tmp_path: pathlib.Path, unit_id: str = "t2-TEST-pipeline") -> tuple[pathlib.Path, pathlib.Path]:
    """A unit dir with a card and a noisy panel long enough for Phase 2."""
    unit = tmp_path / "unit"
    card_path = write_card(
        unit,
        card_toml_text(unit_id=unit_id, asof="2024-06-28", assets=("AAA",), horizons=(21, 63), family="T2-F1"),
    )
    rng = np.random.default_rng(7)
    values = 4.0 + np.cumsum(rng.normal(0.0, 0.05, 400))
    write_panel(
        unit,
        "rates.parquet",
        {"AAA": values},
        dates=pd.bdate_range("2022-10-03", periods=400),
    )
    return card_path, unit


def test_end_to_end_produces_gate_passing_output(tmp_path: pathlib.Path) -> None:
    card_path, unit = _synthetic_unit(tmp_path)
    out = tmp_path / "out"

    report = run(card_path, unit, out, n_draws=200)

    assert report.unit_id == "t2-TEST-pipeline"
    assert report.n_draws == 200
    assert sorted(p.name for p in out.iterdir()) == [
        "forecast.parquet",
        "forecast_meta.json",
        "forecast_rationale.md",
    ]
    frame = pd.read_parquet(out / "forecast.parquet")
    assert set(frame.columns) == {"draw", "asset", "horizon", "value"}
    assert np.isfinite(frame["value"].to_numpy()).all()


def test_rerun_is_byte_identical(tmp_path: pathlib.Path) -> None:
    card_path, unit = _synthetic_unit(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    run(card_path, unit, first, n_draws=200)
    run(card_path, unit, second, n_draws=200)

    for name in ("forecast.parquet", "forecast_meta.json", "forecast_rationale.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_fallback_still_emits_valid_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the primary path raises for the asset, the fallback forecaster runs
    and the submission still passes the gates."""
    card_path, unit = _synthetic_unit(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("arima engine unavailable")

    monkeypatch.setattr(stats_mod, "fit_arima", _boom)
    # agent.main imported the symbol at module import time; patch there too.
    import agent.main as agent_main
    monkeypatch.setattr(agent_main, "fit_arima", _boom)

    out = tmp_path / "out"
    with caplog.at_level("WARNING"):
        report = run(card_path, unit, out, n_draws=200)

    assert report.n_draws == 200
    assert any("fallback" in r.message.lower() for r in caplog.records)
    rationale = (out / "forecast_rationale.md").read_text()
    assert "FALLBACK" in rationale


def test_n_draws_floor_is_respected(tmp_path: pathlib.Path) -> None:
    card_path, unit = _synthetic_unit(tmp_path)
    out = tmp_path / "out"
    report = run(card_path, unit, out, n_draws=50)
    assert report.n_draws == 200  # clamped to the card/scorer floor


def test_log_return_card_runs(tmp_path: pathlib.Path) -> None:
    unit = tmp_path / "unit"
    card_path = write_card(
        unit,
        card_toml_text(
            unit_id="t2-TEST-logret",
            asof="2024-06-28",
            target_type="log_return",
            value_unit="log_return",
        ),
    )
    rng = np.random.default_rng(11)
    write_panel(
        unit,
        "rets.parquet",
        {"AAA": rng.normal(0.0004, 0.008, 400)},
        dates=pd.bdate_range("2022-10-03", periods=400),
    )
    report = run(card_path, unit, tmp_path / "out", n_draws=200)
    assert report.n_draws == 200
