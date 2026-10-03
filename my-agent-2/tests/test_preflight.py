"""Tests for `harness.validate.preflight.validate_gates` and the writer that feeds it.

One known-good submission must pass, and each of the five failure shapes the
build prompt names — four files, a wrong column, a wrong asof, a missing cell and
a NaN value — must fail *the specific gate that owns it*. A test that only
asserted "raises" would pass even if the gates were wired in the wrong order, so
every negative case pins the gate name.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from conftest import make_spec, valid_frame, valid_meta, write_output
from harness.tools.write import WriteError, build_meta, write_deliverables
from harness.validate.preflight import GATES, GateFailure, validate_gates


def _refusal(output_dir: pathlib.Path, card) -> GateFailure:
    """Validate and return the `GateFailure`, failing the test if none is raised."""
    with pytest.raises(GateFailure) as caught:
        validate_gates(output_dir, card)
    return caught.value


# ----------------------------------------------------------------------------------
# the known-good submission
# ----------------------------------------------------------------------------------


def test_known_good_passes_every_gate(tmp_path: pathlib.Path) -> None:
    """A well-formed single-cell submission passes g0-g3."""
    card = make_spec()
    write_output(tmp_path, card, n_draws=200)

    report = validate_gates(tmp_path, card)

    assert report.unit_id == card.unit_id
    assert report.n_draws == 200
    assert report.cell_count == 1
    assert report.gates == GATES


def test_known_good_multi_cell_grid_passes(tmp_path: pathlib.Path) -> None:
    """A four-asset, two-horizon joint submission passes."""
    card = make_spec(asset_ids=("A", "B", "C", "D"), horizons=(21, 63), family="T2-F3")
    write_output(tmp_path, card, n_draws=250)

    report = validate_gates(tmp_path, card)

    assert report.cell_count == 8
    assert report.n_draws == 250


# ----------------------------------------------------------------------------------
# the five known-bad shapes the build prompt names
# ----------------------------------------------------------------------------------


def test_g0_rejects_a_fourth_file(tmp_path: pathlib.Path) -> None:
    """g0: a log file beside the deliverables fails the card."""
    card = make_spec()
    write_output(tmp_path, card, extra_files={"run.log": "trace\n"})

    assert _refusal(tmp_path, card).gate == "g0_integrity"


def test_g1_rejects_a_wrong_column(tmp_path: pathlib.Path) -> None:
    """g1: the parquet's column set is exact, and an extra column is refused."""
    card = make_spec()
    frame = valid_frame(card, 200)
    frame["unit_id"] = card.unit_id  # the design document's stale extra column
    write_output(tmp_path, card, frame=frame)

    assert _refusal(tmp_path, card).gate == "g1_schema"


def test_g2_rejects_a_wrong_asof(tmp_path: pathlib.Path) -> None:
    """g2: the sidecar's asof must equal the card's data_cutoff."""
    card = make_spec()
    meta = valid_meta(card, 200)
    meta["asof"] = "2024-01-03"
    write_output(tmp_path, card, meta=meta)

    assert _refusal(tmp_path, card).gate == "g2_cutoff_resource"


def test_g3_rejects_a_missing_cell(tmp_path: pathlib.Path) -> None:
    """g3: one dropped row leaves the grid uncovered."""
    card = make_spec(asset_ids=("A", "B"), horizons=(21,))
    frame = valid_frame(card, 200).iloc[:-1]
    write_output(tmp_path, card, frame=frame)

    assert _refusal(tmp_path, card).gate == "g3_domain_semantics"


def test_g3_rejects_a_nan_value(tmp_path: pathlib.Path) -> None:
    """g3: NaN never reaches the metric."""
    card = make_spec()
    frame = valid_frame(card, 200)
    frame.loc[0, "value"] = np.nan
    write_output(tmp_path, card, frame=frame)

    assert _refusal(tmp_path, card).gate == "g3_domain_semantics"


# ----------------------------------------------------------------------------------
# the remaining ways each gate can fail
# ----------------------------------------------------------------------------------


def test_g0_rejects_a_missing_file(tmp_path: pathlib.Path) -> None:
    """g0: two files is as wrong as four."""
    card = make_spec()
    write_output(tmp_path, card)
    (tmp_path / "forecast_rationale.md").unlink()

    assert _refusal(tmp_path, card).gate == "g0_integrity"


def test_g0_rejects_a_subdirectory(tmp_path: pathlib.Path) -> None:
    """g0 counts entries, so a stray directory is refused too."""
    card = make_spec()
    write_output(tmp_path, card)
    (tmp_path / "__pycache__").mkdir()

    assert _refusal(tmp_path, card).gate == "g0_integrity"


@pytest.mark.parametrize("draws", [199, 20_001])
def test_g1_rejects_a_draw_count_outside_the_contract(tmp_path: pathlib.Path, draws: int) -> None:
    """g1: n_draws must sit inside [200, 20000]."""
    card = make_spec()
    write_output(tmp_path, card, n_draws=draws)

    assert _refusal(tmp_path, card).gate == "g1_schema"


def test_g1_rejects_a_blank_rationale(tmp_path: pathlib.Path) -> None:
    """g1: the rationale is never scored, but it must exist and be non-blank."""
    card = make_spec()
    write_output(tmp_path, card, rationale="   \n\t\n")

    assert _refusal(tmp_path, card).gate == "g1_schema"


def test_g1_rejects_a_parametric_representation(tmp_path: pathlib.Path) -> None:
    """g1: `parametric` is advertised upstream but has no loader, so it is refused."""
    card = make_spec()
    meta = valid_meta(card, 200)
    meta["representation"] = "parametric"
    write_output(tmp_path, card, meta=meta)

    assert _refusal(tmp_path, card).gate == "g1_schema"


def test_g2_rejects_a_wrong_unit_id(tmp_path: pathlib.Path) -> None:
    """g2: the sidecar must name the card it answers."""
    card = make_spec()
    meta = valid_meta(card, 200)
    meta["unit_id"] = "t2-F1-someone-else"
    write_output(tmp_path, card, meta=meta)

    assert _refusal(tmp_path, card).gate == "g2_cutoff_resource"


def test_g2_rejects_a_wrong_target_type(tmp_path: pathlib.Path) -> None:
    """g2: a level card answered with a log_return draw is a different question."""
    card = make_spec()
    meta = valid_meta(card, 200)
    meta["target"] = "log_return"
    write_output(tmp_path, card, meta=meta)

    assert _refusal(tmp_path, card).gate == "g2_cutoff_resource"


def test_g3_rejects_a_duplicate_cell_even_at_the_right_row_count(tmp_path: pathlib.Path) -> None:
    """g3: occupancy is checked directly, not inferred from the row count."""
    card = make_spec(asset_ids=("A",), horizons=(21, 63))
    frame = valid_frame(card, 200)
    frame["horizon"] = 21  # horizon 21 now doubled, horizon 63 absent, count unchanged
    write_output(tmp_path, card, frame=frame)

    assert _refusal(tmp_path, card).gate == "g3_domain_semantics"


def test_g3_rejects_a_draw_id_outside_the_declared_range(tmp_path: pathlib.Path) -> None:
    """g3: draw ids are contiguous and zero-based."""
    card = make_spec()
    frame = valid_frame(card, 200)
    frame.loc[0, "draw"] = 200
    write_output(tmp_path, card, frame=frame)

    assert _refusal(tmp_path, card).gate == "g3_domain_semantics"


def test_g3_rejects_a_reordered_asset_axis(tmp_path: pathlib.Path) -> None:
    """g3: order is part of the contract, because the realized vector uses it too."""
    card = make_spec(asset_ids=("AAA", "BBB"), horizons=(21,))
    meta = valid_meta(card, 200)
    meta["asset_ids"] = ["BBB", "AAA"]
    write_output(tmp_path, card, meta=meta)

    assert _refusal(tmp_path, card).gate == "g3_domain_semantics"


def test_g3_rejects_a_repeated_grid_label(tmp_path: pathlib.Path) -> None:
    """g3: repeating a label duplicates an easy cell into the average."""
    card = make_spec(asset_ids=("AAA",), horizons=(21,))
    meta = valid_meta(card, 200)
    meta["horizons"] = [21, 21]
    write_output(tmp_path, card, meta=meta)

    assert _refusal(tmp_path, card).gate == "g3_domain_semantics"


def test_g3_rejects_an_unknown_asset(tmp_path: pathlib.Path) -> None:
    """g3: an asset outside the card's grid is refused, never dropped."""
    card = make_spec(asset_ids=("AAA",), horizons=(21,))
    frame = valid_frame(card, 200)
    frame["asset"] = "ZZZ"
    write_output(tmp_path, card, frame=frame)

    assert _refusal(tmp_path, card).gate == "g3_domain_semantics"


def test_failure_message_names_the_gate() -> None:
    """`GateFailure` renders as `gate: reason`, which is what the log line shows."""
    failure = GateFailure("g1_schema", "something is wrong")
    assert str(failure) == "g1_schema: something is wrong"


# ----------------------------------------------------------------------------------
# the writer
# ----------------------------------------------------------------------------------


def test_writer_emits_exactly_three_files(tmp_path: pathlib.Path) -> None:
    """The writer is the only thing that touches the output directory."""
    card = make_spec()
    out = tmp_path / "out"
    draws = np.random.default_rng(0).normal(1.0, 0.1, size=(200, 1, 1))

    report = write_deliverables(draws, build_meta(card, 200), "Derivation.", out, card)

    assert sorted(p.name for p in out.iterdir()) == [
        "forecast.parquet",
        "forecast_meta.json",
        "forecast_rationale.md",
    ]
    assert report.gates == GATES


def test_writer_round_trips_the_draw_array(tmp_path: pathlib.Path) -> None:
    """Draw `d`, asset `i`, horizon `j` lands in the parquet cell the scorer reads."""
    card = make_spec(asset_ids=("A", "B"), horizons=(21, 63))
    out = tmp_path / "out"
    draws = np.arange(200 * 2 * 2, dtype=float).reshape(200, 2, 2)

    write_deliverables(draws, build_meta(card, 200), "Derivation.", out, card)

    frame = pd.read_parquet(out / "forecast.parquet")
    cell = frame[(frame["draw"] == 7) & (frame["asset"] == "B") & (frame["horizon"] == 63)]
    assert len(cell) == 1
    assert cell["value"].iloc[0] == draws[7, 1, 1]


def test_writer_refuses_a_mismatched_shape(tmp_path: pathlib.Path) -> None:
    """A draw array that does not match the card's grid is refused before writing."""
    card = make_spec(asset_ids=("A", "B"))
    out = tmp_path / "out"
    draws = np.zeros((200, 1, 1))

    with pytest.raises(WriteError, match="shape"):
        write_deliverables(draws, build_meta(card, 200), "Derivation.", out, card)


def test_writer_refuses_a_blank_rationale(tmp_path: pathlib.Path) -> None:
    """A blank rationale would fail g1, so it is refused before anything is written."""
    card = make_spec()
    out = tmp_path / "out"
    draws = np.zeros((200, 1, 1))

    with pytest.raises(WriteError, match="rationale"):
        write_deliverables(draws, build_meta(card, 200), "  \n", out, card)


def test_writer_refuses_non_finite_draws(tmp_path: pathlib.Path) -> None:
    """NaN would fail g3; catching it here names the real culprit."""
    card = make_spec()
    out = tmp_path / "out"
    draws = np.full((200, 1, 1), np.nan)

    with pytest.raises(WriteError, match="NaN"):
        write_deliverables(draws, build_meta(card, 200), "Derivation.", out, card)


def test_writer_removes_its_files_when_a_gate_refuses(tmp_path: pathlib.Path) -> None:
    """A refused submission leaves nothing behind, so a stale parquet cannot survive."""
    card = make_spec()
    out = tmp_path / "out"
    draws = np.zeros((200, 1, 1))

    with pytest.raises(GateFailure):
        write_deliverables(draws, {"unit_id": "wrong"}, "Derivation.", out, card)

    assert list(out.iterdir()) == []


def test_writer_is_byte_deterministic(tmp_path: pathlib.Path) -> None:
    """Two identical runs produce byte-identical deliverables."""
    card = make_spec(asset_ids=("A", "B"), horizons=(21, 63))
    draws = np.random.default_rng(3).normal(1.0, 0.1, size=(200, 2, 2))

    first = tmp_path / "first"
    second = tmp_path / "second"
    write_deliverables(draws, build_meta(card, 200), "Derivation.", first, card)
    write_deliverables(draws, build_meta(card, 200), "Derivation.", second, card)

    for name in ("forecast.parquet", "forecast_meta.json", "forecast_rationale.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_build_meta_matches_the_declared_contract() -> None:
    """The sidecar carries exactly the keys the scorer's schema requires."""
    card = make_spec(unit_id="t2-F1-x", asof=dt.date(2024, 1, 2))
    meta = build_meta(card, 500)

    assert set(meta) == {"unit_id", "asof", "representation", "asset_ids", "horizons", "n_draws", "target"}
    assert meta["representation"] == "samples"
    assert meta["asof"] == "2024-01-02"
    assert json.loads(json.dumps(meta)) == meta


def test_build_meta_refuses_a_draw_count_below_the_floor() -> None:
    """The scorer's 200-draw floor is enforced before anything is written."""
    with pytest.raises(WriteError, match="below the scorer's floor"):
        build_meta(make_spec(), 199)
