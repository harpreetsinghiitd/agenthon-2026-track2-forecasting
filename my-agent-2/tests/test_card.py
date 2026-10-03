"""Tests for `harness.tools.card.parse_card`.

Covers the four card families, every required field, each way a card can be
malformed, and the `n_draws_min` default that the worked exemplar depends on.
The last test parses every authored unit in the repository, so a schema drift
between this parser and the real cards fails here rather than at scoring time.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib

import pytest

from conftest import REPO_ROOT, card_toml_text, write_card
from harness.tools.card import (
    DEFAULT_N_DRAWS_MIN,
    FAMILIES,
    CardParseError,
    CardSpec,
    parse_card,
)


def test_parses_every_field(tmp_path: pathlib.Path) -> None:
    """A well-formed card yields all eight fields, correctly typed."""
    path = write_card(
        tmp_path,
        card_toml_text(
            unit_id="t2-F3-curve-2024",
            asof="2024-06-28",
            assets=("UST_2Y", "UST_10Y"),
            horizons=(21, 63),
            target_type="level",
            value_unit="percent_per_annum",
            family="T2-F3",
        ),
    )

    spec = parse_card(path)

    assert spec.unit_id == "t2-F3-curve-2024"
    assert spec.asof == dt.date(2024, 6, 28)
    assert isinstance(spec.asof, dt.date)
    assert spec.asset_ids == ("UST_2Y", "UST_10Y")
    assert spec.horizons == (21, 63)
    assert spec.target_type == "level"
    assert spec.value_unit == "percent_per_annum"
    assert spec.n_draws_min == 200
    assert spec.family == "T2-F3"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_accepts_all_four_families(tmp_path: pathlib.Path, family: str) -> None:
    """Every locked family string parses."""
    path = write_card(tmp_path, card_toml_text(family=family))
    assert parse_card(path).family == family


def test_cell_count_is_the_grid_size(tmp_path: pathlib.Path) -> None:
    """`cell_count` is the product of the two grid axes."""
    path = write_card(tmp_path, card_toml_text(assets=("A", "B", "C"), horizons=(21, 63)))
    assert parse_card(path).cell_count == 6


def test_horizon_order_is_preserved(tmp_path: pathlib.Path) -> None:
    """Declared horizon order is part of the grid contract, so it is not sorted."""
    path = write_card(tmp_path, card_toml_text(horizons=(189, 63, 21)))
    assert parse_card(path).horizons == (189, 63, 21)


def test_missing_n_draws_min_defaults_to_the_scorer_floor(tmp_path: pathlib.Path) -> None:
    """The worked exemplar omits `n_draws_min`; absence means 200, not an error."""
    path = write_card(tmp_path, card_toml_text(n_draws_min=None))
    assert parse_card(path).n_draws_min == DEFAULT_N_DRAWS_MIN


def test_declared_n_draws_min_is_honoured(tmp_path: pathlib.Path) -> None:
    """A declared floor above the default is returned unchanged."""
    path = write_card(tmp_path, card_toml_text(n_draws_min=1000))
    assert parse_card(path).n_draws_min == 1000


def test_missing_file_raises(tmp_path: pathlib.Path) -> None:
    """An absent card is a parse error, not a bare `FileNotFoundError`."""
    with pytest.raises(CardParseError, match="card not found"):
        parse_card(tmp_path / "card.toml")


def test_malformed_toml_raises(tmp_path: pathlib.Path) -> None:
    """Broken TOML names the file rather than leaking the decoder's type."""
    path = write_card(tmp_path, 'key = "unterminated')
    with pytest.raises(CardParseError, match="not valid TOML"):
        parse_card(path)


def test_missing_targets_table_raises(tmp_path: pathlib.Path) -> None:
    """A card with no `[targets]` has no grid and cannot be answered."""
    path = write_card(
        tmp_path,
        '[task]\nid = "x"\n\n[metadata]\ncategory = "T2-F1"\n\n[provenance]\ndata_cutoff = "2024-01-02"\n',
    )
    with pytest.raises(CardParseError, match=r"\[targets\]"):
        parse_card(path)


@pytest.mark.parametrize(
    ("field", "text", "match"),
    [
        ("family", card_toml_text(family="T2-F5"), "category"),
        ("target_type", card_toml_text(target_type="price"), "target_type"),
        ("asof", card_toml_text(asof="28-06-2024"), "data_cutoff"),
        ("asset_ids", card_toml_text(assets=("A", "A")), "repeats"),
        ("horizons", card_toml_text(horizons=(21, 21)), "repeats"),
        ("horizons", card_toml_text(horizons=(0,)), "positive"),
        ("n_draws_min", card_toml_text(n_draws_min=0), "positive integer"),
    ],
)
def test_bad_field_raises_with_the_toml_path(
    tmp_path: pathlib.Path, field: str, text: str, match: str
) -> None:
    """Each malformed field is refused, and the message names the offending path."""
    path = write_card(tmp_path, text)
    with pytest.raises(CardParseError, match=match):
        parse_card(path)


def test_card_spec_is_hashable_and_frozen() -> None:
    """`CardSpec` is immutable, so it can be shared without defensive copying."""
    spec = CardSpec(
        unit_id="u",
        asof=dt.date(2024, 1, 2),
        asset_ids=("A",),
        horizons=(21,),
        target_type="level",
        value_unit="x",
        n_draws_min=200,
        family="T2-F1",
    )
    assert hash(spec) == hash(spec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.unit_id = "other"  # type: ignore[misc]


def test_parses_every_authored_unit_in_the_repository() -> None:
    """Every real `card.toml` shipped in `units/` parses, with the scorer's floor."""
    cards = sorted((REPO_ROOT / "units").glob("*/card.toml"))
    if not cards:
        pytest.skip("no authored units available in this checkout")

    for path in cards:
        spec = parse_card(path)
        assert spec.unit_id == path.parent.name, f"{path}: [task].id must equal the directory name"
        assert spec.family in FAMILIES
        assert spec.asset_ids and spec.horizons
        assert spec.n_draws_min >= DEFAULT_N_DRAWS_MIN
        assert (path.parent.name).startswith("t2-")
