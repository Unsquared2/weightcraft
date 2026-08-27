from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import pytest

from conftest import dates, frame
from weightcraft.frame import DATE_COLUMN, WeightFrame

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix


def test_a_frame_cannot_be_written_to_after_construction() -> None:
    built = frame(("BTC", "ETH"), [[0.5, 0.5]])
    with pytest.raises(ValueError, match="read-only"):
        built.values[0, 0] = 1.0


def test_construction_copies_so_the_caller_cannot_reach_back_in() -> None:
    values = np.asarray([[0.5, 0.5]], dtype=np.float64)
    built = WeightFrame(dates=dates(1), assets=("BTC", "ETH"), values=values)
    values[0, 0] = 99.0
    assert built.values[0, 0] == 0.5


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="does not match"):
        WeightFrame(
            dates=dates(2),
            assets=("BTC",),
            values=np.zeros((1, 1), dtype=np.float64),
        )


def test_duplicate_assets_are_refused() -> None:
    with pytest.raises(ValueError, match="unique"):
        WeightFrame(
            dates=dates(1),
            assets=("BTC", "BTC"),
            values=np.zeros((1, 2), dtype=np.float64),
        )


def test_a_one_dimensional_values_block_is_refused() -> None:
    flat = cast("Matrix", np.zeros(2, dtype=np.float64))
    with pytest.raises(ValueError, match="two-dimensional"):
        WeightFrame(dates=dates(2), assets=("BTC",), values=flat)


def test_two_dimensional_dates_are_refused() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        WeightFrame(
            dates=np.zeros((2, 2), dtype="datetime64[ns]"),
            assets=("BTC",),
            values=np.zeros((2, 1), dtype=np.float64),
        )


def test_equality_counts_a_shared_gap_as_equal() -> None:
    left = frame(("BTC",), [[np.nan]])
    right = frame(("BTC",), [[np.nan]])
    assert left == right
    assert hash(left) == hash(right)


def test_equality_against_another_type_is_not_claimed() -> None:
    assert frame(("BTC",), [[1.0]]) != "BTC"


def test_a_polars_round_trip_preserves_values_and_gaps() -> None:
    built = frame(("BTC", "ETH"), [[0.25, np.nan], [-0.5, 0.5]])
    assert WeightFrame.from_polars(built.to_polars()) == built


def test_a_gap_renders_as_null_rather_than_nan() -> None:
    rendered = frame(("BTC",), [[np.nan]]).to_polars()
    assert rendered.get_column("BTC").to_list() == [None]


def test_a_frame_without_a_date_column_is_refused() -> None:
    with pytest.raises(ValueError, match="date"):
        WeightFrame.from_polars(pl.DataFrame({"BTC": [0.1]}))


def test_from_polars_reads_integer_columns_as_floats() -> None:
    read = WeightFrame.from_polars(
        pl.DataFrame({DATE_COLUMN: [date(2026, 1, 1)], "BTC": [1]})
    )
    assert read.values.dtype == np.float64
    assert read.values.tolist() == [[1.0]]


def test_from_polars_accepts_a_plain_date_column() -> None:
    read = WeightFrame.from_polars(
        pl.DataFrame({DATE_COLUMN: [date(2026, 1, 1)], "BTC": [0.5]})
    )
    assert read.dates[0] == np.datetime64("2026-01-01", "ns")


def test_shape_and_emptiness_report_the_labels() -> None:
    built = frame(("BTC", "ETH"), [[0.5, 0.5], [0.4, 0.6]])
    assert built.shape == (2, 2)
    assert not built.is_empty
    assert repr(built) == "WeightFrame(2 dates x 2 assets)"


def test_with_values_keeps_the_labels() -> None:
    built = frame(("BTC", "ETH"), [[0.5, 0.5]])
    replaced = built.with_values(np.zeros((1, 2), dtype=np.float64))
    assert replaced.assets == built.assets
    assert replaced.values.tolist() == [[0.0, 0.0]]


def test_last_row_is_the_latest_book() -> None:
    built = frame(("BTC", "ETH"), [[0.5, 0.5], [0.1, 0.9]])
    assert built.last_row().tolist() == [0.1, 0.9]


def test_an_empty_frame_has_no_last_row() -> None:
    empty = WeightFrame.from_rows([], ("BTC",), np.zeros((0, 1), dtype=np.float64))
    assert empty.is_empty
    with pytest.raises(ValueError, match="empty frame"):
        empty.last_row()


def test_from_rows_accepts_iso_strings() -> None:
    built = WeightFrame.from_rows(
        ["2026-01-01"], ("BTC",), np.asarray([[1.0]], dtype=np.float64)
    )
    assert built.dates[0] == np.datetime64("2026-01-01", "ns")
