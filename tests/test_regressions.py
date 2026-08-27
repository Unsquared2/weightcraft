"""One test per defect found in review, each named after what it broke."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from weightcraft.align import align
from weightcraft.combine import (
    nanmean_stack,
    nanmedian_stack,
    weighted_nanmean_stack,
)
from weightcraft.costs import book_returns, turnover
from weightcraft.frame import WeightFrame
from weightcraft.normalize import center, gross, to_gross, weights_from_bins
from weightcraft.risk import rolling_sums, trailing_std
from weightcraft.smoothing import ewm_mean, rolling_mean

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix, Vector

from conftest import dates, frame


def test_a_repeated_date_is_refused_rather_than_overwriting_a_row() -> None:
    # `align` places rows by `searchsorted` against the union of every frame's
    # dates, so a duplicate is one row silently replacing another.
    repeated = np.asarray(
        ["2026-01-01", "2026-01-01", "2026-01-02"], dtype="datetime64[ns]"
    )
    with pytest.raises(ValueError, match="dates must be unique"):
        WeightFrame(
            dates=repeated,
            assets=("BTC",),
            values=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float64),
        )


def test_unsorted_dates_are_still_placed_correctly() -> None:
    out_of_order = np.asarray(["2026-01-03", "2026-01-01"], dtype="datetime64[ns]")
    built = WeightFrame(
        dates=out_of_order,
        assets=("BTC",),
        values=np.asarray([[3.0], [1.0]], dtype=np.float64),
    )
    stack = align([built])
    assert stack.values[0, :, 0].tolist() == [1.0, 3.0]


def test_gross_treats_an_infinity_as_missing_rather_than_as_a_large_number() -> None:
    # Summing it made one broken cell divide every good position on that date
    # down to zero.
    assert gross(np.asarray([[1.0, np.inf, 2.0]])).tolist() == [[3.0]]


def test_one_broken_cell_does_not_wipe_the_rest_of_the_row() -> None:
    scaled = to_gross(np.asarray([[np.inf, 0.3, 0.2]]), 1.0)
    assert scaled[0, 1] == pytest.approx(0.6)
    assert scaled[0, 2] == pytest.approx(0.4)


def test_every_reduction_agrees_that_an_infinity_is_missing() -> None:
    stack = np.asarray([[[np.inf, 2.0]], [[3.0, 4.0]]])
    shares: Vector = np.asarray([0.5, 0.5])
    assert nanmean_stack(stack).tolist() == [[3.0, 3.0]]
    assert nanmedian_stack(stack).tolist() == [[3.0, 3.0]]
    assert weighted_nanmean_stack(stack, shares).tolist() == [[3.0, 3.0]]


def test_a_share_that_is_not_a_non_negative_number_is_refused() -> None:
    stack = np.zeros((2, 1, 1))
    for shares in (np.asarray([-1.0, 1.0]), np.asarray([np.nan, 1.0])):
        with pytest.raises(ValueError, match="finite and non-negative"):
            weighted_nanmean_stack(stack, shares)


@pytest.mark.parametrize("magnitude", [1e6, 1e8, 1e10])
def test_trailing_std_survives_a_level_that_dwarfs_the_variation(
    magnitude: float,
) -> None:
    # The differenced cumulative-sum form cancelled catastrophically here and
    # reported a confident zero at 1e10.
    series: Vector = magnitude + np.random.default_rng(0).normal(size=120) * 1e-3
    assert trailing_std(series, 30)[-1] == pytest.approx(
        float(np.std(series[-30:], ddof=1)), rel=1e-9
    )


def test_a_gap_only_voids_the_windows_that_contain_it() -> None:
    sums = rolling_sums(np.asarray([1.0, 2.0, np.nan, 4.0, 5.0, 6.0]), 2)
    assert sums[4:].tolist() == [9.0, 11.0]
    assert np.isnan(sums[[0, 2, 3]]).all()


def test_a_bin_scheme_that_does_not_start_at_zero_is_still_market_neutral() -> None:
    for row in ([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]):
        weights = weights_from_bins(np.asarray([row]))
        assert float(np.nansum(weights)) == pytest.approx(0.0)
        assert float(np.nansum(np.abs(weights))) == pytest.approx(1.0)


def test_a_negative_bin_scheme_is_not_inverted() -> None:
    weights = weights_from_bins(np.asarray([[-1.0, -2.0, -3.0]]))
    assert weights[0, 0] > weights[0, 2]


def test_a_row_where_every_name_lands_in_one_bin_expresses_no_view() -> None:
    assert weights_from_bins(np.asarray([[4.0, 4.0, 4.0]])).tolist() == [
        [0.0, 0.0, 0.0]
    ]


def test_a_date_no_source_covered_does_not_raise_a_warning() -> None:
    # The suite runs under `filterwarnings = ["error"]`, and an uncovered date
    # is what `align` calls the ordinary case.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        centred = center(np.asarray([[np.nan, np.nan], [1.0, 3.0]]))
        binned = weights_from_bins(np.asarray([[np.nan, np.nan, np.nan]]))
    assert np.isnan(centred[0]).all()
    assert np.isnan(binned).all()


def test_a_smoother_counts_observations_not_rows() -> None:
    # pandas' `min_periods` counts observations; answering off one point and
    # presenting it as a settled average is what this used to do.
    ragged: Matrix = np.asarray([[np.nan], [np.nan], [1.0], [2.0], [3.0]])
    assert np.isnan(ewm_mean(ragged, 3)[:4, 0]).all()
    assert ewm_mean(ragged, 3)[4, 0] == pytest.approx(2.4285714, rel=1e-6)
    smoothed = rolling_mean(ragged, 3)
    assert np.isnan(smoothed[:4, 0]).all()
    assert smoothed[4, 0] == pytest.approx(2.0)


def test_a_series_shorter_than_its_window_answers_nothing() -> None:
    assert np.isnan(rolling_mean(np.zeros((2, 1)), 5)).all()
    assert np.isnan(ewm_mean(np.zeros((2, 1)), 5)).all()
    assert np.isnan(trailing_std(np.zeros(2), 5)).all()
    assert np.isnan(rolling_sums(np.zeros(2), 5)).all()


def test_the_transaction_cost_knob_is_not_a_no_op_on_an_aligned_book() -> None:
    # `align` pads every uncovered cell with NaN, so a source that rotates
    # between assets is nothing but gap transitions -- which used to be free.
    rotating = frame(("BTC",), [[1.0], [np.nan]])
    other = frame(("ETH",), [[np.nan], [1.0]])
    weights = align([rotating, other]).values[0]
    returns = np.zeros_like(weights)
    free = book_returns(weights, returns, cost=0.0)
    charged = book_returns(weights, returns, cost=0.01)
    assert float(np.nansum(free)) == pytest.approx(0.0)
    assert float(np.nansum(charged)) < 0.0


def test_an_infinite_weight_is_treated_as_missing_like_everywhere_else() -> None:
    # Keeping it produced `inf - inf` on the following row, which is NaN -- a
    # broken weight that ended up costing nothing at all.
    assert turnover(np.asarray([[0.0], [np.inf]]))[1, 0] == 0.0
    assert turnover(np.asarray([[np.inf], [np.inf]]))[1, 0] == 0.0


def test_a_frame_with_dates_but_no_assets_still_reports_its_dates() -> None:
    empty = WeightFrame(
        dates=dates(2), assets=(), values=np.zeros((2, 0), dtype=np.float64)
    )
    assert empty.is_empty
    assert empty.has_dates
