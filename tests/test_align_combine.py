from __future__ import annotations

import numpy as np
import pytest

from conftest import dates, frame
from weightcraft.align import align
from weightcraft.combine import (
    nanmean_stack,
    nanmedian_stack,
    normalised_shares,
    weighted_nanmean_stack,
    weighted_nanmean_stack_over_time,
)
from weightcraft.frame import WeightFrame


def test_align_takes_the_union_of_assets_rather_than_the_intersection() -> None:
    stack = align([frame(("BTC",), [[1.0]]), frame(("ETH",), [[2.0]])])
    assert stack.assets == ("BTC", "ETH")
    assert np.array_equal(
        stack.values, np.asarray([[[1.0, np.nan]], [[np.nan, 2.0]]]), equal_nan=True
    )


def test_align_takes_the_union_of_dates() -> None:
    early = WeightFrame(
        dates=dates(1, "2026-01-01"),
        assets=("BTC",),
        values=np.asarray([[1.0]], dtype=np.float64),
    )
    late = WeightFrame(
        dates=dates(1, "2026-01-03"),
        assets=("BTC",),
        values=np.asarray([[3.0]], dtype=np.float64),
    )
    stack = align([early, late])
    assert stack.dates.size == 2
    assert stack.count == 2


def test_align_is_indifferent_to_the_order_the_assets_arrive_in() -> None:
    forward = align([frame(("BTC", "ETH"), [[1.0, 2.0]])])
    backward = align([frame(("ETH", "BTC"), [[2.0, 1.0]])])
    assert np.array_equal(forward.values, backward.values, equal_nan=True)
    assert forward.assets == backward.assets


def test_align_refuses_an_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        align([])


def test_align_tolerates_a_frame_with_no_rows() -> None:
    empty = WeightFrame.from_rows([], ("BTC",), np.zeros((0, 1), dtype=np.float64))
    stack = align([frame(("BTC",), [[1.0]]), empty])
    assert np.isnan(stack.values[1]).all()


def test_the_stack_can_rewrap_a_result_under_its_own_labels() -> None:
    stack = align([frame(("BTC", "ETH"), [[1.0, 2.0]])])
    rewrapped = stack.with_values(nanmean_stack(stack.values))
    assert rewrapped.assets == ("BTC", "ETH")
    assert rewrapped.values.tolist() == [[1.0, 2.0]]
    assert repr(stack) == "AlignedStack(1 frames x 1 dates x 2 assets)"


def test_the_mean_skips_a_source_that_is_silent_on_a_cell() -> None:
    stack = align([frame(("BTC",), [[1.0]]), frame(("BTC",), [[np.nan]])])
    assert nanmean_stack(stack.values).tolist() == [[1.0]]


def test_a_cell_no_source_covered_stays_missing() -> None:
    stack = align([frame(("BTC", "ETH"), [[1.0, np.nan]])])
    combined = nanmean_stack(stack.values)
    assert combined[0, 0] == 1.0
    assert np.isnan(combined[0, 1])


def test_the_median_ignores_one_source_going_haywire() -> None:
    stack = align(
        [
            frame(("BTC",), [[0.10]]),
            frame(("BTC",), [[0.12]]),
            frame(("BTC",), [[9.99]]),
        ]
    )
    assert nanmedian_stack(stack.values).tolist() == [[0.12]]
    assert np.isnan(nanmedian_stack(align([frame(("BTC",), [[np.nan]])]).values)).all()


def test_a_weighted_mean_drops_a_gap_from_both_sides() -> None:
    stack = align([frame(("BTC",), [[1.0]]), frame(("BTC",), [[np.nan]])])
    shares = np.asarray([0.25, 0.75])
    # Not 0.25: the silent source is removed from the denominator too, so the
    # cell stays at the level the source that spoke actually set.
    assert weighted_nanmean_stack(stack.values, shares).tolist() == [[1.0]]


def test_a_weighted_mean_of_equal_shares_is_the_plain_mean() -> None:
    stack = align([frame(("BTC",), [[1.0]]), frame(("BTC",), [[3.0]])])
    shares = np.asarray([0.5, 0.5])
    assert weighted_nanmean_stack(stack.values, shares).tolist() == [[2.0]]


def test_a_weighted_mean_refuses_a_share_per_source_mismatch() -> None:
    stack = align([frame(("BTC",), [[1.0]])])
    with pytest.raises(ValueError, match="expected shares of shape"):
        weighted_nanmean_stack(stack.values, np.asarray([0.5, 0.5]))


def test_time_varying_shares_can_hand_a_date_to_one_source() -> None:
    stack = align(
        [frame(("BTC",), [[1.0], [1.0]]), frame(("BTC",), [[3.0], [3.0]])],
    )
    shares = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    assert weighted_nanmean_stack_over_time(stack.values, shares).tolist() == [
        [1.0],
        [3.0],
    ]


def test_time_varying_shares_are_checked_against_the_stack() -> None:
    stack = align([frame(("BTC",), [[1.0]])])
    with pytest.raises(ValueError, match="expected shares of shape"):
        weighted_nanmean_stack_over_time(stack.values, np.zeros((2, 2)))


def test_shares_normalise_to_one_and_fall_back_to_equal() -> None:
    assert normalised_shares(np.asarray([1.0, 3.0])).tolist() == [0.25, 0.75]
    assert normalised_shares(np.asarray([0.0, 0.0])).tolist() == [0.5, 0.5]
    assert normalised_shares(np.asarray([np.inf, 1.0])).tolist() == [0.0, 1.0]
