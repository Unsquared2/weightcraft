from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from conftest import dates, frame
from weightcraft.align import align, carried
from weightcraft.combine import (
    mean_stack,
    nanmean_stack,
    nanmedian_stack,
    normalised_shares,
    weighted_mean_stack,
    weighted_nanmean_stack,
    weighted_nanmean_stack_over_time,
)
from weightcraft.frame import WeightFrame

if TYPE_CHECKING:
    from weightcraft.arrays import Dates


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


def test_the_mean_counts_a_silent_source_as_flat_rather_than_skipping_it() -> None:
    stack = align([frame(("BTC",), [[1.0]]), frame(("BTC",), [[np.nan]])])
    # The counterpart to `test_the_mean_skips_a_source_that_is_silent_on_a_cell`
    # above: the silent source is counted at zero rather than removed, so the
    # mean is halved rather than left where the source that spoke set it.
    assert nanmean_stack(stack.values).tolist() == [[1.0]]
    assert mean_stack(stack.values).tolist() == [[0.5]]


def test_the_mean_reads_missing_the_way_every_reduction_does() -> None:
    """`present` calls an infinity broken, so it counts as flat, not as large."""
    stack = align([frame(("BTC",), [[np.inf]]), frame(("BTC",), [[1.0]])])
    assert mean_stack(stack.values).tolist() == [[0.5]]


def test_the_two_means_agree_when_no_cell_is_missing() -> None:
    stack = align([frame(("BTC",), [[1.0]]), frame(("BTC",), [[-3.0]])])
    assert mean_stack(stack.values).tolist() == nanmean_stack(stack.values).tolist()


def test_the_mean_fills_a_cell_no_source_covered() -> None:
    stack = align([frame(("BTC", "ETH"), [[1.0, np.nan]])])
    assert np.isnan(nanmean_stack(stack.values))[0, 1]
    assert mean_stack(stack.values).tolist() == [[1.0, 0.0]]


def test_a_weighted_mean_keeps_a_gap_in_the_denominator() -> None:
    stack = align([frame(("BTC",), [[1.0]]), frame(("BTC",), [[np.nan]])])
    shares = np.asarray([0.25, 0.75])
    # The counterpart to `test_a_weighted_mean_drops_a_gap_from_both_sides`:
    # the silent source keeps its 0.75 of the denominator, at zero.
    assert weighted_nanmean_stack(stack.values, shares).tolist() == [[1.0]]
    assert weighted_mean_stack(stack.values, shares).tolist() == [[0.25]]


def test_a_weighted_mean_of_equal_shares_is_the_plain_one() -> None:
    stack = align([frame(("BTC",), [[1.0]]), frame(("BTC",), [[np.nan]])])
    shares = np.asarray([0.5, 0.5])
    assert (
        weighted_mean_stack(stack.values, shares).tolist()
        == mean_stack(stack.values).tolist()
    )


def test_a_weighted_mean_refuses_the_same_bad_shares_the_skipping_one_does() -> None:
    stack = align([frame(("BTC",), [[1.0]])])
    with pytest.raises(ValueError, match="expected shares of shape"):
        weighted_mean_stack(stack.values, np.asarray([0.5, 0.5]))
    with pytest.raises(ValueError, match="non-negative"):
        weighted_mean_stack(stack.values, np.asarray([-1.0]))


def test_neither_mean_ever_changes_a_cell_that_was_present() -> None:
    stack = align([frame(("BTC", "ETH"), [[0.4, np.nan]])])
    assert mean_stack(stack.values)[0, 0] == 0.4
    assert weighted_mean_stack(stack.values, np.asarray([1.0]))[0, 0] == 0.4


def test_a_weighted_mean_with_no_share_anywhere_is_missing_not_flat() -> None:
    """Filling is about a silent *cell*; a stack nobody weights says nothing.

    The gap-free guarantee holds given a frame carrying a positive share --
    degenerate shares fall back to what the `nan*` pair answers, so the two
    do not disagree about a case neither can size.
    """
    stack = align([frame(("BTC",), [[1.0]]), frame(("BTC",), [[3.0]])])
    zero = np.zeros(2)
    assert np.isnan(weighted_nanmean_stack(stack.values, zero)).all()
    assert np.isnan(weighted_mean_stack(stack.values, zero)).all()


def test_scaling_every_share_by_the_same_amount_is_the_same_mean() -> None:
    stack = align([frame(("BTC",), [[1.0]]), frame(("BTC",), [[np.nan]])])
    plain = weighted_mean_stack(stack.values, np.asarray([3.0, 1.0]))
    scaled = weighted_mean_stack(stack.values, np.asarray([300.0, 100.0]))
    assert plain.tolist() == scaled.tolist()


# ---------------------------------------------------------------------- carried


def _hours(count: int, start: str = "2026-01-01") -> Dates:
    """`count` consecutive hours from `start`, at nanosecond resolution."""
    origin = np.datetime64(start, "ns")
    hour = np.timedelta64(1, "h").astype("timedelta64[ns]")
    out: Dates = origin + np.arange(count) * hour
    return out


def test_carried_gives_each_grid_row_the_newest_frame_row_at_or_before_it() -> None:
    held = carried(frame(("BTC",), [[1.0], [2.0]]), _hours(48))
    assert np.array_equal(held.dates, _hours(48))
    assert np.array_equal(held.values[:24], np.full((24, 1), 1.0))
    assert np.array_equal(held.values[24:], np.full((24, 1), 2.0))


def test_carried_leaves_grid_rows_before_the_frames_first_row_missing() -> None:
    """Absent because the book had not started, which is not a flat position."""
    held = carried(frame(("BTC",), [[1.0]]), _hours(3, "2025-12-31T22:00"))
    assert np.array_equal(
        held.values, np.asarray([[np.nan], [np.nan], [1.0]]), equal_nan=True
    )


def test_carried_holds_the_last_row_across_every_grid_row_after_it() -> None:
    held = carried(frame(("BTC",), [[1.0]]), _hours(3))
    assert np.array_equal(held.values, np.full((3, 1), 1.0))


def test_carried_keeps_a_missing_cell_missing_rather_than_carrying_a_position() -> None:
    """A hole is "not held", so filling it would reopen a closed position."""
    held = carried(frame(("BTC", "ETH"), [[1.0, np.nan]]), _hours(2))
    assert np.array_equal(
        held.values, np.asarray([[1.0, np.nan], [1.0, np.nan]]), equal_nan=True
    )


def test_carried_blanks_a_row_it_would_have_carried_further_than_max_age() -> None:
    held = carried(frame(("BTC",), [[1.0]]), _hours(4), max_age=np.timedelta64(2, "h"))
    assert np.array_equal(
        held.values,
        np.asarray([[1.0], [1.0], [1.0], [np.nan]]),
        equal_nan=True,
    )


def test_carried_without_a_max_age_carries_without_bound() -> None:
    held = carried(frame(("BTC",), [[1.0]]), _hours(1000))
    assert np.array_equal(held.values, np.full((1000, 1), 1.0))


def test_carried_onto_a_frames_own_dates_returns_that_frame() -> None:
    original = frame(("BTC", "ETH"), [[1.0, np.nan], [2.0, 3.0]])
    assert carried(original, original.dates) == original


def test_carried_is_idempotent() -> None:
    grid = _hours(48)
    once = carried(frame(("BTC",), [[1.0], [2.0]]), grid)
    assert carried(once, grid) == once


def test_carried_is_indifferent_to_the_order_the_frames_rows_arrive_in() -> None:
    sorted_frame = frame(("BTC",), [[1.0], [2.0]])
    shuffled = WeightFrame(
        dates=sorted_frame.dates[::-1],
        assets=sorted_frame.assets,
        values=sorted_frame.values[::-1],
    )
    grid = _hours(48)
    assert carried(shuffled, grid) == carried(sorted_frame, grid)


def test_carried_onto_an_empty_grid_is_an_empty_frame() -> None:
    held = carried(frame(("BTC",), [[1.0]]), _hours(0))
    assert held.shape == (0, 1)


def test_carried_from_a_frame_with_no_rows_is_all_missing_on_the_grid() -> None:
    empty = WeightFrame(
        dates=_hours(0), assets=("BTC",), values=np.empty((0, 1), dtype=np.float64)
    )
    held = carried(empty, _hours(3))
    assert np.array_equal(held.values, np.full((3, 1), np.nan), equal_nan=True)
