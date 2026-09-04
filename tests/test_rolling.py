from __future__ import annotations

import numpy as np
import pytest

from weightcraft.rolling import (
    bars_since_extreme,
    partial_rolling_mean,
    partial_rolling_std,
    rolling_extreme,
    windowed,
)


def test_the_current_row_is_the_last_slice_of_the_window() -> None:
    stack = windowed(np.asarray([[1.0], [2.0], [3.0]]), 2)
    assert stack[-1, :, 0].tolist() == [1.0, 2.0, 3.0]


def test_the_oldest_slice_is_shifted_furthest() -> None:
    stack = windowed(np.asarray([[1.0], [2.0], [3.0]]), 2)
    assert np.isnan(stack[0, 0, 0])
    assert stack[0, 1:, 0].tolist() == [1.0, 2.0]


def test_a_window_refuses_to_be_shorter_than_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        windowed(np.zeros((2, 1)), 0)


def test_a_partial_mean_answers_on_the_first_row() -> None:
    mean = partial_rolling_mean(np.asarray([[1.0], [2.0], [3.0]]), 3, 1)
    assert mean[0, 0] == pytest.approx(1.0)
    assert mean[1, 0] == pytest.approx(1.5)
    assert mean[2, 0] == pytest.approx(2.0)


def test_a_partial_mean_stays_silent_below_min_periods() -> None:
    mean = partial_rolling_mean(np.asarray([[1.0], [2.0]]), 3, 2)
    assert np.isnan(mean[0, 0])
    assert mean[1, 0] == pytest.approx(1.5)


def test_a_full_min_periods_matches_a_strict_full_window() -> None:
    # min_periods == window is the same contract `smoothing.rolling_mean` makes.
    values = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    mean = partial_rolling_mean(values, 2, 2)
    assert np.isnan(mean[0, 0])
    assert mean[1:, 0].tolist() == pytest.approx([1.5, 2.5, 3.5])


def test_a_gap_inside_the_window_lowers_the_count_not_the_answer() -> None:
    mean = partial_rolling_mean(np.asarray([[1.0], [np.nan], [3.0]]), 3, 2)
    assert mean[2, 0] == pytest.approx(2.0)


def test_a_window_holding_only_gaps_stays_missing() -> None:
    mean = partial_rolling_mean(np.asarray([[np.nan], [np.nan]]), 2, 1)
    assert np.isnan(mean[1, 0])


def test_a_partial_mean_refuses_a_zero_window_or_min_periods() -> None:
    with pytest.raises(ValueError, match="window"):
        partial_rolling_mean(np.zeros((2, 1)), 0, 1)
    with pytest.raises(ValueError, match="min_periods"):
        partial_rolling_mean(np.zeros((2, 1)), 1, 0)


def test_bars_since_the_low_is_zero_on_the_bar_that_sets_it() -> None:
    age = bars_since_extreme(np.asarray([[3.0], [1.0], [2.0]]), 3, 1, lowest=True)
    assert age[1, 0] == pytest.approx(0.0)
    assert age[2, 0] == pytest.approx(1.0)


def test_bars_since_the_high_is_zero_on_the_bar_that_sets_it() -> None:
    age = bars_since_extreme(np.asarray([[1.0], [3.0], [2.0]]), 3, 1, lowest=False)
    assert age[1, 0] == pytest.approx(0.0)
    assert age[2, 0] == pytest.approx(1.0)


def test_bars_since_extreme_answers_before_the_window_is_full() -> None:
    age = bars_since_extreme(np.asarray([[5.0], [1.0]]), 5, 1, lowest=True)
    assert age[1, 0] == pytest.approx(0.0)


def test_bars_since_extreme_stays_silent_below_min_periods() -> None:
    age = bars_since_extreme(np.asarray([[5.0], [1.0]]), 5, 3, lowest=True)
    assert np.isnan(age[1, 0])


def test_bars_since_extreme_ignores_gaps_in_the_window() -> None:
    age = bars_since_extreme(np.asarray([[1.0], [np.nan], [3.0]]), 3, 1, lowest=False)
    assert age[2, 0] == pytest.approx(0.0)


def test_bars_since_extreme_refuses_a_zero_min_periods() -> None:
    with pytest.raises(ValueError, match="min_periods"):
        bars_since_extreme(np.zeros((2, 1)), 2, 0, lowest=True)


def test_a_partial_std_matches_numpy_ddof_one_on_a_full_window() -> None:
    values = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    deviation = partial_rolling_std(values, 4, 4)
    assert deviation[3, 0] == pytest.approx(float(np.std([1.0, 2.0, 3.0, 4.0], ddof=1)))


def test_a_partial_std_matches_numpy_ddof_zero() -> None:
    values = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    deviation = partial_rolling_std(values, 4, 4, ddof=0)
    assert deviation[3, 0] == pytest.approx(float(np.std([1.0, 2.0, 3.0, 4.0], ddof=0)))


def test_a_partial_std_needs_two_observations_even_if_min_periods_asks_one() -> None:
    deviation = partial_rolling_std(np.asarray([[1.0], [2.0]]), 2, 1)
    assert np.isnan(deviation[0, 0])
    assert np.isfinite(deviation[1, 0])


def test_a_partial_std_at_ddof_zero_still_answers_on_one_observation() -> None:
    # `min_periods` is satisfied at one row; the *variance* is what stays
    # missing, by the next test's rule, not the observation count.
    deviation = partial_rolling_std(np.asarray([[1.0], [7.0]]), 2, 1, ddof=0)
    assert np.isnan(deviation[0, 0])  # one point has no spread to measure
    assert np.isfinite(deviation[1, 0])


def test_a_constant_window_is_missing_rather_than_a_hard_zero() -> None:
    # A caller almost always divides by this deviation (a z-score, a Sharpe);
    # reporting an exact zero there is a divide-by-zero waiting to happen, so
    # a window with no spread at all answers missing instead.
    deviation = partial_rolling_std(np.full((3, 1), 5.0), 3, 2)
    assert np.isnan(deviation[1, 0])


def test_a_partial_std_refuses_a_negative_ddof() -> None:
    with pytest.raises(ValueError, match="ddof"):
        partial_rolling_std(np.zeros((2, 1)), 2, 1, ddof=-1)


def test_rolling_extreme_reports_the_lowest_value_in_the_window() -> None:
    lowest = rolling_extreme(np.asarray([[3.0], [1.0], [2.0]]), 3, 1, lowest=True)
    assert lowest[2, 0] == pytest.approx(1.0)


def test_rolling_extreme_reports_the_highest_value_in_the_window() -> None:
    highest = rolling_extreme(np.asarray([[1.0], [3.0], [2.0]]), 3, 1, lowest=False)
    assert highest[2, 0] == pytest.approx(3.0)


def test_rolling_extreme_answers_before_the_window_is_full() -> None:
    highest = rolling_extreme(np.asarray([[1.0], [5.0]]), 5, 1, lowest=False)
    assert highest[1, 0] == pytest.approx(5.0)


def test_rolling_extreme_stays_silent_below_min_periods() -> None:
    highest = rolling_extreme(np.asarray([[1.0], [5.0]]), 5, 3, lowest=False)
    assert np.isnan(highest[1, 0])


def test_rolling_extreme_ignores_gaps_in_the_window() -> None:
    highest = rolling_extreme(np.asarray([[1.0], [np.nan], [3.0]]), 3, 1, lowest=False)
    assert highest[2, 0] == pytest.approx(3.0)
