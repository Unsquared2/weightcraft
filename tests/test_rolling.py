from __future__ import annotations

import numpy as np
import pytest

from weightcraft.rolling import bars_since_extreme, partial_rolling_mean, windowed


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
