from __future__ import annotations

import numpy as np
import pytest

from weightcraft.costs import apply_costs, book_returns, lagged, turnover
from weightcraft.smoothing import ewm_mean, lag_rows, rolling_mean


def test_the_first_row_has_no_turnover_because_nothing_moved_into_it() -> None:
    assert turnover(np.asarray([[0.5], [0.5]])).tolist() == [[0.0], [0.0]]


def test_trading_into_and_out_of_a_gap_costs_what_it_should() -> None:
    # An asset a source does not cover is a gap, and `align` produces one for
    # every uncovered cell -- so blanking these made the cost knob a no-op.
    moved = turnover(np.asarray([[np.nan], [1.0], [np.nan], [2.0]]))
    assert moved[:, 0].tolist() == [0.0, 1.0, 1.0, 2.0]


def test_turnover_is_the_absolute_change_in_a_position() -> None:
    moved = turnover(np.asarray([[0.2], [0.5], [0.1]]))
    assert moved[:, 0].tolist() == pytest.approx([0.0, 0.3, 0.4])


def test_turnover_never_reports_a_missing_value() -> None:
    assert np.isfinite(turnover(np.asarray([[np.nan], [0.5], [np.nan]]))).all()


def test_an_infinite_position_is_missing_rather_than_enormous() -> None:
    # Non-finite is missing everywhere in this library, and a missing position
    # is a flat one. Letting an infinity through gave `inf - inf` on the next
    # row -- NaN, i.e. a broken weight that cost nothing at all.
    assert turnover(np.asarray([[0.0], [np.inf]]))[1, 0] == 0.0
    assert turnover(np.asarray([[np.inf], [np.inf]]))[1, 0] == 0.0


def test_a_weight_earns_the_return_of_the_date_after_it() -> None:
    shifted = lagged(np.asarray([[1.0], [2.0], [3.0]]), 0)
    assert np.isnan(shifted[0, 0])
    assert shifted[1:].tolist() == [[1.0], [2.0]]


def test_an_extra_lag_delays_the_book_one_more_row() -> None:
    shifted = lagged(np.asarray([[1.0], [2.0], [3.0]]), 1)
    assert np.isnan(shifted[:2]).all()
    assert shifted[2, 0] == 1.0


def test_a_lag_longer_than_the_history_leaves_nothing() -> None:
    assert np.isnan(lagged(np.asarray([[1.0]]), 5)).all()


def test_a_negative_lag_is_refused() -> None:
    with pytest.raises(ValueError, match="not be negative"):
        lagged(np.asarray([[1.0]]), -1)


def test_a_position_earns_its_lagged_weight_less_what_the_change_cost() -> None:
    weights = np.asarray([[0.0], [1.0], [1.0]])
    returns = np.asarray([[0.0], [0.10], [0.10]])
    earned = apply_costs(weights, returns, cost=0.01)
    assert earned[2, 0] == pytest.approx(1.0 * 0.10 - 0.0)
    assert earned[1, 0] == pytest.approx(0.0 * 0.10 - 0.01)


def test_a_free_book_is_charged_nothing() -> None:
    weights = np.asarray([[0.0], [1.0]])
    returns = np.asarray([[0.0], [0.10]])
    assert apply_costs(weights, returns, cost=0.0)[1, 0] == pytest.approx(0.0)


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="disagree"):
        apply_costs(np.zeros((2, 1)), np.zeros((3, 1)), cost=0.0)


def test_a_negative_cost_is_refused() -> None:
    with pytest.raises(ValueError, match="not be negative"):
        apply_costs(np.zeros((2, 1)), np.zeros((2, 1)), cost=-0.01)


def test_a_row_with_nothing_in_it_sums_to_zero_rather_than_missing() -> None:
    weights = np.asarray([[np.nan], [np.nan]])
    returns = np.asarray([[np.nan], [np.nan]])
    assert book_returns(weights, returns).tolist() == [0.0, 0.0]


def test_the_book_return_is_the_row_sum_of_its_positions() -> None:
    weights = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    returns = np.asarray([[0.0, 0.0], [0.10, 0.20]])
    assert book_returns(weights, returns)[1] == pytest.approx(0.15)


def test_a_rolling_mean_says_nothing_until_its_window_has_filled() -> None:
    smoothed = rolling_mean(np.asarray([[1.0], [2.0], [3.0]]), 2)
    assert np.isnan(smoothed[0, 0])
    assert smoothed[1:].tolist() == [[1.5], [2.5]]


def test_a_window_holding_a_gap_stays_silent_rather_than_averaging_one_point() -> None:
    smoothed = rolling_mean(np.asarray([[1.0], [np.nan], [3.0]]), 2)
    assert np.isnan(smoothed[2, 0])


def test_a_window_holding_only_gaps_stays_missing() -> None:
    smoothed = rolling_mean(np.asarray([[np.nan], [np.nan], [1.0]]), 2)
    assert np.isnan(smoothed[1, 0])


def test_an_ewm_mean_also_waits_for_a_full_window() -> None:
    smoothed = ewm_mean(np.asarray([[1.0], [2.0], [3.0]]), 3)
    assert np.isnan(smoothed[:2]).all()
    assert np.isfinite(smoothed[2, 0])


def test_an_ewm_mean_leans_on_the_most_recent_observation() -> None:
    smoothed = ewm_mean(np.asarray([[1.0], [1.0], [10.0]]), 2)
    plain = rolling_mean(np.asarray([[1.0], [1.0], [10.0]]), 2)
    assert smoothed[2, 0] > plain[2, 0]


def test_an_ewm_mean_of_a_constant_series_is_that_constant() -> None:
    smoothed = ewm_mean(np.full((10, 1), 3.0), 4)
    assert smoothed[-1, 0] == pytest.approx(3.0)


def test_a_gap_dilutes_an_ewm_mean_rather_than_restarting_it() -> None:
    smoothed = ewm_mean(np.asarray([[1.0], [np.nan], [1.0], [1.0]]), 2)
    assert smoothed[3, 0] == pytest.approx(1.0)


def test_a_smoother_refuses_a_zero_window() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        rolling_mean(np.zeros((2, 1)), 0)
    with pytest.raises(ValueError, match="at least 1"):
        ewm_mean(np.zeros((2, 1)), 0)


def test_lagging_by_nothing_returns_the_input() -> None:
    values = np.asarray([[1.0], [2.0]])
    assert lag_rows(values, 0) is values


def test_lagging_leaves_the_head_missing() -> None:
    shifted = lag_rows(np.asarray([[1.0], [2.0]]), 1)
    assert np.isnan(shifted[0, 0])
    assert shifted[1, 0] == 1.0


def test_lagging_past_the_history_leaves_nothing() -> None:
    assert np.isnan(lag_rows(np.asarray([[1.0]]), 3)).all()


def test_a_negative_lag_of_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="not be negative"):
        lag_rows(np.zeros((2, 1)), -1)
