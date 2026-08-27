from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from weightcraft.risk import (
    EqualRiskConfig,
    VolatilityTargetConfig,
    downside_deviation,
    equal_risk_row,
    equal_risk_weights,
    inverse_volatility_share,
    observed_covariance,
    penalised_for_coverage,
    rolling_sums,
    stack_volatility,
    trailing_std,
    volatility_target,
)

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix, Vector

RISK_CONTRIBUTION_TOLERANCE = 0.05


def _returns(seed: int, rows: int, scales: list[float]) -> Matrix:
    generator = np.random.default_rng(seed)
    out: Matrix = generator.normal(size=(rows, len(scales))) * np.asarray(scales)
    return out


def _risk_contributions(covariance: Matrix, weights: Vector) -> Vector:
    marginal = covariance @ weights
    contributions: Vector = weights * marginal
    return contributions / contributions.sum()


def test_rolling_sums_leave_the_head_missing_until_the_window_fills() -> None:
    sums = rolling_sums(np.asarray([1.0, 2.0, 3.0]), 2)
    assert np.isnan(sums[0])
    assert sums[1:].tolist() == [3.0, 5.0]


def test_rolling_sums_refuse_a_zero_window() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        rolling_sums(np.asarray([1.0]), 0)


def test_trailing_std_matches_the_sample_standard_deviation() -> None:
    series = np.asarray([1.0, 2.0, 4.0, 8.0])
    assert trailing_std(series, 3)[2] == pytest.approx(
        float(np.std(series[:3], ddof=1))
    )


def test_trailing_std_of_a_flat_window_is_exactly_zero() -> None:
    assert trailing_std(np.asarray([2.0, 2.0, 2.0]), 3)[2] == 0.0


def test_trailing_std_needs_two_observations_in_the_window() -> None:
    series = np.asarray([np.nan, np.nan, 1.0, 2.0])
    assert np.isnan(trailing_std(series, 2)[1])
    assert np.isfinite(trailing_std(series, 2)[3])


def test_trailing_std_refuses_a_window_of_one() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        trailing_std(np.asarray([1.0]), 1)


def test_downside_deviation_ignores_the_upside_the_book_is_positioned_for() -> None:
    calm = np.asarray([[0.01], [0.01], [-0.01]])
    wild_upside = np.asarray([[0.50], [0.50], [-0.01]])
    assert downside_deviation(calm, 3)[2, 0] == pytest.approx(
        downside_deviation(wild_upside, 3)[2, 0]
    )


def test_downside_deviation_is_missing_when_nothing_lost() -> None:
    assert np.isnan(downside_deviation(np.asarray([[0.1], [0.1]]), 2)[1, 0])


def test_inverse_volatility_sizes_the_quietest_name_the_largest() -> None:
    sized = inverse_volatility_share(
        np.asarray([[0.01, 0.05]]), np.asarray([[1.0, 1.0]])
    )
    assert sized[0, 0] > sized[0, 1]


def test_inverse_volatility_sums_to_the_held_count() -> None:
    sized = inverse_volatility_share(
        np.asarray([[0.01, 0.05, np.nan]]), np.asarray([[1.0, 1.0, np.nan]])
    )
    assert np.nansum(sized) == pytest.approx(2.0)


def test_a_name_the_book_does_not_hold_stays_missing() -> None:
    sized = inverse_volatility_share(
        np.asarray([[0.01, 0.05]]), np.asarray([[1.0, np.nan]])
    )
    assert np.isnan(sized[0, 1])


def test_a_zero_volatility_name_is_zeroed_rather_than_infinite() -> None:
    sized = inverse_volatility_share(
        np.asarray([[0.0, 0.05]]), np.asarray([[1.0, 1.0]])
    )
    assert np.isfinite(sized).all()


def test_pairwise_covariance_survives_a_gap_that_would_void_a_row() -> None:
    window = np.asarray([[1.0, 2.0], [2.0, np.nan], [3.0, 6.0], [4.0, 8.0]])
    covariance = observed_covariance(window)
    assert np.isfinite(covariance).all()
    assert covariance[0, 1] > 0.0


def test_a_complete_window_matches_numpy_cov() -> None:
    window = _returns(1, 20, [0.01, 0.02])
    assert np.allclose(observed_covariance(window), np.cov(window, rowvar=False))


def test_the_coverage_penalty_is_inert_on_a_complete_window() -> None:
    covariance = np.asarray([[1.0, 0.2], [0.2, 4.0]])
    penalised = penalised_for_coverage(covariance, np.asarray([1.0, 1.0]))
    assert np.array_equal(penalised, covariance)


def test_a_barely_observed_name_is_treated_as_riskier() -> None:
    covariance = np.asarray([[1.0, 0.0], [0.0, 4.0]])
    penalised = penalised_for_coverage(covariance, np.asarray([0.2, 1.0]))
    assert penalised[0, 0] > covariance[0, 0]


def test_more_missing_history_means_more_penalty() -> None:
    covariance = np.asarray([[1.0, 0.0], [0.0, 4.0]])
    lightly = penalised_for_coverage(covariance, np.asarray([0.8, 1.0]))
    heavily = penalised_for_coverage(covariance, np.asarray([0.2, 1.0]))
    assert heavily[0, 0] > lightly[0, 0]


def test_a_degenerate_covariance_is_returned_untouched() -> None:
    covariance = np.zeros((2, 2))
    assert np.array_equal(
        penalised_for_coverage(covariance, np.asarray([0.5, 1.0])), covariance
    )


def test_equal_risk_equalises_the_risk_contributions() -> None:
    window = _returns(7, 400, [0.01, 0.03, 0.05])
    holdings = np.ones(3)
    weights = equal_risk_row(window, holdings, shrinkage=0.0)
    contributions = _risk_contributions(np.cov(window, rowvar=False), weights)
    assert np.allclose(contributions, 1.0 / 3.0, atol=RISK_CONTRIBUTION_TOLERANCE)


def test_equal_risk_reduces_to_inverse_volatility_when_nothing_correlates() -> None:
    window = _returns(11, 800, [0.01, 0.02, 0.04])
    weights = equal_risk_row(window, np.ones(3), shrinkage=0.0)
    deviations = np.std(window, axis=0, ddof=1)
    inverse = (1.0 / deviations) / (1.0 / deviations).sum()
    assert np.allclose(weights, inverse, atol=RISK_CONTRIBUTION_TOLERANCE)


def test_a_correlated_pair_splits_one_slot_between_them() -> None:
    generator = np.random.default_rng(3)
    common = generator.normal(size=600) * 0.02
    window = np.column_stack(
        [
            common + generator.normal(size=600) * 0.001,
            common + generator.normal(size=600) * 0.001,
            generator.normal(size=600) * 0.02,
        ]
    )
    weights = equal_risk_row(window, np.ones(3), shrinkage=0.0)
    assert weights[2] > weights[0]
    assert weights[0] == pytest.approx(weights[1], abs=0.05)


def test_equal_risk_holds_nothing_when_nothing_is_held() -> None:
    assert equal_risk_row(_returns(2, 10, [0.01]), np.zeros(1)).tolist() == [0.0]


def test_a_single_holding_takes_the_whole_slot() -> None:
    assert equal_risk_row(_returns(2, 10, [0.01, 0.01]), np.asarray([1.0, 0.0]))[
        0
    ] == pytest.approx(1.0)


def test_too_short_a_window_falls_back_to_equal_weights() -> None:
    weights = equal_risk_row(np.zeros((2, 2)), np.ones(2))
    assert weights.tolist() == [1.0, 1.0]


def test_a_degenerate_window_stops_iterating_rather_than_diverging() -> None:
    weights = equal_risk_row(np.zeros((10, 2)), np.ones(2))
    assert np.isfinite(weights).all()


def test_equal_risk_weights_carry_a_size_forward_between_rebalances() -> None:
    window = _returns(5, 40, [0.01, 0.03])
    holdings = np.ones((40, 2), dtype=np.bool_)
    sized = equal_risk_weights(
        window, holdings, EqualRiskConfig(period=20, rebalance=5)
    )
    assert sized[5].tolist() == sized[9].tolist()
    assert sized[5].tolist() != sized[10].tolist()


def test_equal_risk_weights_sum_to_the_held_count() -> None:
    window = _returns(5, 30, [0.01, 0.03])
    holdings = np.ones((30, 2), dtype=np.bool_)
    sized = equal_risk_weights(window, holdings, EqualRiskConfig(period=20))
    assert sized[-1].sum() == pytest.approx(2.0)


def test_equal_risk_weights_refuse_a_zero_rebalance() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        equal_risk_weights(
            np.zeros((2, 1)),
            np.ones((2, 1), dtype=np.bool_),
            EqualRiskConfig(period=2, rebalance=0),
        )


def test_volatility_targeting_levers_a_quiet_book_up_and_a_wild_one_down() -> None:
    quiet = _returns(4, 200, [0.001])[:, 0]
    wild = _returns(4, 200, [0.10])[:, 0]
    config = VolatilityTargetConfig(period=60, target=0.30)
    assert volatility_target(quiet, config)[-1] > 1.0
    assert volatility_target(wild, config)[-1] < 1.0


def test_volatility_targeting_refuses_the_leverage_a_flat_window_asks_for() -> None:
    scale = volatility_target(
        np.zeros(100), VolatilityTargetConfig(period=30, target=0.30, max_leverage=3.0)
    )
    assert scale.max() <= 3.0


def test_a_levered_book_asks_for_proportionally_less() -> None:
    book = _returns(9, 200, [0.02])[:, 0]
    unlevered = volatility_target(book, VolatilityTargetConfig(period=60, target=0.30))
    levered = volatility_target(
        book,
        VolatilityTargetConfig(period=60, target=0.30, portfolio_leverage=2.0),
    )
    assert levered[-1] == pytest.approx(unlevered[-1] / 2.0)


def test_the_scale_is_one_before_the_window_has_filled() -> None:
    scale = volatility_target(
        _returns(6, 50, [0.02])[:, 0], VolatilityTargetConfig(period=30, target=0.30)
    )
    assert scale[:29].tolist() == [1.0] * 29


def test_stack_volatility_scores_each_frame_separately() -> None:
    stack = np.stack([np.zeros((5, 2)), _returns(8, 5, [0.05, 0.05])])
    deviations = stack_volatility(stack)
    assert deviations[0] == 0.0
    assert deviations[1] > 0.0


def test_a_frame_with_nothing_in_it_scores_nothing() -> None:
    stack = np.full((1, 3, 2), np.nan)
    assert np.isnan(stack_volatility(stack)).all()
