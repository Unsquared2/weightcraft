"""Invariances every function must hold, whatever you feed it.

The pattern is empyrical's: rather than pin a number, state the relationship
that must survive a change of units, an origin shift, or a reordering. These
outlive a refactor and a change of algorithm in a way a golden array does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from canonical import (
    FLAT_LINE,
    MIXED,
    NOISE,
    NON_EMPTY,
    POSITIVE_LINE,
    SPARSE_NOISE,
    correlated_panel,
    panel,
    returns_panel,
)
from weightcraft.align import align
from weightcraft.band import no_trade_band
from weightcraft.combine import (
    nanmean_stack,
    nanmedian_stack,
    weighted_nanmean_stack,
)
from weightcraft.costs import apply_costs, book_returns, lagged, turnover
from weightcraft.cross_section import (
    project_out_rows,
    row_rank_pct,
    standardize_rows,
    top_n_mask,
)
from weightcraft.frame import WeightFrame
from weightcraft.metrics import (
    beta,
    cagr,
    compounded,
    drawdown,
    max_drawdown,
    sharpe,
    to_prices,
)
from weightcraft.normalize import (
    capped,
    center,
    clip_allocation,
    gross,
    net,
    normalised_share,
    quantize,
    rescaled_to_held_count,
    tilt,
    to_gross,
)
from weightcraft.risk import (
    EqualRiskConfig,
    VolatilityTargetConfig,
    equal_risk_row,
    equal_risk_weights,
    inverse_volatility_share,
    observed_covariance,
    penalised_for_coverage,
    rolling_sums,
    trailing_std,
    volatility_target,
)
from weightcraft.smoothing import ewm_mean, lag_rows, rolling_mean

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix, Vector

SERIES = pytest.mark.parametrize(
    "series", NON_EMPTY.values(), ids=list(NON_EMPTY.keys())
)


# --------------------------------------------------------------------------
# Scale invariance: a change of units must not change a ratio or a share.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [0.5, 2.0, 100.0])
def test_sharpe_does_not_care_what_units_the_returns_are_in(factor: float) -> None:
    assert sharpe(NOISE * factor) == pytest.approx(sharpe(NOISE))


@pytest.mark.parametrize("factor", [0.5, 2.0, 100.0])
def test_a_change_of_units_does_not_change_the_inverse_vol_shares(
    factor: float,
) -> None:
    deviation: Matrix = np.asarray([[0.01, 0.03, 0.05]])
    side: Matrix = np.ones((1, 3))
    assert np.allclose(
        inverse_volatility_share(deviation * factor, side),
        inverse_volatility_share(deviation, side),
    )


@pytest.mark.parametrize("factor", [0.5, 2.0, 10.0])
def test_a_change_of_units_does_not_change_the_equal_risk_weights(
    factor: float,
) -> None:
    window = returns_panel(3, 300, [0.01, 0.03, 0.05])
    assert np.allclose(
        equal_risk_row(window * factor, np.ones(3)),
        equal_risk_row(window, np.ones(3)),
        atol=1e-9,
    )


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_a_change_of_units_does_not_change_a_normalised_share(factor: float) -> None:
    measure: Matrix = np.asarray([[1.0, 3.0, 6.0]])
    holdings = np.ones((1, 3), dtype=np.bool_)
    assert np.allclose(
        normalised_share(measure * factor, holdings, power=1.0),
        normalised_share(measure, holdings, power=1.0),
    )


@pytest.mark.parametrize("factor", [0.25, 4.0])
def test_a_change_of_units_does_not_change_a_gross_normalised_book(
    factor: float,
) -> None:
    book: Matrix = np.asarray([[0.2, -0.3, 0.5]])
    assert np.allclose(to_gross(book * factor, 1.0), to_gross(book, 1.0))


@pytest.mark.parametrize("factor", [2.0, 0.5])
def test_trailing_std_scales_with_its_input(factor: float) -> None:
    assert np.allclose(
        trailing_std(NOISE * factor, 30),
        trailing_std(NOISE, 30) * factor,
        equal_nan=True,
    )


@pytest.mark.parametrize("factor", [0.25, 4.0])
def test_capping_commutes_with_a_change_of_units(factor: float) -> None:
    book: Matrix = np.asarray([[0.6, -0.9, 0.3]])
    scaled_then_capped = capped(
        book * factor, max_gross=0.5 * factor, max_net=0.2 * factor
    )
    capped_then_scaled = capped(book, max_gross=0.5, max_net=0.2) * factor
    assert np.allclose(scaled_then_capped, capped_then_scaled)


@pytest.mark.parametrize("factor", [2.0, 5.0])
def test_turnover_scales_with_the_book(factor: float) -> None:
    book: Matrix = np.asarray([[0.2], [0.5], [0.1]])
    assert np.allclose(turnover(book * factor), turnover(book) * factor)


@pytest.mark.parametrize("band", [0.05, 0.3, 1.0, 3.0])
def test_banding_never_trades_more_than_the_raw_target_did(band: float) -> None:
    # A band can only hold a move open; it can never manufacture a trade the
    # raw target itself did not already make.
    target: Matrix = panel(NOISE[:120], SPARSE_NOISE[:120]) * 0.1
    assert turnover(no_trade_band(target, band)).sum() <= turnover(target).sum()


def test_a_band_wide_enough_freezes_the_book_once_entries_stop() -> None:
    # Once every column has entered, an enormous band must hold every
    # position at its first level for the rest of the panel.
    generator = np.random.default_rng(7)
    target: Matrix = generator.normal(size=(50, 4)) * 0.1
    held = no_trade_band(target, 1e9)
    assert np.array_equal(held[10:], np.broadcast_to(held[9], held[10:].shape))


@pytest.mark.parametrize("factor", [2.0, 0.1])
def test_a_quieter_book_asks_for_proportionally_more_leverage(
    factor: float,
) -> None:
    config = VolatilityTargetConfig(period=60, target=0.30, max_leverage=1e9)
    louder = volatility_target(NOISE * factor, config)
    assert louder[-1] == pytest.approx(volatility_target(NOISE, config)[-1] / factor)


# --------------------------------------------------------------------------
# Translation invariance: moving the origin must not change a dispersion.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [1.0, 1e6, 1e9])
def test_shifting_the_level_does_not_change_the_variation(shift: float) -> None:
    # The property the old cumulative-sum implementation failed: at 1e9 it
    # reported a confident zero.
    assert np.allclose(
        trailing_std(NOISE + shift, 30), trailing_std(NOISE, 30), equal_nan=True
    )


@pytest.mark.parametrize("shift", [0.1, -0.1])
def test_shifting_a_benchmark_and_its_asset_together_leaves_beta_alone(
    shift: float,
) -> None:
    underlying = NOISE
    asset = 2.0 * NOISE + 0.001
    assert beta(asset + shift, underlying + shift) == pytest.approx(
        beta(asset, underlying), rel=1e-6
    )


@pytest.mark.parametrize("shift", [0.001, 0.01])
def test_translating_returns_upward_can_only_improve_the_drawdown(
    shift: float,
) -> None:
    # empyrical's max-drawdown translation property: shifting up moves the
    # worst drawdown toward zero, and it can never exceed zero.
    assert max_drawdown(MIXED) <= max_drawdown(MIXED + shift) <= 0.0


@pytest.mark.parametrize("shift", [1.0, 10.0])
def test_shifting_a_row_does_not_change_its_z_score(shift: float) -> None:
    values: Matrix = np.asarray([[1.0, 2.0, 5.0], [np.nan, 4.0, 6.0]])
    assert np.allclose(
        standardize_rows(values + shift), standardize_rows(values), equal_nan=True
    )


@pytest.mark.parametrize("shift", [3.0, -3.0])
def test_shifting_a_row_does_not_change_its_ranking(shift: float) -> None:
    values: Matrix = np.asarray([[1.0, 5.0, 3.0, np.nan]])
    assert np.allclose(
        row_rank_pct(values + shift), row_rank_pct(values), equal_nan=True
    )


def test_any_increasing_transform_leaves_the_ranking_alone() -> None:
    values: Matrix = np.asarray([[0.1, 0.5, 0.3, 0.9]])
    for transform in (np.exp, np.sqrt, lambda x: x**3):
        assert np.allclose(row_rank_pct(transform(values)), row_rank_pct(values))


# --------------------------------------------------------------------------
# Idempotence: applying a normalisation twice must change nothing.
# --------------------------------------------------------------------------


def test_normalising_a_book_twice_is_the_same_as_once() -> None:
    book: Matrix = np.asarray([[0.2, -0.3, 0.5], [1.0, 1.0, -1.0]])
    once = to_gross(book, 1.0)
    assert np.allclose(to_gross(once, 1.0), once)


def test_centering_a_book_twice_is_the_same_as_once() -> None:
    book: Matrix = np.asarray([[0.6, 0.2, 0.2], [np.nan, 1.0, 3.0]])
    once = center(book)
    assert np.allclose(center(once), once, equal_nan=True)


def test_capping_a_book_twice_is_the_same_as_once() -> None:
    book: Matrix = np.asarray([[0.9, -0.9, 0.05]])
    once = clip_allocation(book, 0.1)
    assert np.allclose(clip_allocation(once, 0.1), once)


def test_quantising_a_book_twice_is_the_same_as_once() -> None:
    book: Matrix = np.asarray([[0.117, -0.123, 0.0]])
    once = quantize(book, 0.05)
    assert np.allclose(quantize(once, 0.05), once)


def test_exposure_capping_a_book_twice_is_the_same_as_once() -> None:
    book: Matrix = np.asarray([[0.9, -0.9, 0.05]])
    once = capped(book, max_gross=0.5, max_net=0.2)
    twice = capped(once, max_gross=0.5, max_net=0.2)
    assert np.allclose(twice, once)


@pytest.mark.parametrize("tighter", [0.4, 0.1])
def test_a_tighter_limit_never_leaves_a_larger_book(tighter: float) -> None:
    book: Matrix = np.asarray([[0.9, -0.9, 0.2]])
    loose = gross(capped(book, max_gross=0.5))
    tight = gross(capped(book, max_gross=tighter))
    assert tight[0, 0] <= loose[0, 0]


def test_a_gross_target_survives_a_cap_that_does_not_bind() -> None:
    book: Matrix = np.asarray([[0.2, -0.3, 0.5]])
    on_target = to_gross(book, 1.0)
    assert np.allclose(capped(on_target, max_gross=10.0, max_net=10.0), on_target)


def test_capping_a_centred_book_leaves_it_centred() -> None:
    # Centering already forces net to zero, so only the gross term of the cap
    # can ever bind here.
    book: Matrix = np.asarray([[0.9, -0.3, -0.6, 1.0]])
    centred = center(book)
    result = capped(centred, max_gross=0.5, max_net=0.1)
    assert net(result)[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_standardising_a_row_twice_is_the_same_as_once() -> None:
    values: Matrix = np.asarray([[1.0, 2.0, 5.0, 9.0]])
    once = standardize_rows(values)
    assert np.allclose(standardize_rows(once), once)


def test_rescaling_a_side_twice_is_the_same_as_once() -> None:
    side: Matrix = np.asarray([[1.0, 3.0, np.nan]])
    once = rescaled_to_held_count(side)
    assert np.allclose(rescaled_to_held_count(once), once, equal_nan=True)


def test_projecting_out_a_control_twice_removes_nothing_further() -> None:
    generator = np.random.default_rng(4)
    controls = generator.normal(size=(1, 1, 30))
    target: Matrix = generator.normal(size=(1, 30))
    once = project_out_rows(target, controls)
    assert np.allclose(project_out_rows(once, controls), once, atol=1e-9)


# --------------------------------------------------------------------------
# Composition: two applications must equal one of the combined size.
# --------------------------------------------------------------------------


def test_two_tilts_are_one_tilt_of_their_sum() -> None:
    book: Matrix = np.asarray([[0.5, -0.5]])
    assert np.allclose(tilt(tilt(book, 0.1), 0.2), tilt(book, 0.30000000000000004))


def test_two_lags_are_one_lag_of_their_sum() -> None:
    values: Matrix = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    assert np.array_equal(
        lag_rows(lag_rows(values, 1), 2), lag_rows(values, 3), equal_nan=True
    )


def test_lagging_by_nothing_changes_nothing() -> None:
    values: Matrix = np.asarray([[1.0], [2.0]])
    assert np.array_equal(lag_rows(values, 0), values)


# --------------------------------------------------------------------------
# Order invariance: a reordering of inputs must not change a result.
# --------------------------------------------------------------------------


def test_the_order_the_assets_are_listed_in_does_not_change_the_sizing() -> None:
    deviation: Matrix = np.asarray([[0.01, 0.05, 0.02]])
    side: Matrix = np.ones((1, 3))
    forward = inverse_volatility_share(deviation, side)
    order = [2, 0, 1]
    backward = inverse_volatility_share(deviation[:, order], side[:, order])
    assert np.allclose(backward, forward[:, order])


def test_the_order_the_assets_are_listed_in_does_not_change_a_cap() -> None:
    book: Matrix = np.asarray([[0.9, -0.9, 0.2]])
    order = [2, 0, 1]
    forward = capped(book, max_gross=0.5, max_net=0.1)
    backward = capped(book[:, order], max_gross=0.5, max_net=0.1)
    assert np.allclose(backward, forward[:, order])


def test_the_order_the_assets_are_listed_in_does_not_change_equal_risk() -> None:
    window = returns_panel(6, 400, [0.01, 0.03, 0.05])
    order = [2, 0, 1]
    forward = equal_risk_row(window, np.ones(3))
    backward = equal_risk_row(window[:, order], np.ones(3))
    assert np.allclose(backward, forward[order], atol=1e-9)


def test_the_order_the_frames_arrive_in_does_not_change_the_mean() -> None:
    first = WeightFrame.from_rows(
        ["2026-01-01"], ("A", "B"), np.asarray([[1.0, np.nan]])
    )
    second = WeightFrame.from_rows(["2026-01-01"], ("B", "C"), np.asarray([[2.0, 3.0]]))
    forward = nanmean_stack(align([first, second]).values)
    backward = nanmean_stack(align([second, first]).values)
    assert np.array_equal(forward, backward, equal_nan=True)


# --------------------------------------------------------------------------
# Bounds and structural facts.
# --------------------------------------------------------------------------


@SERIES
def test_a_drawdown_is_never_positive(series: Vector) -> None:
    assert max_drawdown(series) <= 0.0
    below = drawdown(series)
    assert np.all(below[np.isfinite(below)] <= 1e-12)


def test_a_book_that_only_rises_never_draws_down() -> None:
    assert max_drawdown(POSITIVE_LINE) == pytest.approx(0.0)


@SERIES
def test_the_mean_of_a_stack_lies_between_its_smallest_and_largest(
    series: Vector,
) -> None:
    stack = np.stack([panel(series), panel(series * 2.0)])
    combined = nanmean_stack(stack)
    usable = np.where(np.isfinite(stack), stack, np.nan)
    seen = np.isfinite(combined)
    low = np.min(np.where(np.isfinite(usable), usable, np.inf), axis=0)
    high = np.max(np.where(np.isfinite(usable), usable, -np.inf), axis=0)
    assert np.all(combined[seen] >= low[seen] - 1e-12)
    assert np.all(combined[seen] <= high[seen] + 1e-12)


def test_a_smoothed_series_never_leaves_the_range_it_smoothed() -> None:
    values: Matrix = np.asarray([[1.0], [5.0], [3.0], [9.0], [2.0]])
    for smoothed in (rolling_mean(values, 3), ewm_mean(values, 3)):
        seen = np.isfinite(smoothed)
        assert np.all(smoothed[seen] >= values.min())
        assert np.all(smoothed[seen] <= values.max())


def test_a_percentile_rank_lies_in_the_unit_interval() -> None:
    ranked = row_rank_pct(np.asarray([[3.0, 1.0, 2.0, np.nan]]))
    seen = np.isfinite(ranked)
    assert np.all(ranked[seen] > 0.0)
    assert np.all(ranked[seen] <= 1.0)


def test_a_covariance_matrix_is_symmetric_with_and_without_gaps() -> None:
    complete = returns_panel(8, 60, [0.01, 0.02, 0.03])
    gapped = complete.copy()
    gapped[5:9, 1] = np.nan
    for window in (complete, gapped):
        covariance = observed_covariance(window)
        assert np.allclose(covariance, covariance.T)


def test_the_diagonal_of_a_covariance_matrix_is_the_variances() -> None:
    window = returns_panel(9, 80, [0.01, 0.04])
    covariance = observed_covariance(window)
    assert np.allclose(np.diag(covariance), np.var(window, axis=0, ddof=1))


def test_the_coverage_penalty_leaves_the_correlations_alone() -> None:
    window = correlated_panel(10, 300, rho=0.8)
    covariance = observed_covariance(window)
    penalised = penalised_for_coverage(covariance, np.asarray([0.3, 1.0, 1.0]))

    def correlation(matrix: Matrix) -> Matrix:
        deviation = np.sqrt(np.diag(matrix))
        out: Matrix = matrix / np.outer(deviation, deviation)
        return out

    assert np.allclose(correlation(penalised), correlation(covariance), atol=1e-9)


def test_equal_risk_weights_are_positive_and_sum_to_one() -> None:
    weights = equal_risk_row(returns_panel(12, 400, [0.01, 0.03, 0.05]), np.ones(3))
    assert np.all(weights > 0.0)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_equal_risk_gives_identical_assets_identical_weights() -> None:
    column = returns_panel(13, 300, [0.02])
    window = np.column_stack([column, column, column])
    weights = equal_risk_row(window, np.ones(3))
    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=0.02)


def test_shrinking_all_the_way_to_the_diagonal_gives_inverse_volatility() -> None:
    # With every correlation shrunk away, equal risk contribution *is*
    # inverse-volatility weighting, so the two must agree.
    window = correlated_panel(14, 800, rho=0.9)
    weights = equal_risk_row(window, np.ones(3), shrinkage=1.0)
    deviations = np.std(window, axis=0, ddof=1)
    inverse = (1.0 / deviations) / (1.0 / deviations).sum()
    assert np.allclose(weights, inverse, atol=0.01)


def test_the_more_correlated_a_pair_the_less_of_the_book_it_gets() -> None:
    weights = [
        equal_risk_row(correlated_panel(15, 800, rho=rho), np.ones(3), shrinkage=0.0)
        for rho in (0.0, 0.95)
    ]
    assert weights[1][0] + weights[1][1] < weights[0][0] + weights[0][1]


def test_a_free_book_earns_more_than_a_costly_one() -> None:
    weights: Matrix = np.asarray([[0.0], [1.0], [0.0], [1.0]])
    returns: Matrix = np.full((4, 1), 0.01)
    earned = [
        float(np.nansum(book_returns(weights, returns, cost=charge)))
        for charge in (0.0, 0.001, 0.01)
    ]
    assert earned[0] > earned[1] > earned[2]


def test_a_book_that_never_moves_pays_no_cost() -> None:
    weights: Matrix = np.full((5, 2), 0.5)
    returns: Matrix = np.zeros((5, 2))
    assert float(np.nansum(book_returns(weights, returns, cost=0.05))) == 0.0


def test_a_lag_delays_when_a_position_starts_earning() -> None:
    weights: Matrix = np.asarray([[1.0], [1.0], [1.0], [1.0]])
    returns: Matrix = np.asarray([[0.0], [0.1], [0.2], [0.3]])
    without = apply_costs(weights, returns, cost=0.0, lag=0)
    delayed = apply_costs(weights, returns, cost=0.0, lag=1)
    assert without[1, 0] == pytest.approx(0.1)
    assert without[2, 0] == pytest.approx(0.2)
    # A position that did not exist yet earns nothing, rather than nothing
    # *known* -- so the row it is charged on is still billable.
    assert delayed[1, 0] == 0.0
    assert delayed[2, 0] == pytest.approx(0.2)


def test_a_cost_free_book_earns_exactly_its_lagged_weight_times_the_return() -> None:
    weights: Matrix = np.asarray([[0.5], [0.5], [0.5]])
    returns: Matrix = np.asarray([[0.1], [0.2], [0.3]])
    earned = lagged(weights, 0) * returns
    assert np.allclose(
        apply_costs(weights, returns, cost=0.0),
        np.where(np.isfinite(earned), earned, 0.0),
    )


def test_the_top_n_mask_never_selects_more_than_there_are() -> None:
    values: Matrix = np.asarray([[3.0, np.nan, 1.0]])
    assert int(top_n_mask(values, 10).sum()) == 2
    assert int(top_n_mask(values, 1).sum()) == 1


def test_a_median_of_two_frames_is_their_mean() -> None:
    stack = np.stack([panel(MIXED), panel(NOISE[: MIXED.size])])
    assert np.allclose(nanmedian_stack(stack), nanmean_stack(stack))


def test_giving_one_source_the_whole_share_returns_that_source() -> None:
    stack = np.stack([panel(MIXED), panel(FLAT_LINE)])
    shares: Vector = np.asarray([1.0, 0.0])
    assert np.allclose(
        weighted_nanmean_stack(stack, shares), panel(MIXED), equal_nan=True
    )


def test_rolling_sums_are_linear() -> None:
    first, second = NOISE[:60], SPARSE_NOISE[:60]
    combined = rolling_sums(first + second, 5)
    separate = rolling_sums(first, 5) + rolling_sums(second, 5)
    assert np.allclose(combined, separate, equal_nan=True)


def test_a_rolling_mean_of_a_constant_is_that_constant() -> None:
    assert np.allclose(rolling_mean(np.full((10, 2), 7.0), 4)[3:], 7.0)
    assert np.allclose(ewm_mean(np.full((10, 2), 7.0), 4)[3:], 7.0)


def test_a_window_of_one_is_the_series_itself() -> None:
    values: Matrix = np.asarray([[1.0], [2.0], [3.0]])
    assert np.array_equal(rolling_mean(values, 1), values)
    assert np.array_equal(ewm_mean(values, 1), values)


def test_a_smoother_is_affine() -> None:
    values: Matrix = np.asarray([[1.0], [4.0], [2.0], [8.0]])
    assert np.allclose(
        rolling_mean(3.0 * values + 5.0, 2),
        3.0 * rolling_mean(values, 2) + 5.0,
        equal_nan=True,
    )


def test_compounding_is_associative_across_a_split() -> None:
    first, second = MIXED[:10], MIXED[10:]
    whole = compounded(MIXED)
    parts = (1.0 + compounded(first)) * (1.0 + compounded(second)) - 1.0
    assert whole == pytest.approx(parts)


def test_a_constant_return_compounds_to_its_own_growth_rate() -> None:
    daily = 0.001
    series: Vector = np.full(365, daily)
    assert cagr(series, 365.0) == pytest.approx((1.0 + daily) ** 365 - 1.0)


def test_prices_rise_exactly_as_the_returns_say() -> None:
    prices = to_prices(np.asarray([0.1, 0.1, -0.5]))
    assert prices.tolist() == pytest.approx([1.1, 1.21, 0.605])


def test_a_beta_against_itself_is_one_and_against_a_multiple_is_that_multiple() -> None:
    assert beta(NOISE, NOISE) == pytest.approx(1.0)
    assert beta(3.0 * NOISE, NOISE) == pytest.approx(3.0)


def test_a_beta_against_something_unrelated_is_about_nothing() -> None:
    # Same scale as the benchmark, or "beta near zero" would just be measuring
    # the ratio of their standard deviations.
    benchmark: Vector = np.random.default_rng(41).normal(size=4000) * 0.01
    other: Vector = np.random.default_rng(99).normal(size=4000) * 0.01
    assert abs(beta(other, benchmark)) < 0.1


def test_more_noise_means_a_worse_sharpe() -> None:
    quiet: Vector = 0.001 + np.random.default_rng(2).normal(size=500) * 0.001
    loud: Vector = 0.001 + np.random.default_rng(2).normal(size=500) * 0.01
    assert sharpe(quiet) > sharpe(loud)


def test_annualising_a_sharpe_scales_it_by_the_square_root_of_the_period() -> None:
    assert sharpe(NOISE, 252) == pytest.approx(sharpe(NOISE, 1) * np.sqrt(252))


def test_a_book_already_at_its_target_is_left_alone() -> None:
    steady: Vector = np.full(200, 0.0)
    steady[1::2] = 0.02
    steady[::2] = -0.02
    realised = float(np.std(steady[-60:], ddof=1)) * np.sqrt(365)
    scale = volatility_target(
        steady, VolatilityTargetConfig(period=60, target=realised)
    )
    assert scale[-1] == pytest.approx(1.0, rel=1e-6)


def test_equal_risk_weights_across_a_panel_sum_to_the_held_count() -> None:
    returns = returns_panel(16, 60, [0.01, 0.02, 0.03])
    holdings = np.ones((60, 3), dtype=np.bool_)
    sized = equal_risk_weights(returns, holdings, EqualRiskConfig(period=30))
    assert float(sized[-1].sum()) == pytest.approx(3.0)


def test_a_name_that_is_not_held_is_never_sized() -> None:
    returns = returns_panel(17, 60, [0.01, 0.02, 0.03])
    holdings = np.ones((60, 3), dtype=np.bool_)
    holdings[:, 2] = False
    sized = equal_risk_weights(returns, holdings, EqualRiskConfig(period=30))
    assert float(np.abs(sized[:, 2]).sum()) == 0.0


def test_a_book_of_one_asset_is_entirely_that_asset() -> None:
    window = returns_panel(18, 50, [0.02, 0.02])
    assert equal_risk_row(window, np.asarray([1.0, 0.0])).tolist() == [1.0, 0.0]


def test_gross_is_unchanged_by_flipping_every_sign() -> None:
    book: Matrix = np.asarray([[0.2, -0.3, np.nan]])
    assert np.array_equal(gross(-book), gross(book))
