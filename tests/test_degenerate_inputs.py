"""Every public function, swept over the inputs that break naive code.

Empty, one row, one column, constant, all-missing, infinite. The rule the whole
library follows is that none of these may raise, warn, or return a number that
looks settled when it is not -- and the suite runs under
`filterwarnings = ["error"]`, so a leaked numpy RuntimeWarning fails here.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from canonical import EVERY_SERIES, panel
from weightcraft.align import align
from weightcraft.band import no_trade_band
from weightcraft.combine import (
    nanmean_stack,
    nanmedian_stack,
    normalised_shares,
    weighted_nanmean_stack,
    weighted_nanmean_stack_over_time,
)
from weightcraft.costs import apply_costs, book_returns, lagged, turnover
from weightcraft.cross_section import (
    project_out_rows,
    residualize_rows,
    row_counts,
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
    exposure_scale,
    fill_missing,
    gross,
    held,
    net,
    normalised_share,
    quantize,
    rescaled_to_held_count,
    row_sums,
    tilt,
    to_gross,
    weights_from_bins,
)
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
from weightcraft.smoothing import ewm_mean, lag_rows, rolling_mean

if TYPE_CHECKING:
    from collections.abc import Callable

    from weightcraft.arrays import Matrix, Vector

SERIES = pytest.mark.parametrize(
    "series", EVERY_SERIES.values(), ids=list(EVERY_SERIES.keys())
)

# The panels every row-wise function is swept over.
DEGENERATE_PANELS: dict[str, Matrix] = {
    "no_rows": np.zeros((0, 3)),
    "no_columns": np.zeros((4, 0)),
    "nothing_at_all": np.zeros((0, 0)),
    "one_cell": np.asarray([[0.5]]),
    "one_row": np.asarray([[0.2, -0.3, 0.5]]),
    "one_column": np.asarray([[0.2], [-0.3], [0.5]]),
    "all_zero": np.zeros((3, 3)),
    "constant": np.full((3, 3), 0.25),
    "all_missing": np.full((3, 3), np.nan),
    "one_missing_row": np.asarray([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]]),
    "with_infinity": np.asarray([[1.0, np.inf], [2.0, -np.inf]]),
    "tiny": np.full((2, 2), 1e-300),
    "huge": np.full((2, 2), 1e300),
}

PANELS = pytest.mark.parametrize(
    "values", DEGENERATE_PANELS.values(), ids=list(DEGENERATE_PANELS.keys())
)

ROW_WISE: dict[str, Callable[[Matrix], Matrix]] = {
    "row_sums": row_sums,
    "gross": gross,
    "usable_gross": lambda v: to_gross(v, 1.0),
    "center": center,
    "clip": lambda v: clip_allocation(v, 0.1),
    "quantize": lambda v: quantize(v, 0.05),
    "tilt": lambda v: tilt(v, 0.1),
    "weights_from_bins": weights_from_bins,
    "rescaled_to_held_count": rescaled_to_held_count,
    "standardize_rows": standardize_rows,
    "standardize_rows_sample": lambda v: standardize_rows(v, ddof=1),
    "row_rank_pct": row_rank_pct,
    "turnover": turnover,
    "lagged": lambda v: lagged(v, 0),
    "lag_rows": lambda v: lag_rows(v, 1),
    "rolling_mean": lambda v: rolling_mean(v, 2),
    "ewm_mean": lambda v: ewm_mean(v, 2),
    "no_trade_band": lambda v: no_trade_band(v, 0.1),
    "net": net,
    "exposure_scale": lambda v: exposure_scale(v, max_gross=0.5, max_net=0.2),
    "capped": lambda v: capped(v, max_gross=0.5, max_net=0.2),
}


@PANELS
@pytest.mark.parametrize("name", ROW_WISE.keys())
def test_no_row_wise_function_raises_or_warns_on_a_degenerate_panel(
    name: str, values: Matrix
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ROW_WISE[name](values)
    assert result.shape[0] == values.shape[0]


@PANELS
def test_a_frame_survives_every_degenerate_panel(values: Matrix) -> None:
    frame = WeightFrame.from_rows(
        [f"2026-01-{index + 1:02d}" for index in range(values.shape[0])],
        tuple(f"A{index}" for index in range(values.shape[1])),
        values,
    )
    assert frame.shape == values.shape
    assert WeightFrame.from_polars(frame.to_polars()) == frame
    assert align([frame]).values.shape == (1, *values.shape)


@PANELS
def test_a_stack_of_one_degenerate_panel_reduces_without_complaint(
    values: Matrix,
) -> None:
    stack = values[None, :, :]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert nanmean_stack(stack).shape == values.shape
        assert nanmedian_stack(stack).shape == values.shape
        assert weighted_nanmean_stack(stack, np.asarray([1.0])).shape == values.shape
        assert (
            weighted_nanmean_stack_over_time(stack, np.ones((1, values.shape[0]))).shape
            == values.shape
        )


@SERIES
def test_no_metric_raises_on_any_shape_of_series(series: Vector) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(compounded(series), float)
        assert isinstance(sharpe(series), float)
        assert isinstance(max_drawdown(series), float)
        assert to_prices(series).shape == series.shape
        assert drawdown(series).shape == series.shape
        assert isinstance(beta(series, series), float)
        if series.size:
            assert isinstance(cagr(series, float(series.size)), float)


@SERIES
def test_no_rolling_kernel_raises_on_any_shape_of_series(series: Vector) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for period in (2, 5, 1000):
            assert rolling_sums(series, period).shape == series.shape
            assert trailing_std(series, period).shape == series.shape
        assert (
            volatility_target(
                series, VolatilityTargetConfig(period=5, target=0.3)
            ).shape
            == series.shape
        )


@PANELS
def test_no_risk_kernel_raises_on_a_degenerate_panel(values: Matrix) -> None:
    holdings = np.isfinite(values) & (values != 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert downside_deviation(values, 2).shape == values.shape
        assert inverse_volatility_share(values, values).shape == values.shape
        assert normalised_share(values, holdings, power=-1.0).shape == values.shape
        assert (
            equal_risk_weights(values, holdings, EqualRiskConfig(period=2)).shape
            == values.shape
        )
        assert observed_covariance(values).shape[0] == max(values.shape[1], 1)
        assert stack_volatility(values[None, :, :]).size == 1


@pytest.mark.parametrize("size", [1, 2, 3])
def test_equal_risk_handles_a_book_of_any_size(size: int) -> None:
    window = np.random.default_rng(31).normal(size=(50, size)) * 0.02
    weights = equal_risk_row(window, np.ones(size))
    assert weights.size == size
    assert float(weights.sum()) == pytest.approx(1.0)


def test_equal_risk_holds_nothing_when_the_window_is_empty() -> None:
    assert equal_risk_row(np.zeros((0, 2)), np.ones(2)).tolist() == [1.0, 1.0]


def test_the_coverage_penalty_survives_a_single_asset() -> None:
    covariance = np.asarray([[4.0]])
    assert penalised_for_coverage(covariance, np.asarray([0.5])).shape == (1, 1)


@PANELS
def test_a_cost_calculation_survives_a_degenerate_panel(values: Matrix) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert apply_costs(values, values, cost=0.01).shape == values.shape
        assert book_returns(values, values, cost=0.01).shape == (values.shape[0],)


@PANELS
def test_a_cross_sectional_transform_survives_a_degenerate_panel(
    values: Matrix,
) -> None:
    controls = values[None, :, :]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert row_counts(values).shape == (values.shape[0],)
        assert project_out_rows(values, controls).shape == values.shape
        assert residualize_rows(values, controls).shape == values.shape
        if values.shape[1]:
            assert top_n_mask(values, 1).shape == values.shape


def test_a_book_with_no_assets_at_all_still_answers() -> None:
    empty: Matrix = np.zeros((3, 0))
    assert gross(empty).tolist() == [[0.0], [0.0], [0.0]]
    assert to_gross(empty, 1.0).shape == (3, 0)
    assert book_returns(empty, empty, cost=0.1).tolist() == [0.0, 0.0, 0.0]


def test_shares_over_a_single_source_are_all_of_it() -> None:
    assert normalised_shares(np.asarray([0.0])).tolist() == [1.0]
    assert normalised_shares(np.asarray([7.0])).tolist() == [1.0]


def test_a_field_of_all_missing_shares_falls_back_to_equal() -> None:
    assert normalised_shares(np.full(4, np.nan)).tolist() == [0.25] * 4


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: rolling_sums(np.zeros(4), 0), "at least 1"),
        (lambda: trailing_std(np.zeros(4), 1), "at least 2"),
        (lambda: rolling_mean(np.zeros((4, 1)), 0), "at least 1"),
        (lambda: ewm_mean(np.zeros((4, 1)), 0), "at least 1"),
        (lambda: lag_rows(np.zeros((4, 1)), -1), "not be negative"),
        (lambda: lagged(np.zeros((4, 1)), -1), "not be negative"),
        (lambda: apply_costs(np.zeros((4, 1)), np.zeros((4, 1)), cost=-1), "negative"),
        (lambda: apply_costs(np.zeros((4, 1)), np.zeros((3, 1)), cost=0.0), "disagree"),
        (lambda: to_gross(np.zeros((1, 1)), 0.0), "must be positive"),
        (lambda: clip_allocation(np.zeros((1, 1)), 0.0), "must be positive"),
        (lambda: quantize(np.zeros((1, 1)), -1.0), "must be positive"),
        (lambda: no_trade_band(np.zeros((1, 1)), 0.0), "must be positive"),
        (lambda: capped(np.zeros((1, 1)), max_gross=0.0), "must be positive"),
        (lambda: capped(np.zeros((1, 1)), max_gross=-1.0), "must be positive"),
        (lambda: capped(np.zeros((1, 1)), max_net=0.0), "must be positive"),
        (lambda: capped(np.zeros((1, 1)), max_net=-1.0), "must be positive"),
        (lambda: top_n_mask(np.zeros((1, 1)), 0), "at least 1"),
        (lambda: align([]), "at least one frame"),
        (
            lambda: equal_risk_weights(
                np.zeros((2, 1)),
                np.ones((2, 1), dtype=np.bool_),
                EqualRiskConfig(period=2, rebalance=0),
            ),
            "at least 1",
        ),
        (lambda: project_out_rows(np.zeros((1, 2)), np.zeros((1, 2))), "must be"),
        (
            lambda: weighted_nanmean_stack(np.zeros((2, 1, 1)), np.asarray([1.0])),
            "expected shares of shape",
        ),
        (
            lambda: weighted_nanmean_stack(
                np.zeros((2, 1, 1)), np.asarray([-1.0, 1.0])
            ),
            "non-negative",
        ),
        (
            lambda: weighted_nanmean_stack_over_time(
                np.zeros((2, 1, 1)), np.asarray([[np.nan], [1.0]])
            ),
            "non-negative",
        ),
    ],
)
def test_an_impossible_argument_is_refused_by_name(
    call: Callable[[], object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_an_all_missing_row_stays_all_missing_under_a_cap() -> None:
    values: Matrix = np.full((1, 3), np.nan)
    result = capped(values, max_gross=0.5, max_net=0.2)
    assert np.all(np.isnan(result))


def test_a_flat_row_is_left_alone_by_a_cap_rather_than_divided_by_zero() -> None:
    values: Matrix = np.zeros((1, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = capped(values, max_gross=0.5, max_net=0.2)
    assert result.tolist() == [[0.0, 0.0, 0.0]]


def test_one_infinite_cell_does_not_shrink_the_rest_of_its_row_to_nothing() -> None:
    # `usable` blanks the infinity rather than letting it dominate the row's
    # measured gross -- the same guarantee `gross` itself relies on.
    values: Matrix = np.asarray([[0.5, -0.5, np.inf]])
    result = capped(values, max_gross=0.5)
    assert result[0, 0] == pytest.approx(0.25)
    assert result[0, 1] == pytest.approx(-0.25)
    assert np.isinf(result[0, 2])


def test_only_a_finite_non_zero_weight_counts_as_held() -> None:
    # An infinity is missing, not a very large position -- the same rule every
    # reduction in the library follows.
    values: Matrix = np.asarray([[0.0, 0.5, np.nan, np.inf]])
    assert held(values).tolist() == [[False, True, False, False]]


def test_filling_a_gap_prefers_what_is_there() -> None:
    values: Matrix = np.asarray([[0.0, 0.5, np.nan, np.inf]])
    assert fill_missing(values, np.full_like(values, 9.0)).tolist() == [
        [0.0, 0.5, 9.0, np.inf]
    ]


def test_a_stack_of_one_frame_is_that_frame() -> None:
    values = panel(EVERY_SERIES["mixed"])
    assert np.array_equal(nanmean_stack(values[None, :, :]), values, equal_nan=True)


def test_a_gap_long_enough_to_underflow_does_not_blank_the_column() -> None:
    # The exponentially decaying weights reach the denormals after a few
    # hundred missing rows, where numerator and denominator lose precision at
    # different rates and the average drifts. Rescaling holds it exactly.
    column: Matrix = np.full((3000, 1), np.nan)
    column[0] = 1.0
    column[1] = 2.0
    smoothed = ewm_mean(column, 2)[:, 0]
    assert np.allclose(smoothed[1:], 1.75)


def test_a_covariance_over_a_window_too_short_to_have_one_is_zero() -> None:
    assert observed_covariance(np.zeros((1, 3))).tolist() == [[0.0] * 3] * 3
    assert observed_covariance(np.zeros((0, 2))).tolist() == [[0.0, 0.0]] * 2
