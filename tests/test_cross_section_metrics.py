from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from weightcraft.cross_section import (
    project_out_rows,
    residualize_rows,
    row_counts,
    row_rank_pct,
    standardize_rows,
    top_n_mask,
)
from weightcraft.metrics import (
    DEGENERATE_SHARPE,
    beta,
    cagr,
    compounded,
    drawdown,
    max_drawdown,
    sharpe,
    to_prices,
)

if TYPE_CHECKING:
    from weightcraft.arrays import Cube, Matrix


def test_row_counts_report_what_each_row_observed() -> None:
    assert row_counts(np.asarray([[1.0, np.nan], [1.0, 2.0]])).tolist() == [1.0, 2.0]


def test_standardising_leaves_a_row_with_mean_zero_and_unit_dispersion() -> None:
    scaled = standardize_rows(np.asarray([[1.0, 2.0, 3.0]]))
    assert scaled.sum() == pytest.approx(0.0)
    assert float(np.std(scaled[0])) == pytest.approx(1.0)


def test_a_constant_row_scores_zero_rather_than_missing() -> None:
    assert standardize_rows(np.asarray([[2.0, 2.0]])).tolist() == [[0.0, 0.0]]


def test_standardising_preserves_where_the_gaps_were() -> None:
    scaled = standardize_rows(np.asarray([[1.0, np.nan, 3.0]]))
    assert np.isnan(scaled[0, 1])
    assert np.isfinite(scaled[0, [0, 2]]).all()


def test_an_empty_row_stays_empty() -> None:
    assert np.isnan(standardize_rows(np.asarray([[np.nan, np.nan]]))).all()


def test_a_percentile_rank_puts_the_largest_at_one() -> None:
    ranked = row_rank_pct(np.asarray([[3.0, 1.0, 2.0]]))
    assert ranked[0, 0] == pytest.approx(1.0)
    assert ranked[0, 1] == pytest.approx(1 / 3)


def test_a_percentile_rank_skips_a_gap() -> None:
    ranked = row_rank_pct(np.asarray([[3.0, np.nan]]))
    assert ranked[0, 0] == pytest.approx(1.0)
    assert np.isnan(ranked[0, 1])
    assert np.isnan(row_rank_pct(np.asarray([[np.nan]]))).all()


def test_the_top_n_mask_selects_the_largest() -> None:
    assert top_n_mask(np.asarray([[3.0, 1.0, 2.0]]), 2).tolist() == [
        [True, False, True]
    ]


def test_the_top_n_mask_breaks_a_tie_toward_the_earlier_column() -> None:
    assert top_n_mask(np.asarray([[1.0, 1.0]]), 1).tolist() == [[True, False]]


def test_the_top_n_mask_ignores_gaps_and_empty_rows() -> None:
    assert top_n_mask(np.asarray([[np.nan, 1.0]]), 1).tolist() == [[False, True]]
    assert top_n_mask(np.asarray([[np.nan, np.nan]]), 1).tolist() == [[False, False]]


def test_asking_for_no_names_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        top_n_mask(np.asarray([[1.0]]), 0)


def test_a_residual_is_orthogonal_to_the_control_it_was_projected_off() -> None:
    generator = np.random.default_rng(21)
    control: Matrix = generator.normal(size=(1, 40))
    signal: Matrix = 3.0 * control + generator.normal(size=(1, 40)) * 0.1
    residual = project_out_rows(signal, control[None, :, :])
    assert float(residual[0] @ control[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(residual[0].sum()) == pytest.approx(0.0, abs=1e-9)


def test_a_perfectly_explained_signal_leaves_no_residual() -> None:
    control: Matrix = np.asarray([[1.0, 2.0, 3.0, 4.0]])
    residual = project_out_rows(2.0 * control, control[None, :, :])
    assert np.allclose(residual, 0.0, atol=1e-9)


def test_a_collinear_control_is_dropped_rather_than_inverted() -> None:
    control: Matrix = np.asarray([[1.0, 2.0, 3.0]])
    controls = np.stack([control, control])
    assert np.isfinite(project_out_rows(np.asarray([[1.0, 5.0, 2.0]]), controls)).all()


def test_a_row_the_control_does_not_cover_is_left_missing() -> None:
    residual = project_out_rows(
        np.asarray([[1.0, 2.0]]), np.asarray([[[np.nan, np.nan]]])
    )
    assert np.isnan(residual).all()


def test_an_ineligible_asset_is_excluded_from_the_regression() -> None:
    residual = project_out_rows(
        np.asarray([[1.0, 2.0, 3.0]]),
        np.asarray([[[1.0, 2.0, 9.0]]]),
        np.asarray([[True, True, False]], dtype=np.bool_),
    )
    assert np.isnan(residual[0, 2])


def test_a_mis_shaped_control_block_is_refused() -> None:
    with pytest.raises(ValueError, match="must be"):
        project_out_rows(np.zeros((1, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="do not match"):
        project_out_rows(np.zeros((1, 2)), np.zeros((1, 1, 3)))


def test_residualizing_blanks_a_row_too_thin_to_regress() -> None:
    signal = np.asarray([[1.0, np.nan, np.nan]])
    residual = residualize_rows(signal, np.asarray([[[1.0, 2.0, 3.0]]]))
    assert np.isnan(residual).all()


def test_residualizing_can_skip_the_rescale() -> None:
    generator = np.random.default_rng(5)
    signal: Matrix = generator.normal(size=(1, 20))
    controls: Cube = generator.normal(size=(1, 1, 20))
    raw = residualize_rows(signal, controls, standardize=False)
    scaled = residualize_rows(signal, controls, standardize=True)
    assert float(np.std(scaled[0])) == pytest.approx(1.0)
    assert float(np.std(raw[0])) != pytest.approx(1.0)


def test_compounding_multiplies_the_returns_together() -> None:
    assert compounded(np.asarray([0.1, 0.1])) == pytest.approx(0.21)


def test_a_missing_return_leaves_the_price_flat_and_an_infinite_one_a_hole() -> None:
    prices = to_prices(np.asarray([0.1, np.nan, 0.1, np.inf, 0.1]))
    assert prices[0] == pytest.approx(1.1)
    assert np.isnan(prices[1])
    assert prices[2] == pytest.approx(1.21)
    assert np.isnan(prices[3])
    assert prices[4] == pytest.approx(1.331)


def test_cagr_annualises_the_compounded_return() -> None:
    assert cagr(np.asarray([0.0] * 365), 365.0) == pytest.approx(0.0)
    assert cagr(np.asarray([1.0]), 365.0, periods=365) == pytest.approx(1.0)


def test_cagr_refuses_a_zero_length_history() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        cagr(np.asarray([0.1]), 0.0)


def test_a_series_that_never_varies_scores_the_sentinel() -> None:
    assert sharpe(np.zeros(10)) == DEGENERATE_SHARPE


def test_a_constant_non_zero_series_is_not_the_sentinel() -> None:
    assert sharpe(np.full(10, 0.01)) > 1e10


def test_sharpe_annualises_the_ratio() -> None:
    returns = np.asarray([0.01, -0.01, 0.02, 0.0, 0.01])
    expected = float(np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(365)
    assert sharpe(returns) == pytest.approx(expected)


def test_beta_recovers_a_known_slope() -> None:
    generator = np.random.default_rng(13)
    underlying = generator.normal(size=200)
    assert beta(2.0 * underlying, underlying) == pytest.approx(2.0)


def test_beta_needs_two_shared_observations() -> None:
    assert np.isnan(beta(np.asarray([0.1, np.nan]), np.asarray([np.nan, 0.1])))


def test_beta_against_a_flat_benchmark_is_undefined() -> None:
    assert np.isnan(beta(np.asarray([0.1, 0.2, 0.3]), np.zeros(3)))


def test_a_drawdown_is_zero_at_a_new_peak_and_negative_below_it() -> None:
    series = drawdown(np.asarray([0.5, -0.5, 0.0]))
    assert series[0] == pytest.approx(0.0)
    assert series[1] == pytest.approx(-0.5)
    assert series[2] == pytest.approx(-0.5)


def test_the_worst_drawdown_is_the_lowest_point() -> None:
    assert max_drawdown(np.asarray([0.5, -0.5, 0.0])) == pytest.approx(-0.5)


def test_a_book_that_never_fell_has_no_drawdown() -> None:
    assert max_drawdown(np.asarray([0.1, 0.1])) == pytest.approx(0.0)
    assert max_drawdown(np.asarray([np.nan, np.nan])) == 0.0
