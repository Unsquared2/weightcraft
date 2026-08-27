"""Structural facts: alignment, covariance shape, and the shrinkage endpoints.

The covariance tests follow PyPortfolioOpt's pattern -- assert positive
semi-definiteness, check the correlation round trip, and pin the endpoints of
the shrinkage parameter, where the answer is known in closed form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from canonical import correlated_panel, returns_panel
from conftest import dates, frame
from weightcraft.align import align
from weightcraft.combine import nanmean_stack
from weightcraft.frame import WeightFrame
from weightcraft.risk import (
    equal_risk_row,
    observed_covariance,
    penalised_for_coverage,
)

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix

PSD_TOLERANCE = -1e-10


def is_positive_semidefinite(matrix: Matrix) -> bool:
    """Every eigenvalue non-negative, to within rounding."""
    return bool(np.min(np.linalg.eigvalsh(matrix)) >= PSD_TOLERANCE)


def correlation_of(covariance: Matrix) -> Matrix:
    deviation = np.sqrt(np.diag(covariance))
    out: Matrix = covariance / np.outer(deviation, deviation)
    return out


# --------------------------------------------------------------------------
# Covariance structure.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.95, -0.5])
def test_a_complete_covariance_matrix_is_positive_semidefinite(rho: float) -> None:
    assert is_positive_semidefinite(observed_covariance(correlated_panel(41, 400, rho)))


def test_a_correlation_matrix_has_a_unit_diagonal_and_lives_in_minus_one_to_one() -> (
    None
):
    correlation = correlation_of(observed_covariance(correlated_panel(42, 500, 0.7)))
    assert np.allclose(np.diag(correlation), 1.0)
    assert np.all(np.abs(correlation) <= 1.0 + 1e-12)


def test_a_covariance_round_trips_through_its_correlation() -> None:
    covariance = observed_covariance(returns_panel(43, 300, [0.01, 0.03, 0.05]))
    deviation = np.sqrt(np.diag(covariance))
    rebuilt = correlation_of(covariance) * np.outer(deviation, deviation)
    assert np.allclose(rebuilt, covariance)


def test_a_perfectly_correlated_pair_has_a_correlation_of_one() -> None:
    column = returns_panel(44, 200, [0.02])
    covariance = observed_covariance(np.column_stack([column, column]))
    assert correlation_of(covariance)[0, 1] == pytest.approx(1.0)


def test_a_recovered_correlation_is_the_one_that_was_asked_for() -> None:
    for rho in (0.0, 0.4, 0.85):
        window = correlated_panel(45, 4000, rho)
        assert correlation_of(observed_covariance(window))[0, 1] == pytest.approx(
            rho, abs=0.05
        )


def test_full_coverage_leaves_the_matrix_exactly_as_it_was() -> None:
    covariance = observed_covariance(correlated_panel(46, 300, 0.5))
    assert np.array_equal(penalised_for_coverage(covariance, np.ones(3)), covariance)


def test_the_coverage_penalty_keeps_the_matrix_usable() -> None:
    covariance = observed_covariance(correlated_panel(47, 400, 0.6))
    penalised = penalised_for_coverage(covariance, np.asarray([0.2, 0.6, 1.0]))
    assert is_positive_semidefinite(penalised)
    assert np.allclose(penalised, penalised.T)


# --------------------------------------------------------------------------
# Shrinkage endpoints, where the answer is known in closed form.
# --------------------------------------------------------------------------


def test_no_shrinkage_leaves_the_correlations_in_play() -> None:
    # With a strongly correlated pair, the correlated names must be sized down
    # relative to what inverse volatility alone would give them.
    window = correlated_panel(48, 2000, rho=0.95)
    unshrunk = equal_risk_row(window, np.ones(3), shrinkage=0.0)
    shrunk = equal_risk_row(window, np.ones(3), shrinkage=1.0)
    assert unshrunk[0] < shrunk[0]
    assert unshrunk[2] > shrunk[2]


def test_full_shrinkage_ignores_the_correlations_entirely() -> None:
    window = correlated_panel(49, 2000, rho=0.95)
    deviations = np.std(window, axis=0, ddof=1)
    inverse = (1.0 / deviations) / (1.0 / deviations).sum()
    assert np.allclose(
        equal_risk_row(window, np.ones(3), shrinkage=1.0), inverse, atol=0.01
    )


@pytest.mark.parametrize("shrinkage", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_every_shrinkage_setting_produces_a_usable_book(shrinkage: float) -> None:
    weights = equal_risk_row(
        correlated_panel(50, 800, 0.7), np.ones(3), shrinkage=shrinkage
    )
    assert np.all(weights > 0.0)
    assert float(weights.sum()) == pytest.approx(1.0)


@pytest.mark.parametrize("iterations", [1, 5, 50, 500])
def test_the_fixed_point_stays_sane_however_long_it_runs(iterations: int) -> None:
    weights = equal_risk_row(
        correlated_panel(51, 800, 0.6), np.ones(3), iterations=iterations
    )
    assert np.all(np.isfinite(weights))
    assert float(weights.sum()) == pytest.approx(1.0)


def test_the_fixed_point_converges_rather_than_oscillating() -> None:
    # The undamped update oscillates forever; the geometric-mean damping is
    # what makes these two agree.
    window = correlated_panel(52, 1000, 0.7)
    many = equal_risk_row(window, np.ones(3), iterations=500)
    more = equal_risk_row(window, np.ones(3), iterations=2000)
    assert np.allclose(many, more, atol=1e-9)


# --------------------------------------------------------------------------
# Alignment.
# --------------------------------------------------------------------------


def test_each_frame_keeps_its_own_values_at_its_own_labels() -> None:
    """The one invariant alignment must never break."""
    first = frame(("BTC", "ETH"), [[0.1, 0.2], [0.3, 0.4]])
    second = WeightFrame(
        dates=dates(2, "2026-01-02"),
        assets=("ETH", "SOL"),
        values=np.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64),
    )
    stack = align([first, second])
    for depth, source in enumerate((first, second)):
        for row, date in enumerate(source.dates):
            for column, asset in enumerate(source.assets):
                placed = stack.values[
                    depth,
                    int(np.flatnonzero(stack.dates == date)[0]),
                    stack.assets.index(asset),
                ]
                assert placed == source.values[row, column]


def test_aligning_an_already_aligned_set_changes_nothing() -> None:
    first = frame(("BTC", "ETH"), [[0.1, 0.2]])
    second = frame(("BTC", "ETH"), [[0.3, 0.4]])
    once = align([first, second])
    again = align([once.with_values(once.values[0]), once.with_values(once.values[1])])
    assert np.array_equal(once.values, again.values, equal_nan=True)
    assert once.assets == again.assets


def test_aligning_one_frame_leaves_it_alone() -> None:
    only = frame(("ETH", "BTC"), [[0.2, 0.8]])
    stack = align([only])
    # Sorted, so the labels may be reordered -- but every cell must follow.
    for column, asset in enumerate(only.assets):
        assert stack.values[0, 0, stack.assets.index(asset)] == only.values[0, column]


def test_a_frame_that_covers_nothing_contributes_nothing() -> None:
    real = frame(("BTC",), [[0.5]])
    nothing = WeightFrame(
        dates=dates(0), assets=(), values=np.zeros((0, 0), dtype=np.float64)
    )
    stack = align([real, nothing])
    assert np.array_equal(nanmean_stack(stack.values), real.values, equal_nan=True)


def test_the_union_grows_with_every_frame_added() -> None:
    frames = [
        frame(("BTC",), [[0.1]]),
        frame(("ETH",), [[0.2]]),
        frame(("SOL",), [[0.3]]),
    ]
    sizes = [len(align(frames[: count + 1]).assets) for count in range(3)]
    assert sizes == [1, 2, 3]


def test_a_stack_reports_how_many_frames_it_holds() -> None:
    stack = align([frame(("BTC",), [[0.1]]), frame(("BTC",), [[0.2]])])
    assert stack.count == 2
    assert stack.values.shape[0] == 2
