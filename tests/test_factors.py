"""The canonical factor model, and the cap that keeps a book inside its limits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from weightcraft import (
    CANONICAL_FACTORS,
    CanonicalFactorConfig,
    DrawdownScaleConfig,
    canonical_factor_returns,
    drawdown_scale,
    factor_cap_blend,
    factor_capped,
    factor_exposure,
    gross,
    net,
    rolling_factor_betas,
)
from weightcraft.cross_section import project_out_rows
from weightcraft.factors import _momentum_characteristic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from weightcraft.arrays import Cube, Matrix, Vector

LIMITS: Vector = np.array([0.15, 0.15, 0.15])


def book(row: Sequence[float]) -> Matrix:
    return np.asarray(row, dtype=np.float64).reshape(1, -1)


def cube(loadings: Sequence[Sequence[float]] | Matrix) -> Cube:
    """`(factors, assets)` laid out as the one-date cube the cap expects."""
    return np.asarray(loadings, dtype=np.float64)[:, None, :]


def a_row(seed: int, assets: int = 20) -> tuple[Matrix, Cube]:
    """A cash-neutral book: market loadings near one, the other two near zero."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, assets)
    raw -= raw.mean()
    weights = book(raw / np.abs(raw).sum())
    loadings = cube(
        np.vstack(
            [
                rng.normal(1.0, 0.35, assets),
                rng.normal(0.0, 0.6, assets),
                rng.normal(0.0, 0.6, assets),
            ]
        )
    )
    return weights, loadings


def a_breaching_row(seed: int, assets: int = 20) -> tuple[Matrix, Cube]:
    """`a_row`, redrawn until it actually breaches a limit.

    Deterministic per seed. A test asserting what capping does to a breaching
    book is worthless if handed a compliant one.
    """
    for offset in range(200):
        weights, loadings = a_row(seed * 1000 + offset, assets)
        if worst(weights, loadings) > 1.0:
            return weights, loadings
    msg = f"no breaching row found for seed {seed}"
    raise AssertionError(msg)


def worst(values: Matrix, betas: Cube, limits: Vector = LIMITS) -> float:
    return float(np.max(np.abs(factor_exposure(values, betas)) / limits))


# --- exposure ------------------------------------------------------------


def test_exposure_is_the_weighted_sum_of_loadings() -> None:
    weights = book([0.5, -0.5])
    betas = cube([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_allclose(factor_exposure(weights, betas), [[-1.0, 0.0, 0.0]])


def test_a_cell_missing_a_loading_contributes_nothing() -> None:
    betas = cube([[1.0, np.nan], [1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(factor_exposure(book([0.5, 99.0]), betas), [[0.5] * 3])


def test_a_missing_weight_contributes_nothing() -> None:
    betas = cube([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(factor_exposure(book([0.5, np.nan]), betas), [[0.5] * 3])


def test_exposure_refuses_betas_that_do_not_match_the_book() -> None:
    with pytest.raises(ValueError, match="do not match"):
        factor_exposure(book([1.0, 2.0]), cube([[1.0], [1.0], [1.0]]))


def test_exposure_refuses_a_cube_of_the_wrong_rank() -> None:
    flat: Cube = np.ones((3, 2))
    with pytest.raises(ValueError, match="factors, dates, assets"):
        factor_exposure(book([1.0, 2.0]), flat)


# --- the cap: the properties that matter ---------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_a_capped_book_is_inside_every_limit(seed: int) -> None:
    weights, betas = a_breaching_row(seed)
    assert worst(factor_capped(weights, betas, LIMITS), betas) <= 1.0 + 1e-9


@pytest.mark.parametrize("seed", range(40))
def test_capping_preserves_gross(seed: int) -> None:
    weights, betas = a_breaching_row(seed)
    capped = factor_capped(weights, betas, LIMITS)
    assert float(gross(capped)[0, 0]) == pytest.approx(float(gross(weights)[0, 0]))


def test_a_compliant_book_is_returned_unchanged() -> None:
    weights = book([0.5, -0.5])
    betas = cube([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(factor_cap_blend(weights, betas, LIMITS), [[0.0]])
    np.testing.assert_allclose(factor_capped(weights, betas, LIMITS), weights)


@pytest.mark.parametrize("seed", range(20))
def test_the_cap_binds_exactly_rather_than_overshooting(seed: int) -> None:
    """A cap that hedged further than asked would throw away deliberate tilt."""
    weights, betas = a_breaching_row(seed)
    assert worst(factor_capped(weights, betas, LIMITS), betas) == pytest.approx(
        1.0, abs=1e-6
    )


@pytest.mark.parametrize("seed", range(20))
def test_net_scales_by_the_same_ratio_the_exposure_does(seed: int) -> None:
    """The intercept makes net one of the functionals the blend annihilates."""
    weights, betas = a_breaching_row(seed)
    capped = factor_capped(weights, betas, LIMITS)
    before = factor_exposure(weights, betas)
    after = factor_exposure(capped, betas)
    ratio = float(np.max(np.abs(after)) / np.max(np.abs(before)))
    assert float(net(capped)[0, 0]) == pytest.approx(
        ratio * float(net(weights)[0, 0]), abs=1e-9
    )


@pytest.mark.parametrize("seed", range(20))
def test_capping_never_increases_the_absolute_net(seed: int) -> None:
    weights, betas = a_breaching_row(seed)
    capped = factor_capped(weights, betas, LIMITS)
    assert abs(float(net(capped)[0, 0])) <= abs(float(net(weights)[0, 0])) + 1e-12


def test_a_name_without_loadings_is_still_held_at_its_own_weight() -> None:
    """Upstream drops such a name; here that would be an instruction to liquidate."""
    weights, betas = a_breaching_row(3, assets=20)
    betas[:, 0, 7] = np.nan
    capped = factor_capped(weights, betas, LIMITS)
    assert capped[0, 7] == pytest.approx(weights[0, 7])
    assert np.isfinite(capped).all()


def test_the_cap_refuses_limits_that_are_not_positive() -> None:
    weights, betas = a_breaching_row(1)
    with pytest.raises(ValueError, match="must be positive"):
        factor_capped(weights, betas, np.array([0.15, 0.0, 0.15]))


def test_the_cap_refuses_one_limit_per_factor() -> None:
    weights, betas = a_breaching_row(1)
    with pytest.raises(ValueError, match="one entry per factor"):
        factor_capped(weights, betas, np.array([0.15, 0.15]))


# --- the regressions this design was built around -------------------------


def test_solving_the_blend_before_the_rescale_would_breach() -> None:
    """The trap: the projection shrinks gross, and restoring it re-inflates exposure.

    Guards the shipped solve against the naive one, which breached on the great
    majority of rows and by up to 3.7x the limit.
    """
    breached = 0
    for seed in range(200):
        weights, betas = a_breaching_row(seed)
        exposures = factor_exposure(weights, betas)[0]
        naive = 1.0 - float(np.min(LIMITS / np.abs(exposures)))
        neutral = project_out_rows(weights, betas)
        blended = (1.0 - naive) * weights + naive * neutral
        rescaled = blended * (gross(weights) / gross(blended))
        if worst(rescaled, betas) > 1.0 + 1e-9:
            breached += 1
        assert worst(factor_capped(weights, betas, LIMITS), betas) <= 1.0 + 1e-9
    assert breached > 100, "the naive solve must actually be shown to breach"


def test_the_non_monotone_row_is_still_capped_exactly() -> None:
    """A row where the rescaled exposure rises before it falls.

    Bisection is justified by concavity, not by monotonicity -- this row is the
    counterexample to the monotonicity argument, and must still land on the limit.
    """
    weights, betas = a_row(304, assets=8)
    held = float(gross(weights)[0, 0])
    neutral = project_out_rows(weights, betas)
    phi = []
    for step in (0.0, 0.2, 0.415, 0.5):
        blended = (1.0 - step) * weights + step * neutral
        phi.append((1.0 - step) * held / float(gross(blended)[0, 0]))
    assert phi[2] > phi[0], "this row is meant to be the non-monotone one"
    assert worst(factor_capped(weights, betas, LIMITS), betas) == pytest.approx(
        1.0, abs=1e-6
    )


def test_a_row_too_thin_to_hedge_is_shrunk_instead() -> None:
    """Four names fit four regressors exactly, leaving a residual of pure noise."""
    weights = book([0.4, -0.3, 0.2, -0.1])
    rng = np.random.default_rng(2)
    betas = cube(
        np.vstack(
            [rng.normal(1.0, 0.35, 4), rng.normal(0.0, 0.6, 4), rng.normal(0.0, 0.6, 4)]
        )
    )
    assert worst(weights, betas) > 1.0, "the fixture must actually breach"
    np.testing.assert_allclose(factor_cap_blend(weights, betas, LIMITS), [[0.0]])
    capped = factor_capped(weights, betas, LIMITS)
    assert worst(capped, betas) <= 1.0 + 1e-9
    # A uniform shrink keeps every ratio between names.
    ratio = capped[0] / weights[0]
    np.testing.assert_allclose(ratio, ratio[0])


def test_a_near_degenerate_row_is_not_blown_up_by_restoring_gross() -> None:
    """Five names pass a count guard, but the residual is a sliver of the book."""
    weights, betas = a_breaching_row(5, assets=5)
    capped = factor_capped(weights, betas, LIMITS)
    assert worst(capped, betas) <= 1.0 + 1e-9
    assert float(gross(capped)[0, 0]) <= float(gross(weights)[0, 0]) + 1e-12


# --- point in time --------------------------------------------------------


def test_betas_do_not_read_their_own_date() -> None:
    """Perturbing the last row must leave every earlier beta bit-identical."""
    rng = np.random.default_rng(0)
    assets = rng.normal(0.0, 0.02, (240, 12))
    factors = rng.normal(0.0, 0.01, (240, 3))
    config = CanonicalFactorConfig(beta_window=60, beta_min_periods=30)
    before = rolling_factor_betas(assets, factors, config=config)

    assets[-1] *= 100.0
    factors[-1] *= 100.0
    after = rolling_factor_betas(assets, factors, config=config)
    np.testing.assert_array_equal(before, after)


def test_a_beta_recovers_a_known_loading() -> None:
    rng = np.random.default_rng(1)
    factors = rng.normal(0.0, 0.01, (400, 3))
    loadings = np.array([1.5, -0.4, 0.8])
    assets = (factors @ loadings).reshape(-1, 1)
    betas = rolling_factor_betas(
        assets, factors, config=CanonicalFactorConfig(beta_window=120)
    )
    np.testing.assert_allclose(betas[:, -1, 0], loadings, atol=1e-8)


def test_betas_are_laid_out_as_project_out_rows_wants_its_controls() -> None:
    rng = np.random.default_rng(4)
    betas = rolling_factor_betas(
        rng.normal(0.0, 0.02, (200, 7)),
        rng.normal(0.0, 0.01, (200, 3)),
        config=CanonicalFactorConfig(beta_window=60, beta_min_periods=30),
    )
    assert betas.shape == (3, 200, 7)


def test_betas_are_missing_until_the_window_fills() -> None:
    rng = np.random.default_rng(6)
    betas = rolling_factor_betas(
        rng.normal(0.0, 0.02, (200, 4)),
        rng.normal(0.0, 0.01, (200, 3)),
        config=CanonicalFactorConfig(beta_window=60, beta_min_periods=30),
    )
    assert np.isnan(betas[:, :30, :]).all()
    assert np.isfinite(betas[:, -1, :]).all()


def test_betas_refuse_panels_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="dates"):
        rolling_factor_betas(np.zeros((10, 2)), np.zeros((9, 3)))


# --- the factor returns ---------------------------------------------------


def a_panel(dates: int = 300, assets: int = 30) -> tuple[Matrix, Matrix]:
    rng = np.random.default_rng(11)
    steps = rng.normal(0.001, 0.03, (dates, assets))
    prices = 100.0 * np.cumprod(1.0 + steps, axis=0)
    caps = prices * np.linspace(1e6, 1e9, assets)
    return prices, caps


def test_factor_returns_have_one_column_per_canonical_factor() -> None:
    prices, caps = a_panel()
    factors = canonical_factor_returns(prices, caps)
    assert factors.shape == (prices.shape[0], len(CANONICAL_FACTORS))


def test_the_market_factor_tracks_a_cap_weighted_book() -> None:
    prices, caps = a_panel()
    factors = canonical_factor_returns(prices, caps)
    returns = prices[1:] / prices[:-1] - 1.0
    weights = caps[:-1] / caps[:-1].sum(axis=1, keepdims=True)
    expected = (returns * weights).sum(axis=1)
    np.testing.assert_allclose(factors[1:, 0], expected, atol=1e-12)


def test_the_first_date_has_no_factor_return() -> None:
    prices, caps = a_panel()
    assert np.isnan(canonical_factor_returns(prices, caps)[0]).all()


def test_a_universe_mask_excludes_a_name_from_the_cross_section() -> None:
    prices, caps = a_panel()
    members = np.ones(prices.shape, dtype=np.bool_)
    members[:, 0] = False
    everything = canonical_factor_returns(prices, caps)
    without = canonical_factor_returns(prices, caps, members=members)
    assert not np.allclose(everything[1:, 0], without[1:, 0], equal_nan=True), (
        "masking the largest name must move the market return"
    )


def test_a_name_the_mask_never_selects_does_not_change_the_other_dates() -> None:
    """Survivorship: a whole-history decision must not leak into every date."""
    prices, caps = a_panel()
    members = np.ones(prices.shape, dtype=np.bool_)
    kept = canonical_factor_returns(prices, caps, members=members)

    members[:, -1] = False
    dropped_always = canonical_factor_returns(prices, caps, members=members)
    trimmed = canonical_factor_returns(
        prices[:, :-1], caps[:, :-1], members=members[:, :-1]
    )
    np.testing.assert_allclose(dropped_always, trimmed, equal_nan=True)
    assert not np.allclose(kept, dropped_always, equal_nan=True)


def test_factor_returns_refuse_panels_of_different_shapes() -> None:
    with pytest.raises(ValueError, match="do not match"):
        canonical_factor_returns(np.ones((5, 3)), np.ones((5, 4)))


def test_factor_returns_refuse_a_mask_of_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="do not match"):
        canonical_factor_returns(
            np.ones((5, 3)), np.ones((5, 3)), members=np.ones((5, 2), dtype=np.bool_)
        )


def test_a_minimum_asset_count_blanks_a_thin_cross_section() -> None:
    prices, caps = a_panel(dates=40, assets=4)
    factors = canonical_factor_returns(
        prices, caps, config=CanonicalFactorConfig(minimum_assets=6)
    )
    assert np.isnan(factors[:, 0]).all()


# --- configuration --------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"beta_window": 2}, "beta_window"),
        ({"beta_min_periods": 2}, "beta_min_periods"),
        ({"beta_min_periods": 500}, "beta_min_periods"),
        ({"momentum_window": 0}, "momentum_window"),
        ({"minimum_assets": 0}, "minimum_assets"),
    ],
)
def test_the_config_refuses_settings_it_cannot_honour(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CanonicalFactorConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window": 0, "drawdown_limit": 0.2}, "window"),
        ({"window": 10, "drawdown_limit": 0.0}, "drawdown_limit"),
        ({"window": 10, "drawdown_limit": 0.2, "max_leverage": 0.0}, "max_leverage"),
    ],
)
def test_the_drawdown_config_refuses_settings_it_cannot_honour(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DrawdownScaleConfig(**kwargs)  # type: ignore[arg-type]


# --- drawdown sizing ------------------------------------------------------


def test_a_book_at_its_high_water_mark_is_not_cut() -> None:
    rising: Vector = np.full(50, 0.01)
    scale = drawdown_scale(rising, DrawdownScaleConfig(window=20, drawdown_limit=0.2))
    np.testing.assert_allclose(scale, 1.0)


def test_a_book_at_its_limit_is_cut_to_nothing() -> None:
    returns: Vector = np.concatenate([np.zeros(10), np.array([-0.2]), np.zeros(5)])
    scale = drawdown_scale(returns, DrawdownScaleConfig(window=20, drawdown_limit=0.2))
    assert scale[10] == pytest.approx(0.0, abs=1e-9)


def test_the_cut_is_proportional_to_the_drawdown() -> None:
    returns: Vector = np.concatenate([np.zeros(10), np.array([-0.1]), np.zeros(5)])
    scale = drawdown_scale(returns, DrawdownScaleConfig(window=20, drawdown_limit=0.2))
    assert scale[10] == pytest.approx(0.5, abs=1e-9)


def test_the_scale_never_exceeds_the_leverage_ceiling() -> None:
    rng = np.random.default_rng(9)
    scale = drawdown_scale(
        rng.normal(0.0, 0.05, 500),
        DrawdownScaleConfig(window=30, drawdown_limit=0.2, max_leverage=1.5),
    )
    assert scale.max() <= 1.5 + 1e-12
    assert scale.min() >= 0.0


def test_an_empty_series_has_no_scale() -> None:
    assert (
        drawdown_scale(
            np.zeros(0), DrawdownScaleConfig(window=10, drawdown_limit=0.2)
        ).size
        == 0
    )


def test_the_window_bounds_how_far_back_the_peak_is_remembered() -> None:
    """A peak older than the window stops cutting the book."""
    returns: Vector = np.concatenate([np.array([0.5]), np.array([-0.2]), np.zeros(30)])
    short = drawdown_scale(returns, DrawdownScaleConfig(window=5, drawdown_limit=0.2))
    long = drawdown_scale(returns, DrawdownScaleConfig(window=40, drawdown_limit=0.2))
    assert short[-1] > long[-1]


# --- degenerate inputs ----------------------------------------------------


def test_an_empty_panel_has_no_betas() -> None:
    betas = rolling_factor_betas(np.zeros((0, 0)), np.zeros((0, 3)))
    assert betas.shape == (3, 0, 0)


def test_a_panel_too_short_to_reach_the_window_has_no_betas() -> None:
    rng = np.random.default_rng(12)
    betas = rolling_factor_betas(
        rng.normal(0.0, 0.02, (10, 4)),
        rng.normal(0.0, 0.01, (10, 3)),
        config=CanonicalFactorConfig(beta_window=60, beta_min_periods=30),
    )
    assert np.isnan(betas).all()


def test_a_collinear_window_falls_back_to_the_minimum_norm_answer() -> None:
    """A rank-deficient design must answer, not raise -- the pinv branch."""
    rng = np.random.default_rng(13)
    factors = rng.normal(0.0, 0.01, (200, 3))
    factors[:, 2] = factors[:, 1]
    assets = (factors @ np.array([1.0, 0.5, 0.5])).reshape(-1, 1)
    betas = rolling_factor_betas(
        assets,
        factors,
        config=CanonicalFactorConfig(beta_window=60, beta_min_periods=30),
    )
    assert np.isfinite(betas[:, -1, 0]).all()


def test_an_all_missing_book_is_returned_untouched() -> None:
    weights: Matrix = np.full((1, 4), np.nan)
    betas: Cube = np.ones((3, 1, 4))
    np.testing.assert_array_equal(
        np.isnan(factor_capped(weights, betas, LIMITS)), np.isnan(weights)
    )


def test_a_book_with_no_loadings_at_all_is_returned_untouched() -> None:
    weights = book([0.5, -0.5, 0.25, -0.25])
    betas: Cube = np.full((3, 1, 4), np.nan)
    np.testing.assert_allclose(factor_capped(weights, betas, LIMITS), weights)
    np.testing.assert_allclose(factor_cap_blend(weights, betas, LIMITS), [[0.0]])


def test_a_flat_book_is_left_alone() -> None:
    weights: Matrix = np.zeros((1, 6))
    betas: Cube = np.ones((3, 1, 6))
    np.testing.assert_allclose(factor_capped(weights, betas, LIMITS), weights)


def test_an_empty_book_has_no_rows_to_cap() -> None:
    weights: Matrix = np.zeros((0, 3))
    betas: Cube = np.zeros((3, 0, 3))
    assert factor_capped(weights, betas, LIMITS).shape == (0, 3)
    assert factor_exposure(weights, betas).shape == (0, 3)


def test_a_book_whose_projection_vanishes_is_shrunk_not_divided() -> None:
    """Two names against three factors: the residual is exactly zero."""
    weights = book([0.6, -0.4])
    betas = cube([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
    assert worst(weights, betas) > 1.0
    np.testing.assert_allclose(factor_cap_blend(weights, betas, LIMITS), [[0.0]])
    assert worst(factor_capped(weights, betas, LIMITS), betas) <= 1.0 + 1e-9


def test_an_all_missing_return_series_still_yields_a_scale() -> None:
    scale = drawdown_scale(
        np.full(20, np.nan), DrawdownScaleConfig(window=5, drawdown_limit=0.2)
    )
    assert np.isfinite(scale).all()


def test_factor_returns_survive_a_panel_of_missing_prices() -> None:
    factors = canonical_factor_returns(
        np.full((30, 5), np.nan), np.full((30, 5), np.nan)
    )
    assert np.isnan(factors).all()


def test_factor_returns_survive_non_positive_prices_and_caps() -> None:
    prices, caps = a_panel(dates=60, assets=10)
    prices[10, 0] = -1.0
    caps[10, :] = 0.0
    assert np.isfinite(canonical_factor_returns(prices, caps)[20:, 0]).any()


def test_an_overflowing_book_is_refused_rather_than_capped_to_nonsense() -> None:
    """A book whose own arithmetic overflows has no exposure worth trusting.

    Nothing raises, nothing warns, and the row comes back as it went in rather
    than blanked by an inf/inf or flattened on the strength of an overflow.
    """
    weights = book([1e308, -1e308, 1e307, -1e307, 1e306, -1e306])
    rng = np.random.default_rng(21)
    betas = cube(
        np.vstack(
            [
                rng.normal(1.0, 0.35, 6),
                rng.normal(0.0, 0.6, 6),
                rng.normal(0.0, 0.6, 6),
            ]
        )
    )
    np.testing.assert_allclose(factor_cap_blend(weights, betas, LIMITS), [[0.0]])
    np.testing.assert_allclose(factor_capped(weights, betas, LIMITS), weights)


def test_a_wide_compliant_book_is_left_alone() -> None:
    """Wide enough to hedge, but inside every limit, so nothing is done to it."""
    weights, betas = a_row(77, assets=20)
    small: Vector = np.array([10.0, 10.0, 10.0])
    assert worst(weights, betas, small) <= 1.0
    np.testing.assert_allclose(factor_cap_blend(weights, betas, small), [[0.0]])
    np.testing.assert_allclose(factor_capped(weights, betas, small), weights)


def test_a_wide_book_lying_in_the_span_of_its_own_factors_cannot_be_hedged() -> None:
    """The residual vanishes, so there is no direction to hedge along."""
    rng = np.random.default_rng(31)
    loadings = np.vstack(
        [rng.normal(1.0, 0.35, 5), rng.normal(0.0, 0.6, 5), rng.normal(0.0, 0.6, 5)]
    )
    design = np.vstack([np.ones(5), loadings])
    weights = book(design.T @ np.array([0.05, 0.4, -0.2, 0.3]))
    betas = cube(loadings)
    assert worst(weights, betas) > 1.0, "the fixture must actually breach"
    np.testing.assert_allclose(factor_cap_blend(weights, betas, LIMITS), [[0.0]])
    capped = factor_capped(weights, betas, LIMITS)
    assert worst(capped, betas) <= 1.0 + 1e-9
    ratio = capped[0] / weights[0]
    np.testing.assert_allclose(ratio, ratio[0])


# --- what a mutation of the model would have to break ---------------------


def test_the_momentum_sort_is_formed_before_the_date_it_sorts() -> None:
    """The subtlest look-ahead here: grading the sort on the return it predicts.

    Tested on the formation alone, because the factor return at a date legitimately
    moves when that date's prices move -- it *is* that date's return. What may not
    move is which bucket each name was put in.
    """
    rng = np.random.default_rng(17)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.03, (120, 12)), axis=0)
    before = _momentum_characteristic(prices, 5)

    spiked = prices.copy()
    spiked[-1] *= 5.0
    after = _momentum_characteristic(spiked, 5)
    np.testing.assert_array_equal(np.isnan(before), np.isnan(after))
    np.testing.assert_allclose(before, after, equal_nan=True)


def test_the_momentum_sort_does_move_when_an_earlier_price_does() -> None:
    """The other half: a lag that never reads anything would pass the test above."""
    rng = np.random.default_rng(17)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.03, (120, 12)), axis=0)
    before = _momentum_characteristic(prices, 5)
    spiked = prices.copy()
    spiked[-2] *= 5.0
    assert not np.allclose(before[-1], _momentum_characteristic(spiked, 5)[-1])


def test_the_size_sort_takes_the_outer_thirty_percent() -> None:
    """Ten assets: the small leg is the smallest three, the big leg the largest."""
    rng = np.random.default_rng(19)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, (40, 10)), axis=0)
    caps = np.tile(np.linspace(1e6, 1e9, 10), (40, 1))
    factors = canonical_factor_returns(
        prices, caps, config=CanonicalFactorConfig(minimum_assets=1)
    )
    returns = prices[1:] / prices[:-1] - 1.0
    weights = caps[:-1]
    small = (returns[:, :3] * weights[:, :3]).sum(axis=1) / weights[:, :3].sum(axis=1)
    big = (returns[:, 7:] * weights[:, 7:]).sum(axis=1) / weights[:, 7:].sum(axis=1)
    np.testing.assert_allclose(factors[1:, 1], small - big, atol=1e-12)


def test_a_market_cap_reported_as_zero_takes_the_previous_day_s() -> None:
    """A glitched cap must not drop the name out of every portfolio for the day.

    Filled twice on purpose: once for a missing cap, once for a non-positive
    one. Dropping the second fill silently narrows the cross-section.
    """
    rng = np.random.default_rng(23)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, (40, 8)), axis=0)
    caps = np.tile(np.linspace(1e6, 1e9, 8), (40, 1))
    config = CanonicalFactorConfig(minimum_assets=8)
    clean = canonical_factor_returns(prices, caps, config=config)

    glitched = caps.copy()
    glitched[20, 3] = 0.0
    healed = canonical_factor_returns(prices, glitched, config=config)
    # The carried cap equals the day before's, so the whole row is unchanged and
    # the date still has its full complement of names.
    np.testing.assert_allclose(clean, healed, equal_nan=True)


def test_the_first_beta_arrives_the_row_after_the_minimum_is_met() -> None:
    """Pins the off-by-one: `min_periods` rows of history, then a loading."""
    rng = np.random.default_rng(29)
    config = CanonicalFactorConfig(beta_window=60, beta_min_periods=10)
    betas = rolling_factor_betas(
        rng.normal(0.0, 0.02, (40, 3)), rng.normal(0.0, 0.01, (40, 3)), config=config
    )
    answered = np.flatnonzero(np.isfinite(betas[0, :, 0]))
    assert answered.size, "the panel is long enough to answer somewhere"
    assert int(answered[0]) == config.beta_min_periods


def test_the_backstop_shrink_spares_the_cells_it_cannot_measure() -> None:
    """The uniform shrink is a risk response, and an unmeasured cell carries no risk.

    Exercises the shrink path rather than the hedge path: a nearly constant row
    has no direction to hedge along, so the backstop is what caps it -- and the
    backstop must honour the same promise the hedge does.
    """
    loadings: Cube = np.full((3, 1, 5), np.nan)
    loadings[:, 0, :4] = 1.0
    weights = book([0.5, 0.5, 0.5, 0.5, 4.0])
    assert worst(weights, loadings, np.array([0.5, 0.5, 0.5])) > 1.0
    capped = factor_capped(weights, loadings, np.array([0.5, 0.5, 0.5]))
    assert capped[0, 4] == pytest.approx(4.0)
    assert worst(capped, loadings, np.array([0.5, 0.5, 0.5])) <= 1.0 + 1e-9


def test_a_book_that_loses_everything_is_not_sized_back_up() -> None:
    """Past zero the ratio of equity to peak stops being a drawdown.

    Two negatives divide to a positive, so a wiped-out book reads as recovered
    and comes back at full leverage. It has not recovered; it is gone.
    """
    returns: Vector = np.concatenate([np.array([0.0, -1.5, -0.5]), np.zeros(6)])
    scale = drawdown_scale(returns, DrawdownScaleConfig(window=5, drawdown_limit=0.1))
    assert scale[0] == pytest.approx(1.0)
    np.testing.assert_allclose(scale[1:], 0.0)


def test_a_nearly_constant_book_is_shrunk_rather_than_blown_up() -> None:
    """An equal-weight book has no residual: the intercept accounts for all of it.

    So there is no direction to hedge along, and restoring the gross would
    multiply whatever numerical noise is left by an unbounded factor. It is
    shrunk instead -- which reaches the limit, and says so by blending zero.
    """
    rng = np.random.default_rng(41)
    assets = 40
    loadings = cube(
        np.vstack(
            [
                rng.normal(1.0, 0.4, assets),
                rng.normal(0.0, 0.4, assets),
                rng.normal(0.0, 0.4, assets),
            ]
        )
    )
    flat = book([1.0 / assets] * assets)
    limits: Vector = np.array([0.3, 0.2, 0.2])
    assert worst(flat, loadings, limits) > 1.0
    np.testing.assert_allclose(factor_cap_blend(flat, loadings, limits), [[0.0]])
    capped = factor_capped(flat, loadings, limits)
    assert worst(capped, loadings, limits) <= 1.0 + 1e-9
    # Uniformly, so every name keeps its share of the book.
    ratio = capped[0] / flat[0]
    np.testing.assert_allclose(ratio, ratio[0])


def test_restoring_the_gross_never_stretches_a_row_without_bound() -> None:
    """The guard is a ratio, so two rows a rounding error apart cannot diverge."""
    rng = np.random.default_rng(43)
    assets = 40
    loadings = cube(
        np.vstack(
            [
                rng.normal(1.0, 0.4, assets),
                rng.normal(0.0, 0.4, assets),
                rng.normal(0.0, 0.4, assets),
            ]
        )
    )
    limits: Vector = np.array([0.3, 0.2, 0.2])
    flat = np.full(assets, 1.0 / assets)
    for dispersion in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1):
        raw = np.abs(flat + dispersion * rng.normal(0.0, 1.0, assets))
        weights = book(raw / raw.sum())
        capped = factor_capped(weights, loadings, limits)
        stretch = float(np.abs(capped).max()) / float(np.abs(weights).max())
        assert stretch <= 4.0 + 1e-9, f"dispersion {dispersion} stretched {stretch}x"
        assert worst(capped, loadings, limits) <= 1.0 + 1e-9


def test_an_infinite_market_cap_is_carried_over_rather_than_used() -> None:
    """A cap that cannot be a weight is one the model does not have.

    The model this ports ranks an infinity highest, which ranks fine and then
    overflows the value-weighted sum it is used in, taking the whole date with
    it. Treated as missing it takes the previous day's cap, like every other
    unusable reading -- and the date survives.
    """
    rng = np.random.default_rng(47)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, (60, 8)), axis=0)
    caps = np.tile(np.linspace(1e6, 1e9, 8), (60, 1))
    config = CanonicalFactorConfig(minimum_assets=8)
    clean = canonical_factor_returns(prices, caps, config=config)

    glitched = caps.copy()
    glitched[30, -1] = np.inf
    healed = canonical_factor_returns(prices, glitched, config=config)
    np.testing.assert_allclose(clean, healed, equal_nan=True)
    assert np.isfinite(healed[31:, 0]).any(), "the date must survive the glitch"
