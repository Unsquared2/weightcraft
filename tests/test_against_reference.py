"""Every kernel against a second, deliberately naive implementation.

The vectorised versions here are fast because they avoid the obvious loop. That
is exactly what makes them easy to get subtly wrong, so each one is checked
against the obvious loop, written out in this file and owing nothing to the
implementation it checks. This is the pattern empyrical uses against pandas and
PyPortfolioOpt uses against hand-computed matrices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from canonical import NOISE, SPARSE_NOISE, correlated_panel, returns_panel
from weightcraft.band import no_trade_band
from weightcraft.costs import turnover
from weightcraft.cross_section import (
    project_out_rows,
    row_rank_pct,
    standardize_rows,
    top_n_mask,
)
from weightcraft.metrics import (
    beta,
    compounded,
    drawdown,
    max_drawdown,
    sharpe,
    to_prices,
)
from weightcraft.normalize import capped, gross, to_gross
from weightcraft.risk import (
    equal_risk_row,
    inverse_volatility_share,
    observed_covariance,
    rolling_sums,
    trailing_std,
)
from weightcraft.smoothing import ewm_mean, lag_rows, rolling_mean

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix, Vector

# --------------------------------------------------------------------------
# The naive implementations. Loops, no cleverness, no shared helpers.
# --------------------------------------------------------------------------


def naive_rolling_sums(values: Vector, period: int) -> Vector:
    out = np.full(values.size, np.nan)
    for end in range(period, values.size + 1):
        out[end - 1] = float(np.sum(values[end - period : end]))
    return out


def naive_trailing_std(series: Vector, period: int) -> Vector:
    out = np.full(series.size, np.nan)
    for end in range(period, series.size + 1):
        window = series[end - period : end]
        seen = window[np.isfinite(window)]
        if seen.size >= 2:
            out[end - 1] = float(np.std(seen, ddof=1))
    return out


def naive_rolling_mean(values: Matrix, window: int) -> Matrix:
    rows, columns = values.shape
    out = np.full((rows, columns), np.nan)
    for column in range(columns):
        for end in range(window, rows + 1):
            block = values[end - window : end, column]
            if np.isfinite(block).all():
                out[end - 1, column] = float(np.mean(block))
    return out


def naive_ewm_mean(values: Matrix, span: int) -> Matrix:
    """`ewm(span, min_periods=span, adjust=True)`: a weighted sum, written out."""
    decay = 1.0 - 2.0 / (span + 1.0)
    rows, columns = values.shape
    out = np.full((rows, columns), np.nan)
    for column in range(columns):
        seen = 0
        for row in range(rows):
            numerator = 0.0
            denominator = 0.0
            for back in range(row + 1):
                value = values[row - back, column]
                if np.isfinite(value):
                    numerator += decay**back * value
                    denominator += decay**back
            if np.isfinite(values[row, column]):
                seen += 1
            if seen >= span and denominator > 0.0:
                out[row, column] = numerator / denominator
    return out


def naive_turnover(weights: Matrix) -> Matrix:
    rows, columns = weights.shape
    out = np.zeros((rows, columns))
    for column in range(columns):
        for row in range(1, rows):
            now = weights[row, column]
            before = weights[row - 1, column]
            out[row, column] = abs(
                (now if np.isfinite(now) else 0.0)
                - (before if np.isfinite(before) else 0.0)
            )
    return out


def naive_no_trade_band(target: Matrix, band: float) -> Matrix:
    """A literal port of the scalar, row-by-row loop this function replaces."""
    rows, columns = target.shape
    out = np.full((rows, columns), np.nan)
    current = [float("nan")] * columns
    for row in range(rows):
        wanted = [float(v) for v in target[row]]
        seen = [abs(v) for v in wanted if np.isfinite(v)]
        scale = sum(seen) / len(seen) if seen else float("nan")
        if not np.isfinite(scale) or scale <= 0.0:
            current = wanted
        else:
            for column in range(columns):
                before = current[column]
                after = wanted[column]
                gap = not np.isfinite(before) or not np.isfinite(after)
                if gap or abs(after - before) > band * scale:
                    current[column] = after
        out[row] = current
    return out


def naive_capped(
    values: Matrix, *, max_gross: float | None, max_net: float | None
) -> Matrix:
    """Row by row, cell by cell: no broadcasting, no keepdims."""
    rows, columns = values.shape
    out = np.zeros((rows, columns))
    for row in range(rows):
        cells = [float(v) for v in values[row]]
        finite = [v for v in cells if np.isfinite(v)]
        row_gross = sum(abs(v) for v in finite)
        row_net = sum(finite)
        scale = 1.0
        if max_gross is not None and row_gross > 0.0:
            scale = min(scale, max_gross / row_gross)
        if max_net is not None and abs(row_net) > 0.0:
            scale = min(scale, max_net / abs(row_net))
        for column in range(columns):
            out[row, column] = cells[column] * scale
    return out


def naive_pairwise_covariance(window: Matrix) -> Matrix:
    columns = window.shape[1]
    out = np.zeros((columns, columns))
    for i in range(columns):
        for j in range(columns):
            both = np.isfinite(window[:, i]) & np.isfinite(window[:, j])
            if both.sum() > 1:
                left = window[both, i]
                right = window[both, j]
                out[i, j] = float(
                    np.sum((left - left.mean()) * (right - right.mean()))
                    / (both.sum() - 1)
                )
    return out


def naive_z_score(values: Matrix, ddof: int = 0) -> Matrix:
    out = np.full(values.shape, np.nan)
    for row in range(values.shape[0]):
        seen = [v for v in values[row] if np.isfinite(v)]
        if len(seen) <= ddof:
            continue
        mean = sum(seen) / len(seen)
        variance = sum((v - mean) ** 2 for v in seen) / (len(seen) - ddof)
        deviation = variance**0.5
        for column, value in enumerate(values[row]):
            if np.isfinite(value):
                out[row, column] = (value - mean) / deviation if deviation else 0.0
    return out


def naive_rank_pct(values: Matrix) -> Matrix:
    out = np.full(values.shape, np.nan)
    for row in range(values.shape[0]):
        seen = [(v, c) for c, v in enumerate(values[row]) if np.isfinite(v)]
        if not seen:
            continue
        for position, (_, column) in enumerate(sorted(seen), start=1):
            out[row, column] = position / len(seen)
    return out


def naive_prices(returns: Vector, base: float = 1.0) -> Vector:
    out = np.full(returns.size, np.nan)
    price = base
    for index, value in enumerate(returns):
        if np.isfinite(value):
            price *= 1.0 + value
            out[index] = price
    return out


def naive_max_drawdown(returns: Vector) -> float:
    prices = naive_prices(returns)
    peak = -np.inf
    worst = 0.0
    for price in prices:
        if not np.isfinite(price):
            continue
        peak = max(peak, price)
        worst = min(worst, price / peak - 1.0)
    return worst


def naive_sharpe(returns: Vector, period: int) -> float:
    seen = returns[np.isfinite(returns)]
    if seen.size < 2:
        return -10.0
    deviation = float(np.std(seen, ddof=1))
    if deviation == 0.0:
        return -10.0 if float(np.mean(seen)) == 0.0 else float("inf")
    return float(float(np.mean(seen)) / deviation * period**0.5)


def naive_beta(returns: Vector, underlying: Vector) -> float:
    both = np.isfinite(returns) & np.isfinite(underlying)
    left, right = returns[both], underlying[both]
    if left.size < 2:
        return float("nan")
    covariance = float(np.sum((left - left.mean()) * (right - right.mean())))
    variance = float(np.sum((right - right.mean()) ** 2))
    return float(covariance / variance) if variance else float("nan")


def naive_equal_risk(window: Matrix, iterations: int = 5000) -> Vector:
    """Cyclical coordinate descent, the standard alternative ERC solver.

    A completely different algorithm from the damped fixed point under test:
    it solves `sigma_ii w_i^2 + (sum_j!=i sigma_ij w_j) w_i - b_i = 0` for one
    weight at a time. Agreement between the two is strong evidence both are
    solving the risk-parity problem rather than sharing a mistake.
    """
    covariance = np.cov(window, rowvar=False)
    size = covariance.shape[0]
    budget = 1.0 / size
    weight: Vector = np.full(size, 1.0 / size)
    for _ in range(iterations):
        for i in range(size):
            others = float(covariance[i] @ weight - covariance[i, i] * weight[i])
            variance = float(covariance[i, i])
            weight[i] = (
                -others + np.sqrt(others * others + 4.0 * variance * budget)
            ) / (2.0 * variance)
    solved: Vector = weight / weight.sum()
    return solved


# --------------------------------------------------------------------------
# The comparisons.
# --------------------------------------------------------------------------

WINDOWS = pytest.mark.parametrize("period", [2, 3, 7, 30])


@WINDOWS
@pytest.mark.parametrize("series", [NOISE, SPARSE_NOISE], ids=["clean", "gapped"])
def test_rolling_sums_match_the_obvious_loop(series: Vector, period: int) -> None:
    assert np.allclose(
        rolling_sums(series, period),
        naive_rolling_sums(series, period),
        equal_nan=True,
    )


@WINDOWS
@pytest.mark.parametrize("series", [NOISE, SPARSE_NOISE], ids=["clean", "gapped"])
def test_trailing_std_matches_the_obvious_loop(series: Vector, period: int) -> None:
    assert np.allclose(
        trailing_std(series, period),
        naive_trailing_std(series, period),
        equal_nan=True,
        rtol=1e-12,
    )


@pytest.mark.parametrize("magnitude", [1.0, 1e6, 1e10, 1e-10])
def test_trailing_std_matches_the_loop_at_every_scale(magnitude: float) -> None:
    series: Vector = NOISE[:80] + magnitude
    assert np.allclose(
        trailing_std(series, 20),
        naive_trailing_std(series, 20),
        equal_nan=True,
        rtol=1e-9,
    )


@pytest.mark.parametrize("window", [1, 2, 5])
def test_a_rolling_mean_matches_the_obvious_loop(window: int) -> None:
    values: Matrix = np.column_stack([NOISE[:40], SPARSE_NOISE[:40]])
    assert np.allclose(
        rolling_mean(values, window),
        naive_rolling_mean(values, window),
        equal_nan=True,
    )


@pytest.mark.parametrize("span", [1, 2, 3, 6])
def test_an_ewm_mean_matches_the_weighted_sum_it_claims_to_be(span: int) -> None:
    values: Matrix = np.column_stack([NOISE[:30], SPARSE_NOISE[:30]])
    assert np.allclose(
        ewm_mean(values, span), naive_ewm_mean(values, span), equal_nan=True
    )


def test_turnover_matches_the_obvious_loop() -> None:
    weights: Matrix = np.column_stack([NOISE[:40], SPARSE_NOISE[:40]])
    assert np.allclose(turnover(weights), naive_turnover(weights))


@pytest.mark.parametrize("gaps", [False, True], ids=["complete", "gapped"])
def test_a_covariance_matrix_matches_the_pairwise_loop(*, gaps: bool) -> None:
    window = returns_panel(21, 90, [0.01, 0.02, 0.04])
    if gaps:
        window = window.copy()
        window[10:20, 1] = np.nan
        window[30:33, 2] = np.nan
    assert np.allclose(observed_covariance(window), naive_pairwise_covariance(window))


def test_a_complete_covariance_matrix_matches_numpy() -> None:
    window = returns_panel(22, 120, [0.01, 0.03])
    assert np.allclose(observed_covariance(window), np.cov(window, rowvar=False))


@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_a_z_score_matches_the_obvious_loop(ddof: int) -> None:
    values: Matrix = np.asarray(
        [
            [1.0, 2.0, 5.0],
            [np.nan, 4.0, 6.0],
            [3.0, 3.0, 3.0],
            [np.nan] * 3,
            [7.0, np.nan, np.nan],
        ]
    )
    assert np.allclose(
        standardize_rows(values, ddof=ddof),
        naive_z_score(values, ddof=ddof),
        equal_nan=True,
    )


def test_a_percentile_rank_matches_the_obvious_loop() -> None:
    values: Matrix = np.asarray(
        [[3.0, 1.0, 2.0], [np.nan, 5.0, 5.0], [1.0, np.nan, np.nan]]
    )
    assert np.allclose(row_rank_pct(values), naive_rank_pct(values), equal_nan=True)


def test_the_top_n_mask_matches_sorting_the_row() -> None:
    values: Matrix = np.column_stack([NOISE[:20], SPARSE_NOISE[:20], NOISE[20:40]])
    for top_n in (1, 2, 3):
        expected = np.zeros(values.shape, dtype=np.bool_)
        for row in range(values.shape[0]):
            present = [c for c in range(values.shape[1]) if np.isfinite(values[row, c])]
            chosen = sorted(present, key=lambda c: -values[row, c])[:top_n]
            expected[row, chosen] = True
        assert np.array_equal(top_n_mask(values, top_n), expected)


def test_a_projection_matches_least_squares() -> None:
    """`project_out_rows` must agree with `np.linalg.lstsq` on the same system."""
    generator = np.random.default_rng(23)
    controls = generator.normal(size=(2, 4, 25))
    target: Matrix = generator.normal(size=(4, 25))
    residual = project_out_rows(target, controls)
    for row in range(target.shape[0]):
        design = np.column_stack([np.ones(25), controls[0, row], controls[1, row]])
        coefficients, *_ = np.linalg.lstsq(design, target[row], rcond=None)
        assert np.allclose(
            residual[row], target[row] - design @ coefficients, atol=1e-9
        )


def test_prices_match_the_obvious_loop() -> None:
    assert np.allclose(to_prices(NOISE[:50]), naive_prices(NOISE[:50]), equal_nan=True)


def test_the_worst_drawdown_matches_the_obvious_loop() -> None:
    for series in (NOISE[:80], SPARSE_NOISE[:80]):
        assert max_drawdown(series) == pytest.approx(naive_max_drawdown(series))


def test_the_drawdown_series_never_exceeds_the_worst_point() -> None:
    series = NOISE[:80]
    below = drawdown(series)
    assert float(np.nanmin(below)) == pytest.approx(max_drawdown(series))


@pytest.mark.parametrize("period", [1, 12, 252, 365])
def test_sharpe_matches_the_obvious_loop(period: int) -> None:
    for series in (NOISE, SPARSE_NOISE):
        assert sharpe(series, period) == pytest.approx(naive_sharpe(series, period))


def test_beta_matches_the_obvious_loop() -> None:
    assert beta(SPARSE_NOISE, NOISE) == pytest.approx(naive_beta(SPARSE_NOISE, NOISE))


def test_compounding_matches_the_obvious_loop() -> None:
    total = 1.0
    for value in SPARSE_NOISE:
        if np.isfinite(value):
            total *= 1.0 + value
    assert compounded(SPARSE_NOISE) == pytest.approx(total - 1.0)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_equal_risk_agrees_with_coordinate_descent(rho: float) -> None:
    window = correlated_panel(24, 1500, rho=rho)
    assert np.allclose(
        equal_risk_row(window, np.ones(3), shrinkage=0.0),
        naive_equal_risk(window),
        atol=0.02,
    )


def test_equal_risk_actually_equalises_what_coordinate_descent_equalises() -> None:
    window = correlated_panel(25, 1500, rho=0.6)
    covariance = np.cov(window, rowvar=False)
    for weights in (
        equal_risk_row(window, np.ones(3), shrinkage=0.0),
        naive_equal_risk(window),
    ):
        contributions = weights * (covariance @ weights)
        assert np.allclose(contributions / contributions.sum(), 1 / 3, atol=0.03)


def test_inverse_volatility_matches_the_reciprocal_by_hand() -> None:
    deviation: Matrix = np.asarray([[0.01, 0.02, 0.04]])
    reciprocal = 1.0 / deviation[0]
    expected = reciprocal / reciprocal.sum() * 3.0
    assert np.allclose(
        inverse_volatility_share(deviation, np.ones((1, 3)))[0], expected
    )


def test_rescaling_to_a_gross_target_matches_dividing_by_hand() -> None:
    book: Matrix = np.asarray([[0.2, -0.6, 0.2]])
    assert np.allclose(to_gross(book, 1.0)[0], book[0] / 1.0)
    assert float(gross(to_gross(book, 2.5))[0, 0]) == pytest.approx(2.5)


@pytest.mark.parametrize("periods", [0, 1, 3])
def test_lagging_matches_slicing_by_hand(periods: int) -> None:
    values: Matrix = np.arange(12, dtype=np.float64).reshape(6, 2)
    shifted = lag_rows(values, periods)
    assert np.array_equal(shifted[periods:], values[: 6 - periods])
    assert np.isnan(shifted[:periods]).all()


@pytest.mark.parametrize("band", [0.05, 0.2, 0.5, 2.0])
def test_no_trade_band_matches_the_scalar_loop(band: float) -> None:
    generator = np.random.default_rng(41)
    target: Matrix = generator.normal(size=(60, 5)) * 0.1
    target[generator.random(target.shape) < 0.15] = np.nan
    target[generator.random(target.shape) < 0.05] = 0.0
    assert np.array_equal(
        no_trade_band(target, band), naive_no_trade_band(target, band), equal_nan=True
    )


def test_capped_matches_the_row_by_row_loop() -> None:
    generator = np.random.default_rng(53)
    target: Matrix = generator.normal(size=(60, 5)) * 0.3
    target[generator.random(target.shape) < 0.1] = np.nan
    assert np.allclose(
        capped(target, max_gross=0.5, max_net=0.2),
        naive_capped(target, max_gross=0.5, max_net=0.2),
        equal_nan=True,
    )
