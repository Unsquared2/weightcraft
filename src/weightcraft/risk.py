"""Risk-based sizing: trailing volatility, inverse-vol, equal risk contribution."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from weightcraft.normalize import normalised_share, row_sums

if TYPE_CHECKING:
    from weightcraft.arrays import BoolMatrix, Cube, Matrix, Vector

_MINIMUM_COVARIANCE_ROWS = 3
_MINIMUM_STD_OBSERVATIONS = 2
_DEGENERATE_SLICE_WARNING = "Mean of empty slice"
_BLOCK_ROWS = 4096


@dataclass(frozen=True, slots=True)
class EqualRiskConfig:
    """The knobs of the equal-risk fixed point, as one immutable value."""

    period: int
    rebalance: int = 1
    iterations: int = 50
    shrinkage: float = 0.5


@dataclass(frozen=True, slots=True)
class VolatilityTargetConfig:
    """The knobs of volatility targeting, as one immutable value."""

    period: int
    target: float
    periods_per_year: int = 365
    max_leverage: float = 3.0
    portfolio_leverage: float = 1.0


def rolling_sums(values: Vector, period: int) -> Vector:
    """Trailing sums of `period` observations, the head NaN until the window fills.

    Windowed rather than differenced from a cumulative sum: a single NaN in a
    cumulative sum poisons every window after it, not just the ones containing
    it.
    """
    if period < 1:
        msg = f"period must be at least 1, got {period}"
        raise ValueError(msg)
    out: Vector = np.full(values.size, np.nan)
    if values.size < period:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, period)
    out[period - 1 :] = windows.sum(axis=-1)
    return out


def trailing_std(series: Vector, period: int) -> Vector:
    """`rolling(period).std(ddof=1, min_periods=max(period, 2))` over a series.

    Centred on the window's own mean rather than differenced from cumulative
    sums of squares: the latter cancels catastrophically once the level dwarfs
    the variation, and reports a confident zero for a series like `1e10 + noise`.
    """
    if period < _MINIMUM_STD_OBSERVATIONS:
        msg = f"period must be at least {_MINIMUM_STD_OBSERVATIONS}, got {period}"
        raise ValueError(msg)
    out: Vector = np.full(series.size, np.nan)
    if series.size < period:
        return out

    finite = np.where(np.isfinite(series), series, np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(finite, period)
    counts = np.isfinite(windows).sum(axis=-1)

    # A `sliding_window_view` is a view, but `(windows - mean) ** 2` is not:
    # materialising it whole costs `rows x period` floats, which is 300 MB for
    # a 200k-row series at a 90-bar window. Blocked, the working set is fixed.
    squared: Vector = np.empty(windows.shape[0], dtype=np.float64)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", over="ignore"):
        warnings.filterwarnings("ignore", message=_DEGENERATE_SLICE_WARNING)
        for start in range(0, windows.shape[0], _BLOCK_ROWS):
            block = windows[start : start + _BLOCK_ROWS]
            mean = np.nanmean(block, axis=-1, keepdims=True)
            squared[start : start + _BLOCK_ROWS] = np.nansum(
                (block - mean) ** 2, axis=-1
            )

    with np.errstate(invalid="ignore", divide="ignore"):
        deviation = np.sqrt(squared / np.maximum(counts - 1.0, 1.0))
    out[period - 1 :] = np.where(counts >= _MINIMUM_STD_OBSERVATIONS, deviation, np.nan)
    return out


def downside_deviation(returns: Matrix, period: int) -> Matrix:
    """Trailing `sqrt(sum(min(r, 0)^2) / n_losing)`; NaN when nothing lost."""
    if returns.shape[1] == 0:
        empty: Matrix = np.zeros(returns.shape, dtype=np.float64)
        return empty
    losses = np.where(np.isfinite(returns) & (returns < 0.0), returns, 0.0)
    losing = (losses < 0.0).astype(np.float64)
    with np.errstate(over="ignore"):
        squared_losses = losses**2
    squared = np.stack(
        [rolling_sums(squared_losses[:, i], period) for i in range(returns.shape[1])],
        axis=1,
    )
    counts = np.stack(
        [rolling_sums(losing[:, i], period) for i in range(returns.shape[1])], axis=1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        deviation: Matrix = np.where(counts > 0.0, np.sqrt(squared / counts), np.nan)
    return deviation


def inverse_volatility_share(deviation: Matrix, side: Matrix) -> Matrix:
    """One side of the book, weighted by the inverse of its share of the risk.

    Only `+inf` is zeroed, never `-inf`: a negative deviation is not a
    zero-volatility name, it is a broken input, and it should stay visible.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        deviations = np.where(np.isnan(side), np.nan, deviation)
        inverse = 1.0 / (deviations / row_sums(deviations))
        inverse = np.where(inverse == np.inf, 0.0, inverse)
        share = inverse / row_sums(inverse)
    weighted: Matrix = share * (~np.isnan(side)).sum(axis=1, keepdims=True)
    return weighted


def observed_covariance(window: Matrix) -> Matrix:
    """Pairwise-complete covariance, so a gap costs one pair rather than a row.

    A window too short to have a covariance answers zeros rather than warning
    about its own degrees of freedom.
    """
    columns = window.shape[1]
    if window.shape[0] < _MINIMUM_STD_OBSERVATIONS or columns == 0:
        degenerate: Matrix = np.zeros((max(columns, 1), max(columns, 1)))
        return degenerate

    present = np.isfinite(window)
    if bool(present.all()):
        with np.errstate(over="ignore", invalid="ignore"):
            complete: Matrix = np.atleast_2d(np.cov(window, rowvar=False))
        return complete

    counted = present.astype(np.float64)
    filled = np.where(present, window, 0.0)

    with np.errstate(over="ignore", invalid="ignore"):
        counts = counted.T @ counted
        means = (filled.T @ counted) / np.maximum(counts, 1.0)
        covariance = (filled.T @ filled - counts * means * means.T) / np.maximum(
            counts - 1.0, 1.0
        )
    pairwise: Matrix = np.where(counts > 1.0, covariance, 0.0)
    return pairwise


def penalised_for_coverage(covariance: Matrix, coverage: Vector) -> Matrix:
    """Blend each variance toward the riskiest name in proportion to missing history.

    Correlations are preserved; only the scale of a thinly observed name moves,
    so a newcomer is treated as risky rather than as reliably calm.
    """
    if bool(np.all(coverage >= 1.0)):
        return covariance

    variance = np.diag(covariance)
    worst = float(np.max(variance))
    if not np.isfinite(worst) or worst <= 0.0:
        return covariance

    deviation = np.sqrt(np.maximum(variance, 0.0))
    reciprocal = np.where(
        deviation > 0.0, 1.0 / np.where(deviation > 0.0, deviation, 1.0), 0.0
    )
    correlation = covariance * np.outer(reciprocal, reciprocal)
    np.fill_diagonal(correlation, 1.0)

    penalised = np.sqrt(coverage * variance + (1.0 - coverage) * worst)
    scaled: Matrix = correlation * np.outer(penalised, penalised)
    return scaled


def equal_risk_row(
    window: Matrix,
    holdings: Vector,
    *,
    iterations: int = 50,
    shrinkage: float = 0.5,
) -> Vector:
    """Sizes whose risk contributions are equal, by damped fixed point.

    The update is the geometric mean of the old and new weight; the undamped
    `w <- sigma^2 / (n * (Sigma w)_i)` oscillates and never lands. The marginal
    is floored because a sample covariance over a few hundred bars is not
    positive semi-definite, and `(Sigma w)_i` goes negative for the calmest name
    on the first iteration.
    """
    sizes: Vector = np.zeros(holdings.shape[0])
    chosen = np.flatnonzero(holdings)
    if chosen.size == 0:
        return sizes
    if window.shape[0] < _MINIMUM_COVARIANCE_ROWS or chosen.size == 1:
        sizes[chosen] = 1.0
        return sizes

    observed = window[:, chosen]
    covariance = penalised_for_coverage(
        observed_covariance(observed), np.isfinite(observed).mean(axis=0)
    )
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    floor = max(np.trace(covariance) / covariance.shape[0], 1e-18) * 1e-6

    weight = np.full(chosen.size, 1.0 / chosen.size)
    for _ in range(iterations):
        marginal = np.maximum(covariance @ weight, floor)
        variance = float(weight @ covariance @ weight)
        if not np.isfinite(variance) or variance <= 0:
            break
        updated = np.sqrt(weight * variance / (chosen.size * marginal))
        total = float(updated.sum())
        if not np.isfinite(total) or total <= 0:
            break
        weight = updated / total

    sizes[chosen] = weight
    return sizes


def equal_risk_weights(
    returns: Matrix, holdings: BoolMatrix, config: EqualRiskConfig
) -> Matrix:
    """Equal-risk sizes per date, recomputing the covariance every `rebalance` rows.

    The window is inclusive of the current row, and a size is carried forward
    between rebalances rather than recomputed.
    """
    if config.rebalance < 1:
        msg = f"rebalance must be at least 1, got {config.rebalance}"
        raise ValueError(msg)
    rows, columns = returns.shape
    sizes: Matrix = np.zeros((rows, columns))
    carried: Vector = np.zeros(columns)
    for row in range(rows):
        if row % config.rebalance == 0:
            carried = equal_risk_row(
                returns[max(0, row - config.period + 1) : row + 1],
                holdings[row].astype(np.float64),
                iterations=config.iterations,
                shrinkage=config.shrinkage,
            )
        sizes[row] = np.where(holdings[row], carried, 0.0)
    return normalised_share(sizes, holdings, power=1.0)


def volatility_target(book_returns: Vector, config: VolatilityTargetConfig) -> Vector:
    """A per-date scalar taking the book's trailing volatility to `target`.

    `portfolio_leverage` divides the target because this runs before the
    portfolio's own leverage and cap and cannot see them: left at 1.0 against a
    2x book, a 30% target realises 60%.
    """
    realised = trailing_std(book_returns, config.period) * np.sqrt(
        config.periods_per_year
    )
    wanted = config.target / max(config.portfolio_leverage, 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(realised > 0, wanted / realised, 1.0)
    # A window the book barely moved in asks for enormous leverage, which is an
    # artefact of the estimate rather than an opportunity.
    clipped: Vector = np.clip(np.nan_to_num(scale, nan=1.0), 0.0, config.max_leverage)
    return clipped


def stack_volatility(stack: Cube, *, ddof: int = 1) -> Vector:
    """The standard deviation of each frame in a stack, over every present cell."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        deviations = [
            float(np.nanstd(frame[np.isfinite(frame)], ddof=ddof))
            if np.isfinite(frame).sum() > ddof
            else np.nan
            for frame in stack
        ]
    out: Vector = np.asarray(deviations, dtype=np.float64)
    return out
