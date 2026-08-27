"""Scalar performance statistics: growth, risk-adjusted return, drawdown."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from weightcraft.arrays import Vector

DEGENERATE_SHARPE = -10.0
"""What a return series with no variation at all scores, instead of NaN."""

_MINIMUM_COVARIANCE_OBSERVATIONS = 2


def compounded(returns: Vector) -> float:
    """`prod(1 + r) - 1`, skipping missing observations."""
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.nanprod(1.0 + returns) - 1.0)


def to_prices(returns: Vector, base: float = 1.0) -> Vector:
    """Compound a return series into a price series starting at `base`.

    A missing return leaves the price flat; an infinite one leaves a hole,
    rather than poisoning every price after it.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        steps = np.where(np.isfinite(returns), 1.0 + returns, np.nan)
        prices: Vector = base * np.nancumprod(np.where(np.isnan(steps), 1.0, steps))
    return np.where(np.isnan(steps), np.nan, prices)


def cagr(returns: Vector, days: float, periods: int = 365) -> float:
    """Compound annual growth rate over `days` calendar days."""
    if days <= 0.0:
        msg = f"days must be positive, got {days}"
        raise ValueError(msg)
    return float(np.abs(compounded(returns) + 1.0) ** (periods / days) - 1.0)


def sharpe(returns: Vector, annualization_period: int = 365) -> float:
    """Annualised mean over standard deviation, `ddof=1`.

    A series that never varies has no Sharpe ratio; it scores
    `DEGENERATE_SHARPE` so a sort puts it last instead of dropping it.
    """
    seen = returns[np.isfinite(returns)]
    if seen.size < _MINIMUM_COVARIANCE_OBSERVATIONS:
        return DEGENERATE_SHARPE
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        ratio = np.mean(seen) / np.std(seen, ddof=1)
    if not np.isfinite(ratio):
        return DEGENERATE_SHARPE
    return float(ratio * np.sqrt(annualization_period))


def beta(returns: Vector, underlying: Vector) -> float:
    """Slope of `returns` on `underlying`, over the dates both observed."""
    both = np.isfinite(returns) & np.isfinite(underlying)
    if both.sum() < _MINIMUM_COVARIANCE_OBSERVATIONS:
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = np.cov(returns[both], underlying[both])
    denominator = float(covariance[1, 1])
    if denominator == 0.0:
        return float("nan")
    return float(covariance[0, 1] / denominator)


def drawdown(returns: Vector) -> Vector:
    """Price over its running peak, minus one -- zero or negative throughout."""
    prices = to_prices(returns)
    peak = np.maximum.accumulate(np.where(np.isnan(prices), -np.inf, prices))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        series = prices / peak - 1.0
    cleaned: Vector = np.where(np.isfinite(series), series, np.nan) + 0.0
    return cleaned


def max_drawdown(returns: Vector) -> float:
    """The worst point of the drawdown series, or 0.0 if it never fell."""
    series = drawdown(returns)
    if not np.isfinite(series).any():
        return 0.0
    return float(np.nanmin(series))
