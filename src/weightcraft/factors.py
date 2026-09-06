"""Cryptocurrency factor risk: the canonical three factors, and capping a book.

The factor definitions follow Liu, Tsyvinski and Wu's cryptocurrency
three-factor model -- a value-weighted market return, small-minus-big, and a
winner-minus-loser momentum spread -- with per-asset loadings from a rolling
regression that reads only observations strictly earlier than the date it
answers for.

Where a neutraliser drives a book's exposure to zero, `factor_capped` moves it
only as far as a declared limit requires: a deliberate tilt survives, an
accidental one is trimmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from weightcraft.cross_section import project_out_rows, row_rank_pct

if TYPE_CHECKING:
    from weightcraft.arrays import BoolMatrix, BoolVector, Cube, Matrix, Vector

CANONICAL_FACTORS = ("market", "size", "momentum")
"""The three factor columns `canonical_factor_returns` returns, in order."""

_TAIL_FRACTION = 0.30
"""The size and momentum sorts take the outer 30% of each cross-section."""

_HALF = 0.50
_BETA_CUBE_RANK = 3
_BISECTION_PASSES = 40
"""Enough to resolve the blend to well past float64's useful precision."""

_MAXIMUM_AMPLIFICATION = 4.0
"""How far restoring the gross may stretch the hedged row before it is refused.

Restoring the original gross multiplies the residual by `gross / gross(residual)`,
so a row whose residual is a sliver of itself comes back as that sliver blown up
-- a book with no relation to its input. Refusing above this bounds the stretch;
below it the row is shrunk uniformly instead, which always reaches the limit.

The bound has to be a ratio rather than a floor near zero. A floor of `1e-6`
admits a millionfold stretch, and puts a discontinuity there: two rows a
rounding error apart come back one shrunk and one stretched.
"""


@dataclass(frozen=True, slots=True)
class CanonicalFactorConfig:
    """The knobs of the canonical model, as one immutable value."""

    beta_window: int = 180
    beta_min_periods: int = 90
    momentum_window: int = 21
    minimum_assets: int = 6

    def __post_init__(self) -> None:
        terms = len(CANONICAL_FACTORS) + 1
        if self.beta_window < terms:
            msg = f"beta_window must be at least {terms}, got {self.beta_window}"
            raise ValueError(msg)
        if not terms <= self.beta_min_periods <= self.beta_window:
            msg = (
                f"beta_min_periods must be between {terms} and beta_window, "
                f"got {self.beta_min_periods}"
            )
            raise ValueError(msg)
        if self.momentum_window < 1:
            msg = f"momentum_window must be positive, got {self.momentum_window}"
            raise ValueError(msg)
        if self.minimum_assets < 1:
            msg = f"minimum_assets must be positive, got {self.minimum_assets}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DrawdownScaleConfig:
    """The knobs of drawdown-limited sizing, as one immutable value."""

    window: int
    drawdown_limit: float
    max_leverage: float = 1.0

    def __post_init__(self) -> None:
        if self.window < 1:
            msg = f"window must be positive, got {self.window}"
            raise ValueError(msg)
        if self.drawdown_limit <= 0.0:
            msg = f"drawdown_limit must be positive, got {self.drawdown_limit}"
            raise ValueError(msg)
        if self.max_leverage <= 0.0:
            msg = f"max_leverage must be positive, got {self.max_leverage}"
            raise ValueError(msg)


def _shifted(values: Matrix, periods: int) -> Matrix:
    """Every row moved `periods` down, the head left missing."""
    out: Matrix = np.full_like(values, np.nan)
    if periods < values.shape[0]:
        out[periods:] = values[: values.shape[0] - periods]
    return out


def _forward_fill(values: Matrix) -> Matrix:
    """Each column carried down over its gaps; a leading gap stays missing."""
    observed = ~np.isnan(values)
    rows = np.where(observed, np.arange(values.shape[0])[:, None], 0)
    np.maximum.accumulate(rows, axis=0, out=rows)
    filled: Matrix = np.take_along_axis(values, rows, axis=0)
    return filled


def _percentage_change(values: Matrix, periods: int = 1) -> Matrix:
    """Row-over-row relative change, gaps propagating rather than filling."""
    previous = _shifted(values, periods)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        change = values / previous - 1.0
    out: Matrix = np.where(np.isfinite(change), change, np.nan)
    return out


def _value_weighted_return(
    returns: Matrix, weights: Matrix, members: BoolMatrix, minimum_assets: int
) -> Vector:
    """The weighted mean return of the members carrying a positive weight.

    A date with fewer than `minimum_assets` qualifying names is not a portfolio,
    and comes back missing rather than as a thin average.
    """
    valid = members & ~np.isnan(returns) & ~np.isnan(weights) & (weights > 0.0)
    counted = valid.sum(axis=1)
    numerator = np.where(valid, returns * weights, 0.0).sum(axis=1)
    denominator = np.where(valid, weights, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        weighted = numerator / denominator
    out: Vector = np.where(counted >= minimum_assets, weighted, np.nan)
    return out


def _momentum_characteristic(prices: Matrix, window: int) -> Matrix:
    """The momentum sort is formed on prices shifted by one, never on today's.

    Its own function so the lag can be tested on its own. Dropping the shift
    leaves every other property of the model intact and quietly grades the sort
    on the very return it is meant to predict -- the subtlest look-ahead this
    module has, and invisible in anything but a point-in-time test.
    """
    return _percentage_change(_shifted(prices, 1), window)


def canonical_factor_returns(
    prices: Matrix,
    market_caps: Matrix,
    *,
    members: BoolMatrix | None = None,
    config: CanonicalFactorConfig | None = None,
) -> Matrix:
    """Daily market, size and momentum factor returns from prices and caps.

    `(dates, 3)`, columns in `CANONICAL_FACTORS` order. Both panels are
    `(dates, assets)` on one grid. Portfolio formation reads lagged caps and
    lagged momentum, so a date's own return never chooses its own constituents.

    `members` is the point-in-time universe mask. Unlike the model this ports,
    an asset the mask never selects is *not* dropped from the whole sample: that
    would let a whole-history decision into every date's cross-section, which is
    survivorship leaking into a factor return.
    """
    settings = config if config is not None else CanonicalFactorConfig()
    if prices.shape != market_caps.shape:
        msg = f"prices {prices.shape} do not match market_caps {market_caps.shape}"
        raise ValueError(msg)
    held: BoolMatrix = (
        np.ones(prices.shape, dtype=np.bool_) if members is None else members
    )
    if held.shape != prices.shape:
        msg = f"members {held.shape} do not match prices {prices.shape}"
        raise ValueError(msg)

    with np.errstate(invalid="ignore"):
        priced: Matrix = np.where(prices > 0.0, prices, np.nan)
    # Filled twice: a missing cap takes the previous day's, and one that is not a
    # usable weight -- zero, negative, or infinite -- is a glitch that takes it
    # too, rather than dropping the name out of every value-weighted portfolio
    # for the day. Infinity is excluded here rather than ranked highest, as the
    # model this ports does: it ranks fine but overflows the weighted sum it is
    # then used in, and a cap that cannot be used as a weight is one this does
    # not have.
    caps = _forward_fill(market_caps)
    with np.errstate(invalid="ignore"):
        caps = _forward_fill(np.where(np.isfinite(caps) & (caps > 0.0), caps, np.nan))

    returns = _percentage_change(priced)
    lagged_caps: Matrix = np.where(held, _shifted(caps, 1), np.nan)

    market = _value_weighted_return(returns, lagged_caps, held, settings.minimum_assets)

    size_rank = row_rank_pct(lagged_caps)
    size = _value_weighted_return(
        returns, lagged_caps, held & (size_rank <= _TAIL_FRACTION), 1
    ) - _value_weighted_return(
        returns, lagged_caps, held & (size_rank > 1.0 - _TAIL_FRACTION), 1
    )

    characteristic = _momentum_characteristic(priced, settings.momentum_window)
    spreads = []
    for half in (size_rank <= _HALF, size_rank > _HALF):
        eligible = held & half & ~np.isnan(characteristic)
        rank = row_rank_pct(np.where(eligible, characteristic, np.nan))
        spreads.append(
            _value_weighted_return(
                returns, lagged_caps, eligible & (rank > 1.0 - _TAIL_FRACTION), 1
            )
            - _value_weighted_return(
                returns, lagged_caps, eligible & (rank <= _TAIL_FRACTION), 1
            )
        )
    # A size half without a spread leaves the date without a momentum factor,
    # rather than handing back the other half's.
    momentum = (spreads[0] + spreads[1]) / 2.0

    stacked: Matrix = np.column_stack([market, size, momentum])
    return stacked


def _trailing_totals(values: Matrix | Cube, window: int) -> Matrix | Cube:
    """Totals over the `window` rows ending at each row, that row included.

    Accumulated inside blocks of `window` rows rather than differenced from one
    running cumulative sum. Crypto returns carry outliers many orders of
    magnitude above a window's own scale, and a running total never forgets one:
    every later window becomes the difference of two totals dominated by a
    single early observation.
    """
    rows = values.shape[0]
    blocks = -(-(rows + window) // window)
    padded = np.zeros((blocks * window, *values.shape[1:]), dtype=np.float64)
    padded[window : window + rows] = values
    shaped = padded.reshape(blocks, window, *values.shape[1:])

    prefix = np.cumsum(shaped, axis=1)
    suffix = np.cumsum(shaped[:, ::-1], axis=1)[:, ::-1]

    first = np.arange(rows) + 1
    last = first + window - 1
    spills = (last // window) > (first // window)
    carried = prefix[last // window, last % window]
    totals: Matrix | Cube = suffix[first // window, first % window] + np.where(
        spills.reshape(-1, *([1] * (values.ndim - 1))), carried, 0.0
    )
    return totals


def _before(values: Matrix | Cube, window: int) -> Matrix | Cube:
    """`_trailing_totals` over the rows strictly *earlier* than each row.

    The one line that makes an exposure point in time. Zero-filled rather than
    blanked: these are contributions to a total, so the row that falls off the
    top contributes nothing rather than voiding every window it appears in.
    """
    lagged = np.zeros_like(values)
    lagged[1:] = values[:-1]
    return _trailing_totals(lagged, window)


def rolling_factor_betas(
    asset_returns: Matrix,
    factor_returns: Matrix,
    *,
    config: CanonicalFactorConfig | None = None,
) -> Cube:
    """Rolling OLS loadings, one regression per date and asset.

    `(factors, dates, assets)` -- the layout `project_out_rows` wants its
    controls in, so nothing downstream transposes. The regression at each date
    reads observations *strictly before* it, over at most `beta_window` of them,
    and needs `beta_min_periods` where the asset and every factor are present.

    Solved through the normal equations over the whole panel at once; a
    rank-deficient window falls back to the pseudo-inverse, which is the
    minimum-norm answer a per-date least-squares solve would have given.
    """
    settings = config if config is not None else CanonicalFactorConfig()
    if asset_returns.shape[0] != factor_returns.shape[0]:
        msg = (
            f"asset_returns has {asset_returns.shape[0]} dates, "
            f"factor_returns has {factor_returns.shape[0]}"
        )
        raise ValueError(msg)
    dates, assets = asset_returns.shape
    factors = factor_returns.shape[1]
    window, min_periods = settings.beta_window, settings.beta_min_periods

    design: Matrix = np.column_stack([np.ones(dates), factor_returns])
    usable = np.isfinite(design).all(axis=1)

    out: Cube = np.full((factors, dates, assets), np.nan)
    if dates == 0 or assets == 0:
        return out
    reached = np.arange(dates) >= min_periods

    # A column block at a time. The cross-product cube is (dates, assets, terms,
    # terms) and `_trailing_totals` allocates several more of that size, so
    # solving the whole panel at once costs gigabytes on a real one -- in a
    # container whose worker count is already capped to police memory.
    for first in range(0, assets, _ASSET_BLOCK):
        block = asset_returns[:, first : first + _ASSET_BLOCK]
        answered = _block_betas(
            block, design, usable, reached, window=window, min_periods=min_periods
        )
        out[:, :, first : first + _ASSET_BLOCK] = np.moveaxis(answered[:, :, 1:], 2, 0)
    return np.where(np.isfinite(out), out, np.nan)


def _block_betas(  # noqa: PLR0913 - one per input the block solve reads
    block: Matrix,
    design: Matrix,
    usable: BoolVector,
    reached: BoolVector,
    *,
    window: int,
    min_periods: int,
) -> Cube:
    """One block's loadings, intercept included: `(dates, assets, terms)`."""
    dates, assets = block.shape
    terms = design.shape[1]
    answered: Cube = np.full((dates, assets, terms), np.nan)

    valid = np.isfinite(block) & usable[:, None]
    target = np.where(valid, block, 0.0)
    rows = np.where(valid[:, :, None], design[:, None, :], 0.0)

    systems = _before(rows[:, :, :, None] * rows[:, :, None, :], window)
    targets = _before(rows * target[:, :, None], window)
    observations = _before(valid.astype(np.float64), window)

    solvable = (observations >= min_periods) & reached[:, None]
    if not solvable.any():
        return answered
    systems, targets = systems[solvable], targets[solvable]

    # Unit diagonal before solving: the intercept's column and the factor
    # returns' differ by orders of magnitude, and the singularity test below
    # only means something on a scaled system.
    scale = np.sqrt(np.diagonal(systems, axis1=1, axis2=2))
    scale = np.where(scale > 0.0, scale, 1.0)
    scaled = systems / (scale[:, :, None] * scale[:, None, :])
    with np.errstate(invalid="ignore", over="ignore"):
        singular = np.abs(np.linalg.det(scaled)) <= _SINGULAR_DETERMINANT

    scaled_targets = (targets / scale)[:, :, None]
    solved: Matrix = np.full((len(systems), terms), np.nan)
    if (~singular).any():
        solved[~singular] = np.linalg.solve(
            scaled[~singular], scaled_targets[~singular]
        )[:, :, 0]
    if singular.any():
        solved[singular] = (
            np.linalg.pinv(scaled[singular]) @ scaled_targets[singular]
        )[:, :, 0]

    answered[solvable] = solved / scale
    return answered


_SINGULAR_DETERMINANT = 1e-10
_ASSET_BLOCK = 128
"""Columns solved at once, bounding the cross-product cube's size."""


def _usable(values: Matrix, betas: Cube) -> BoolMatrix:
    """Cells where the weight and every one of its loadings are present."""
    mask: BoolMatrix = np.isfinite(values) & np.isfinite(betas).all(axis=0)
    return mask


def factor_exposure(values: Matrix, betas: Cube) -> Matrix:
    """Each row's exposure to each factor: `(dates, factors)`.

    A cell missing either its weight or a loading contributes nothing, rather
    than voiding the row -- the same rule the rest of this library keeps.
    """
    _checked_betas(values, betas)
    usable = _usable(values, betas)
    held = np.where(usable, values, 0.0)
    loadings = np.where(usable[None, :, :], betas, 0.0)
    with np.errstate(invalid="ignore", over="ignore"):
        exposures: Matrix = np.einsum("da,kda->dk", held, loadings)
    return exposures


def _checked_betas(values: Matrix, betas: Cube) -> None:
    if betas.ndim != _BETA_CUBE_RANK:
        msg = f"betas must be (factors, dates, assets), got {betas.shape}"
        raise ValueError(msg)
    if betas.shape[1:] != values.shape:
        msg = f"betas {betas.shape[1:]} do not match values {values.shape}"
        raise ValueError(msg)


def _checked_limits(limits: Vector, factors: int) -> Vector:
    if limits.shape != (factors,):
        msg = f"limits must hold one entry per factor, got {limits.shape}"
        raise ValueError(msg)
    if np.any(~(limits > 0.0)):
        msg = f"every limit must be positive, got {limits}"
        raise ValueError(msg)
    return limits


def factor_cap_blend(values: Matrix, betas: Cube, limits: Vector) -> Matrix:
    """Per-row fraction in [0, 1] to blend toward the factor-neutral projection.

    Zero where a row is already inside every limit, and zero where the row is
    too thin to regress -- `factor_capped` shrinks such a row uniformly instead.

    The fraction is solved against the exposure *after* the gross is restored,
    which is not the same as solving against the exposure before it. Writing
    `rho` for the tightest ratio of limit to exposure, a row is compliant
    exactly when `F(t) = (1 - t) * gross - rho * gross(blend(t))` is not
    positive. `F` is a linear function minus a positive multiple of a convex
    one, so it is concave, and it runs from positive at `t = 0` to negative at
    `t = 1`: it crosses zero once, and bisection finds that crossing. Solving
    the pre-rescale exposure instead looks right and breaches, because the
    projection shrinks the gross that is then restored.
    """
    _checked_betas(values, betas)
    _checked_limits(limits, betas.shape[0])
    usable = _usable(values, betas)
    neutral = project_out_rows(values, betas, usable)

    out: Matrix = np.zeros((values.shape[0], 1), dtype=np.float64)
    for row in range(values.shape[0]):
        columns = np.flatnonzero(usable[row])
        if columns.size <= betas.shape[0] + 1:
            # Fewer observations than regressors leaves a residual that is
            # noise, not a hedge.
            continue
        weights = values[row, columns]
        residual = neutral[row, columns]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            held = float(np.abs(weights).sum())
            hedged = float(np.abs(residual).sum())
            exposures = betas[:, row, columns] @ weights
            worst = float(np.max(np.abs(exposures) / limits))
        if not np.isfinite(residual).all() or not np.isfinite(held):
            continue
        if held <= 0.0 or not np.isfinite(hedged):
            continue
        if hedged * _MAXIMUM_AMPLIFICATION < held:
            # Too little of this row survives the projection to hedge with. A
            # row that is nearly constant is the ordinary case: the intercept
            # takes all of it, so there is no direction to hedge along and only
            # a uniform shrink is left.
            continue
        if not np.isfinite(worst) or worst <= 1.0:
            continue
        out[row, 0] = _bisect(weights, residual, held, 1.0 / worst)
    return out


def _bisect(weights: Vector, residual: Vector, held: float, rho: float) -> float:
    """The smallest blend fraction whose rescaled row is inside every limit."""
    low, high = 0.0, 1.0
    with np.errstate(invalid="ignore", over="ignore"):
        for _ in range(_BISECTION_PASSES):
            middle = (low + high) / 2.0
            blended = (1.0 - middle) * weights + middle * residual
            slack = (1.0 - middle) * held - rho * float(np.abs(blended).sum())
            low, high = (middle, high) if slack > 0.0 else (low, middle)
    return high


def factor_capped(values: Matrix, betas: Cube, limits: Vector) -> Matrix:
    """Each row hedged toward factor-neutral only as far as its limits require.

    A row that *can* be hedged keeps the gross it arrived with, so the book is
    reshaped rather than resized. A row that cannot -- one the projection leaves
    almost nothing of, which a nearly constant row always is -- is shrunk
    uniformly instead. Both reach the limit; only the first preserves gross, and
    `factor_cap_blend` says which happened by returning zero for the second.

    A cell whose loadings are missing is left exactly as it was, by both paths:
    it is not measurable against a limit, it adds nothing to the exposure being
    capped, and to a live book dropping or resizing it is a trade nobody asked
    for.
    """
    _checked_betas(values, betas)
    _checked_limits(limits, betas.shape[0])
    usable = _usable(values, betas)
    blend = factor_cap_blend(values, betas, limits)
    neutral = project_out_rows(values, betas, usable)

    held = np.where(usable, values, 0.0)
    hedge = np.where(usable, np.nan_to_num(neutral, nan=0.0), 0.0)
    blended = (1.0 - blend) * held + blend * hedge

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        before = np.abs(held).sum(axis=1, keepdims=True)
        after = np.abs(blended).sum(axis=1, keepdims=True)
        ratio = np.where(after > 0.0, before / after, 1.0)
        # A row whose own gross overflows has no ratio worth trusting; leaving it
        # alone is honest, where scaling it by inf/inf would blank the book.
        restored = blended * np.where(np.isfinite(ratio), ratio, 1.0)
    out: Matrix = np.where(usable, restored, values)

    exposures = factor_exposure(out, betas)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        worst = np.max(np.abs(exposures) / limits, axis=1, keepdims=True)
        # An exposure that is not finite is not a measurement, so it is not a
        # breach either -- shrinking against it would flatten the book on the
        # strength of an overflow.
        breaching = np.isfinite(worst) & (worst > 1.0)
        shrink = np.where(breaching, 1.0 / np.where(breaching, worst, 1.0), 1.0)
    # Only the measured cells. A cell with no loadings adds nothing to the
    # exposure being shrunk against, so shrinking it would be a trade taken for
    # no risk reason -- and it is the same cell this function has just promised
    # to leave alone.
    scaled: Matrix = np.where(usable, out * shrink, out)
    return scaled


def drawdown_scale(book_returns: Vector, config: DrawdownScaleConfig) -> Vector:
    """A per-date multiplier that cuts the book as its drawdown approaches the limit.

    `max_leverage` at the high-water mark, zero once the trailing drawdown
    reaches `drawdown_limit`, and linear between.

    A missing return is read as a flat day rather than voiding the series, which
    is what lets a book with a gap still be sized. A book whose equity reaches
    zero or goes through it is cut to nothing and stays there: past that point
    the ratio of equity to peak is not a drawdown, and a book that has lost more
    than everything is not one to size back up.
    """
    if book_returns.size == 0:
        empty: Vector = np.zeros(0, dtype=np.float64)
        return empty
    compounded = np.cumprod(1.0 + np.nan_to_num(book_returns, nan=0.0))
    window = min(config.window, compounded.size)
    # `-inf` padding, so the first rows take the maximum over what exists rather
    # than needing a full window -- the expanding peak a book has before it has
    # `window` of history.
    padded = np.concatenate(
        [np.full(window - 1, -np.inf, dtype=np.float64), compounded]
    )
    peak = np.maximum.accumulate(
        np.lib.stride_tricks.sliding_window_view(padded, window), axis=-1
    )[:, -1]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        # `peak` includes the current row, so `fall` is never positive and only
        # the floor can bind.
        fall = compounded / peak - 1.0
        scale = (np.maximum(fall / config.drawdown_limit, -1.0) + 1.0) * (
            config.max_leverage
        )
    # Once equity has been wiped out the arithmetic above stops meaning
    # anything -- two negatives divide to a positive, and the book reads as
    # recovered. It is not recovered; it is gone.
    wiped = np.minimum.accumulate(compounded) <= 0.0
    clipped: Vector = np.where(
        wiped, 0.0, np.clip(np.nan_to_num(scale, nan=0.0), 0.0, config.max_leverage)
    )
    return clipped
