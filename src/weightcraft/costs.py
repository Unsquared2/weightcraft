"""Turnover, and what it costs to trade it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix, Vector


def turnover(weights: Matrix) -> Matrix:
    """`|diff|` per position, treating a gap as a flat position rather than a hold.

    A cell that goes from missing to held has been traded into, and one that
    goes from held to missing has been traded out of; blanking both, as a plain
    `diff` does, charges nothing for either. On an aligned book -- where every
    asset a source does not cover is a gap by construction -- that made the
    transaction-cost knob a no-op.
    """
    # Non-finite is missing, as everywhere else in this library, and a missing
    # position is a flat one. Leaving an infinity in produced `inf - inf` here,
    # which is NaN -- a broken weight that costs nothing at all.
    filled = np.where(np.isfinite(weights), weights, 0.0)
    moved: Matrix = np.empty_like(filled)
    with np.errstate(over="ignore", invalid="ignore"):
        np.subtract(filled[1:], filled[:-1], out=moved[1:])
    # `[:1]`, not `[0]`, so a panel with no rows is a no-op rather than an
    # index error. The first row has no prior book to have moved from; a caller
    # who wants the cost of establishing the book prepends a flat row.
    moved[:1] = 0.0
    np.abs(moved, out=moved)
    return moved


def lagged(weights: Matrix, lag: int) -> Matrix:
    """Positions shifted forward by `1 + lag` rows; the head goes missing.

    The one is unconditional: a weight decided on a date can only earn the
    return of the date after it.
    """
    if lag < 0:
        msg = f"lag must not be negative, got {lag}"
        raise ValueError(msg)
    shifted: Matrix = np.empty_like(weights)
    offset = 1 + lag
    shifted[offset:] = weights[: max(weights.shape[0] - offset, 0)]
    shifted[:offset] = np.nan
    return shifted


def apply_costs(
    weights: Matrix, returns: Matrix, *, cost: float, lag: int = 0
) -> Matrix:
    """Per-position `w[t-1-lag] * r[t] - cost * |dw[t]|`.

    A position with no prior weight, or no return to earn, contributes nothing
    on the earning side -- but is still charged for the trade. Letting the
    missing earning void the whole cell is how a book that rotates between
    assets came to be billed for half the turnover it actually did.
    """
    if weights.shape != returns.shape:
        msg = f"weights {weights.shape} and returns {returns.shape} disagree"
        raise ValueError(msg)
    if cost < 0.0:
        msg = f"cost must not be negative, got {cost}"
        raise ValueError(msg)
    with np.errstate(invalid="ignore", over="ignore"):
        earned = lagged(weights, lag) * returns
    return np.where(np.isfinite(earned), earned, 0.0) - turnover(weights) * cost


def book_returns(
    weights: Matrix, returns: Matrix, *, cost: float = 0.0, lag: int = 0
) -> Vector:
    """The book's own return per date: the row sum of `apply_costs`."""
    total: Vector = np.nansum(apply_costs(weights, returns, cost=cost, lag=lag), axis=1)
    return total
