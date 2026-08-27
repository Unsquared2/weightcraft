"""Reductions over an aligned stack: the arithmetic every ensemble is made of."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from weightcraft.arrays import Cube, Matrix, Vector

# An all-missing cell is the ordinary case here -- an asset only some sources
# carry -- and numpy warns about it on every reduction. The answer it gives
# (NaN) is the one we want, so the warning is noise rather than signal.
_ALL_NAN_WARNING = "All-NaN slice encountered"
_EMPTY_MEAN_WARNING = "Mean of empty slice"


def present(values: Cube) -> np.ndarray[tuple[int, int, int], np.dtype[np.bool_]]:
    """Which cells carry a usable number.

    Every reduction in this module asks the question the same way. An infinity
    is *missing*, not large: it is a broken input, and averaging it would turn
    one bad cell into a whole bad row.
    """
    mask: np.ndarray[tuple[int, int, int], np.dtype[np.bool_]] = np.isfinite(values)
    return mask


def _usable(stack: Cube) -> Cube:
    """The stack with every non-finite cell blanked, copied only if it must be."""
    mask = present(stack)
    if bool(mask.all()):
        return stack
    blanked: Cube = np.where(mask, stack, np.nan)
    return blanked


def _validated_shares(shares: Vector | Matrix, expected: tuple[int, ...]) -> None:
    if shares.shape != expected:
        msg = f"expected shares of shape {expected}, got {shares.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(shares) & (shares >= 0.0)):
        msg = "shares must be finite and non-negative"
        raise ValueError(msg)


def nanmean_stack(stack: Cube) -> Matrix:
    """Mean across frames per cell, skipping missing; all-missing stays NaN."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_EMPTY_MEAN_WARNING)
        mean: Matrix = np.nanmean(_usable(stack), axis=0)
    return mean


def nanmedian_stack(stack: Cube) -> Matrix:
    """Median across frames per cell, skipping missing; all-missing stays NaN."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_ALL_NAN_WARNING)
        median: Matrix = np.nanmedian(_usable(stack), axis=0)
    return median


def weighted_nanmean_stack(stack: Cube, shares: Vector) -> Matrix:
    """Weighted mean across frames, a missing cell dropped from both sides.

    Dropping it from the denominator too is what keeps a cell only half the
    sources carry at the level those sources set, rather than shrunk toward zero
    in proportion to who was silent.
    """
    _validated_shares(shares, (stack.shape[0],))
    usable = present(stack)
    weights = shares[:, None, None] * usable
    numerator = np.nansum(np.where(usable, stack, 0.0) * weights, axis=0)
    denominator = weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        combined: Matrix = np.where(denominator > 0.0, numerator / denominator, np.nan)
    return combined


def weighted_nanmean_stack_over_time(stack: Cube, shares: Matrix) -> Matrix:
    """`weighted_nanmean_stack` with a share per (frame, date) rather than per frame."""
    _validated_shares(shares, stack.shape[:2])
    usable = present(stack)
    weights = shares[:, :, None] * usable
    numerator = np.nansum(np.where(usable, stack, 0.0) * weights, axis=0)
    denominator = weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        combined: Matrix = np.where(denominator > 0.0, numerator / denominator, np.nan)
    return combined


def normalised_shares(raw: Vector) -> Vector:
    """Scale non-negative scores to sum to one; an all-zero set falls back to equal."""
    finite = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    total = float(finite.sum())
    if total <= 0.0:
        equal: Vector = np.full(raw.size, 1.0 / raw.size, dtype=np.float64)
        return equal
    shares: Vector = np.asarray(finite / total, dtype=np.float64)
    return shares
