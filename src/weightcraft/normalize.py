"""Turn a raw book into a sized one: gross, net, caps, lot sizes."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from weightcraft.arrays import BoolMatrix, Matrix


# A row no source covered is the ordinary case, and numpy warns on every
# reduction over one. The NaN it returns is the answer we want.
_ALL_NAN_WARNING = "All-NaN slice encountered"
_EMPTY_MEAN_WARNING = "Mean of empty slice"


def usable(values: Matrix) -> Matrix:
    """The values with every non-finite cell blanked.

    An infinity is missing, not large. `gross` used to sum it, which made one
    broken cell divide every good position on that date down to zero.
    """
    blanked: Matrix = np.where(np.isfinite(values), values, np.nan)
    return blanked


def row_sums(values: Matrix) -> Matrix:
    """Per-row sum of the present cells, kept as a column for broadcasting."""
    total: Matrix = np.nansum(values, axis=1, keepdims=True)
    return total


def held(values: Matrix) -> BoolMatrix:
    """Which cells carry a position at all."""
    mask: BoolMatrix = np.isfinite(values) & (values != 0.0)
    return mask


def fill_missing(preferred: Matrix, fallback: Matrix) -> Matrix:
    """`preferred` wherever it is present, `fallback` where it is not."""
    filled: Matrix = np.where(np.isnan(preferred), fallback, preferred)
    return filled


def normalised_share(measure: Matrix, holdings: BoolMatrix, *, power: float) -> Matrix:
    """`measure ** power`, renormalised per row and scaled by how many are held.

    The output sums to the held count rather than to one: a constructor
    multiplies an existing book rather than replacing its gross.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        raw = np.power(measure, power) if power != 1.0 else measure
        raw = np.where(np.isfinite(raw), raw, np.nan)
        share = raw / np.nansum(raw, axis=1, keepdims=True)
    share = np.nan_to_num(share, nan=0.0, posinf=0.0, neginf=0.0)
    scaled: Matrix = share * holdings.sum(axis=1, keepdims=True)
    return scaled


def rescaled_to_held_count(side: Matrix) -> Matrix:
    """A per-side multiplier, renormalised to sum to how many names it holds."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rescaled: Matrix = (
            side / row_sums(side) * (~np.isnan(side)).sum(axis=1, keepdims=True)
        )
    return rescaled


def gross(values: Matrix) -> Matrix:
    """Per-row gross exposure: the sum of the absolute weights, skipping missing."""
    with np.errstate(over="ignore"):
        total: Matrix = np.nansum(np.abs(usable(values)), axis=1, keepdims=True)
    return total


def to_gross(values: Matrix, target: float) -> Matrix:
    """Rescale each row to a fixed gross exposure, leaving an empty row alone.

    Without this an ensemble's size swings with how much its members happen to
    agree, which turns disagreement into an accidental exposure bet.
    """
    if target <= 0.0:
        msg = f"gross target must be positive, got {target}"
        raise ValueError(msg)
    current = gross(values)
    scalable = np.isfinite(current) & (current > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled: Matrix = np.where(scalable, values * target / current, values)
    return scaled


def net(values: Matrix) -> Matrix:
    """Per-row net exposure: the signed sum of the weights, skipping missing."""
    with np.errstate(over="ignore"):
        total: Matrix = np.nansum(usable(values), axis=1, keepdims=True)
    return total


def exposure_scale(
    values: Matrix, *, max_gross: float | None = None, max_net: float | None = None
) -> Matrix:
    """Per-row multiplier in (0, 1] that brings a row inside both exposure limits.

    A limit of exactly zero is refused like every other positive-only limit
    here -- `center` is the tool for a book that must be exactly net zero.
    """
    if max_gross is not None and max_gross <= 0.0:
        msg = f"max_gross must be positive, got {max_gross}"
        raise ValueError(msg)
    if max_net is not None and max_net <= 0.0:
        msg = f"max_net must be positive, got {max_net}"
        raise ValueError(msg)
    scale: Matrix = np.ones((values.shape[0], 1), dtype=np.float64)
    if max_gross is not None:
        current = gross(values)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.minimum(scale, np.where(current > 0.0, max_gross / current, 1.0))
    if max_net is not None:
        current = np.abs(net(values))
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.minimum(scale, np.where(current > 0.0, max_net / current, 1.0))
    return scale


def capped(
    values: Matrix, *, max_gross: float | None = None, max_net: float | None = None
) -> Matrix:
    """Each row shrunk uniformly until it satisfies both exposure limits."""
    scaled: Matrix = values * exposure_scale(
        values, max_gross=max_gross, max_net=max_net
    )
    return scaled


def center(values: Matrix) -> Matrix:
    """Subtract each row's mean, forcing zero net exposure per date."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", message=_EMPTY_MEAN_WARNING)
        mean = np.nanmean(usable(values), axis=1, keepdims=True)
    centred: Matrix = values - np.nan_to_num(mean, nan=0.0)
    return centred


def clip_allocation(values: Matrix, cap: float) -> Matrix:
    """Cap every position at `cap` in absolute terms."""
    if cap <= 0.0:
        msg = f"allocation cap must be positive, got {cap}"
        raise ValueError(msg)
    capped: Matrix = np.clip(values, -cap, cap)
    return capped


def quantize(values: Matrix, step: float) -> Matrix:
    """Round every position to a multiple of `step` -- a lot size."""
    if step <= 0.0:
        msg = f"quantisation step must be positive, got {step}"
        raise ValueError(msg)
    rounded: Matrix = np.round(values / step) * step
    return rounded


def tilt(values: Matrix, amount: float) -> Matrix:
    """Add `amount / n_held` to every held name, shifting the book's net exposure."""
    holdings = held(values)
    counts = holdings.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_name = np.where(counts > 0, amount / counts, 0.0)
    tilted: Matrix = np.where(holdings, values + per_name, values)
    return tilted


def weights_from_bins(binned: Matrix) -> Matrix:
    """Map quantile bins onto [-1, 1] and normalise each row to gross one.

    Both endpoints are used, not just the maximum: dividing by the row maximum
    alone only lands on [-1, 1] when the lowest bin happens to be exactly zero,
    and turns a 1-based or negative bin scheme into a silently net-long book.
    """
    if binned.shape[1] == 0:
        empty: Matrix = np.zeros(binned.shape, dtype=np.float64)
        return empty
    present = np.isfinite(binned)
    blanked = usable(binned)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.filterwarnings("ignore", message=_ALL_NAN_WARNING)
        row_min = np.nanmin(blanked, axis=1, keepdims=True)
        row_max = np.nanmax(blanked, axis=1, keepdims=True)
        spread = row_max - row_min
        # A row where every name lands in the same bin expresses no view, so it
        # holds nothing rather than holding everything long.
        weights = np.where(spread > 0.0, (blanked - row_min) / spread * 2.0 - 1.0, 0.0)
        total = np.nansum(np.abs(weights), axis=1, keepdims=True)
        scaled: Matrix = np.where(total > 0.0, weights / total, 0.0)
    return np.where(present, scaled, np.nan)
