"""Smoothers that refuse to answer before their window has filled.

Both variants require a full window, so two books smoothed differently still
start on the same row and their turnover stays comparable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix

_MINIMUM_WINDOW = 1
_UNDERFLOW_FLOOR = 1e-100


def _validated(window: int) -> int:
    if window < _MINIMUM_WINDOW:
        msg = f"window must be at least {_MINIMUM_WINDOW}, got {window}"
        raise ValueError(msg)
    return window


def rolling_mean(values: Matrix, window: int) -> Matrix:
    """`rolling(window, min_periods=window).mean()` down each column.

    A window holding any gap is left missing, which is what makes two books
    smoothed differently start on the same row.
    """
    _validated(window)
    if values.shape[0] < window:
        empty: Matrix = np.full(values.shape, np.nan)
        return empty
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    padded_sum = np.concatenate(
        [np.zeros((1, values.shape[1])), np.cumsum(filled, axis=0)]
    )
    padded_count = np.concatenate(
        [np.zeros((1, values.shape[1])), np.cumsum(finite.astype(np.float64), axis=0)]
    )
    totals = padded_sum[window:] - padded_sum[:-window]
    counts = padded_count[window:] - padded_count[:-window]
    out: Matrix = np.full(values.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        # `counts >= window`, not `> 0`: a full window means `window`
        # observations, so a column with a ragged history stays silent rather
        # than answering off one point and presenting it as a settled average.
        out[window - 1 :] = np.where(counts >= window, totals / counts, np.nan)
    return out


def ewm_mean(values: Matrix, span: int) -> Matrix:
    """`ewm(span=span, min_periods=span).mean()`, adjusted, down each column.

    A gap decays the running weight without contributing to it, so it dilutes
    the average rather than restarting it -- and, as in pandas, does not count
    toward the `min_periods` a column must see before it answers.
    """
    _validated(span)
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    numerator = np.zeros(values.shape[1])
    denominator = np.zeros(values.shape[1])
    seen = np.zeros(values.shape[1])
    out: Matrix = np.full(values.shape, np.nan)
    for row in range(values.shape[0]):
        present = np.isfinite(values[row])
        numerator = numerator * decay + np.where(present, values[row], 0.0)
        denominator = denominator * decay + present
        # Counts observations, not rows, so `min_periods` means what pandas
        # means by it: a column that started late does not get to answer early.
        seen = seen + present
        # A gap long enough decays the running weight into the denormals, where
        # the two sides lose precision at different rates and the ratio drifts.
        # Rescaling holds the ratio exactly and keeps the weight in range.
        faded = (denominator > 0.0) & (denominator < _UNDERFLOW_FLOOR)
        if bool(faded.any()):
            numerator = np.where(faded, numerator / denominator, numerator)
            denominator = np.where(faded, 1.0, denominator)
        with np.errstate(invalid="ignore", divide="ignore"):
            average = np.where(denominator > 0.0, numerator / denominator, np.nan)
        out[row] = np.where(seen >= span, average, np.nan)
    return out


def lag_rows(values: Matrix, periods: int) -> Matrix:
    """`shift(periods)` down each column; the head goes missing."""
    if periods < 0:
        msg = f"periods must not be negative, got {periods}"
        raise ValueError(msg)
    if periods == 0:
        return values
    shifted: Matrix = np.full(values.shape, np.nan)
    shifted[periods:] = values[: max(values.shape[0] - periods, 0)]
    return shifted
