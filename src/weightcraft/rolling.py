"""Rolling views and statistics that answer before the window is full.

Unlike `smoothing.rolling_mean`, which insists a window be complete before it
answers, every function here takes an explicit `min_periods` and answers as
soon as that many observations are present -- the shape a real cross-section
needs, where a name that listed mid-window, or a feed that skipped a day,
should not silently drop out of the newest rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from weightcraft.arrays import Cube, Matrix

_MINIMUM_WINDOW = 1
_MINIMUM_PERIODS = 1


def _validated(window: int, min_periods: int) -> None:
    if window < _MINIMUM_WINDOW:
        msg = f"window must be at least {_MINIMUM_WINDOW}, got {window}"
        raise ValueError(msg)
    if min_periods < _MINIMUM_PERIODS:
        msg = f"min_periods must be at least {_MINIMUM_PERIODS}, got {min_periods}"
        raise ValueError(msg)


def windowed(values: Matrix, window: int) -> Cube:
    """`(window, rows, columns)`, the trailing window ending at each row.

    Index 0 is the *oldest* observation in the window and index `window - 1`
    is the current row -- load-bearing for `bars_since_extreme`, which reads
    an age straight off the index.
    """
    if window < _MINIMUM_WINDOW:
        msg = f"window must be at least {_MINIMUM_WINDOW}, got {window}"
        raise ValueError(msg)
    rows, columns = values.shape
    padded = np.concatenate([np.full((window - 1, columns), np.nan), values])
    stack: Cube = np.stack([padded[k : k + rows] for k in range(window)])
    return stack


def partial_rolling_mean(values: Matrix, window: int, min_periods: int) -> Matrix:
    """Trailing mean of up to `window` rows, answered from `min_periods` on.

    A window near the start of the series shrinks rather than waiting for a
    full `window` to accumulate, so a name still building its history answers
    as soon as it has enough -- unlike `smoothing.rolling_mean`, which is
    silent until the row at index `window - 1`.
    """
    _validated(window, min_periods)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    zero = np.zeros((1, values.shape[1]))
    total = np.concatenate([zero, np.cumsum(filled, axis=0)])
    counts = np.concatenate([zero, np.cumsum(finite, axis=0)])
    head = np.maximum(np.arange(len(values) + 1) - window, 0)
    windowed_total = total[1:] - total[head[1:]]
    windowed_count = counts[1:] - counts[head[1:]]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean: Matrix = np.where(
            windowed_count >= min_periods,
            windowed_total / np.where(windowed_count > 0.0, windowed_count, 1.0),
            np.nan,
        )
    return mean


def bars_since_extreme(
    values: Matrix, window: int, min_periods: int, *, lowest: bool
) -> Matrix:
    """Rows since the window's lowest (`lowest=True`) or highest value.

    0 means the extreme is the current row. Answered once `min_periods`
    observations are present in the window, not only once it is full.
    """
    _validated(window, min_periods)
    stack = windowed(values, window)
    finite = np.isfinite(stack)
    filled = np.where(finite, stack, np.inf if lowest else -np.inf)
    position = filled.argmin(axis=0) if lowest else filled.argmax(axis=0)
    age = (window - 1) - position
    enough = finite.sum(axis=0) >= min_periods
    result: Matrix = np.where(enough, age, np.nan).astype(np.float64)
    return result
