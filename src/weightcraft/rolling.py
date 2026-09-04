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


def _windowed_sums(values: Matrix, window: int) -> tuple[Matrix, Matrix, Matrix]:
    """Windowed sum, sum-of-squares and observation count, ending at each row.

    `partial_rolling_mean` and `partial_rolling_std` are both this, read
    differently -- a mean needs only the sum, a variance needs the sum of
    squares too, and computing both once is what keeps either one an
    O(rows x columns) cumulative-sum problem rather than an O(window x rows x
    columns) one.
    """
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    zero = np.zeros((1, values.shape[1]))
    total = np.concatenate([zero, np.cumsum(filled, axis=0)])
    squares = np.concatenate([zero, np.cumsum(filled**2, axis=0)])
    counts = np.concatenate([zero, np.cumsum(finite, axis=0)])
    head = np.maximum(np.arange(len(values) + 1) - window, 0)
    total_out: Matrix = total[1:] - total[head[1:]]
    squares_out: Matrix = squares[1:] - squares[head[1:]]
    counts_out: Matrix = counts[1:] - counts[head[1:]]
    return total_out, squares_out, counts_out


def partial_rolling_mean(values: Matrix, window: int, min_periods: int) -> Matrix:
    """Trailing mean of up to `window` rows, answered from `min_periods` on.

    A window near the start of the series shrinks rather than waiting for a
    full `window` to accumulate, so a name still building its history answers
    as soon as it has enough -- unlike `smoothing.rolling_mean`, which is
    silent until the row at index `window - 1`.
    """
    _validated(window, min_periods)
    total, _, counts = _windowed_sums(values, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean: Matrix = np.where(
            counts >= min_periods, total / np.where(counts > 0.0, counts, 1.0), np.nan
        )
    return mean


def partial_rolling_std(
    values: Matrix, window: int, min_periods: int, *, ddof: int = 1
) -> Matrix:
    """Trailing standard deviation of up to `window` rows, from `min_periods` on.

    `ddof=1` (the default) is `numpy.std(ddof=1)`, the sample deviation;
    `ddof=0` is the population one. Dividing needs `ddof + 1` observations on
    top of whatever `min_periods` itself demands, so the two floors combine.
    """
    if ddof < 0:
        msg = f"ddof must not be negative, got {ddof}"
        raise ValueError(msg)
    required = max(min_periods, ddof + 1)
    _validated(window, required)
    total, squares, counts = _windowed_sums(values, window)
    safe = np.where(counts > ddof, counts, ddof + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = (squares - total**2 / safe) / (safe - ddof)
    enough = counts >= required
    deviation: Matrix = np.where(
        enough & (variance > 0.0), np.sqrt(np.maximum(variance, 0.0)), np.nan
    )
    return deviation


def rolling_extreme(
    values: Matrix, window: int, min_periods: int, *, lowest: bool
) -> Matrix:
    """The trailing window's lowest (`lowest=True`) or highest value.

    Answered once `min_periods` observations are present, not only once the
    window is full -- the value `bars_since_extreme` reports the age of.
    """
    _validated(window, min_periods)
    stack = windowed(values, window)
    finite = np.isfinite(stack)
    reducer = np.nanmin if lowest else np.nanmax
    with np.errstate(invalid="ignore"):
        extreme = reducer(np.where(finite, stack, np.nan), axis=0)
    enough = finite.sum(axis=0) >= min_periods
    result: Matrix = np.where(enough, extreme, np.nan)
    return result


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
