"""Price-driven changes in portfolio weights between target events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix


def drifted_weights(
    targets: Matrix, reference_prices: Matrix, current_prices: Matrix
) -> Matrix:
    """Price-drifted rows; unvalued/insolvent rows are NaN, zeros need no prices."""
    if targets.shape != reference_prices.shape or targets.shape != current_prices.shape:
        msg = "targets and prices must have the same shape"
        raise ValueError(msg)
    priced = (
        np.isfinite(reference_prices)
        & (reference_prices > 0.0)
        & np.isfinite(current_prices)
        & (current_prices > 0.0)
    )
    valid = np.all(np.isfinite(targets) & ((targets == 0.0) | priced), axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative = np.where(targets == 0.0, 1.0, current_prices / reference_prices)
        exposure = targets * relative
        # Future funding cash flows affect NAV separately from price-valued exposure.
        nav = 1.0 + np.sum(targets * (relative - 1.0), axis=1, keepdims=True)
        valid_nav = np.isfinite(nav) & (nav > 0.0)
        result: Matrix = np.where(valid[:, None] & valid_nav, exposure / nav, np.nan)
    return result
