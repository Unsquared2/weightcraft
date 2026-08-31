"""Hold a position through a move too small to be worth trading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from weightcraft.normalize import usable

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix, Vector


def no_trade_band(target: Matrix, band: float) -> Matrix:
    """Freeze a held position until it moves more than `band` times the row's own scale.

    Entries and exits are never banded -- only the size of a name already held
    into a row, or still held from the row before, can be frozen there.
    """
    if band <= 0.0:
        msg = f"band must be positive, got {band}"
        raise ValueError(msg)
    blanked = usable(target)
    held: Matrix = np.full_like(target, np.nan)
    current: Vector = np.full(target.shape[1], np.nan)
    for row in range(target.shape[0]):
        wanted: Vector = blanked[row]
        finite = np.isfinite(wanted)
        with np.errstate(invalid="ignore"):
            scale = np.nanmean(np.abs(wanted)) if finite.any() else np.nan
        if not np.isfinite(scale) or scale <= 0.0:
            current = wanted.copy()
        else:
            moved = ~np.isfinite(current) | ~finite
            moved |= np.abs(wanted - np.where(np.isfinite(current), current, 0.0)) > (
                band * scale
            )
            current = np.where(moved, wanted, current)
        held[row] = current
    return held
