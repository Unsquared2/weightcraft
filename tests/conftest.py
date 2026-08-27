from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from weightcraft.frame import WeightFrame

if TYPE_CHECKING:
    from weightcraft.arrays import Dates, Matrix


def dates(count: int, start: str = "2026-01-01") -> Dates:
    """`count` consecutive days from `start`, at nanosecond resolution."""
    origin = np.datetime64(start, "ns")
    day = np.timedelta64(1, "D").astype("timedelta64[ns]")
    out: Dates = origin + np.arange(count) * day
    return out


def frame(assets: tuple[str, ...], values: list[list[float]]) -> WeightFrame:
    """A frame over consecutive days, from nested lists."""
    block: Matrix = np.asarray(values, dtype=np.float64).reshape(
        len(values), len(assets)
    )
    return WeightFrame(dates=dates(len(values)), assets=assets, values=block)


def panel_frame(values: Matrix, start: str = "2026-01-01") -> WeightFrame:
    """A frame over consecutive days, with generated asset names."""
    rows, columns = values.shape
    return WeightFrame(
        dates=dates(rows, start),
        assets=tuple(f"A{index}" for index in range(columns)),
        values=values,
    )
