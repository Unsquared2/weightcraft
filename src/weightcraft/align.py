"""Line several frames up on a shared grid, so combining them is one array op."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import numpy as np

from weightcraft.frame import WeightFrame

if TYPE_CHECKING:
    from collections.abc import Sequence

    from weightcraft.arrays import Cube, Dates, Duration, Matrix


@dataclass(frozen=True, slots=True, eq=False)
class AlignedStack:
    """`values` is (frames, dates, assets) over the union of every frame's labels.

    A cell no frame covered, and a cell a frame covered but left blank, are both
    NaN — the distinction never survives an average, so it is not recorded.
    """

    dates: Dates
    assets: tuple[str, ...]
    values: Cube

    @override
    def __repr__(self) -> str:
        frames, dates, assets = self.values.shape
        return f"AlignedStack({frames} frames x {dates} dates x {assets} assets)"

    @property
    def count(self) -> int:
        return int(self.values.shape[0])

    def with_values(self, values: Matrix) -> WeightFrame:
        """Rewrap a (dates, assets) result under this stack's labels."""
        return WeightFrame(dates=self.dates, assets=self.assets, values=values)


def align(frames: Sequence[WeightFrame]) -> AlignedStack:
    """Stack frames on the union of their dates and assets, absent cells NaN.

    The numpy equivalent of `pl.concat(how="diagonal")`: order-independent, and
    an asset only one frame carries is kept rather than dropped.
    """
    if not frames:
        msg = "align needs at least one frame"
        raise ValueError(msg)

    dates: Dates = np.unique(np.concatenate([f.dates for f in frames]))
    assets = tuple(sorted({asset for frame in frames for asset in frame.assets}))
    stacked: Cube = np.full((len(frames), dates.size, len(assets)), np.nan)

    asset_position = {asset: index for index, asset in enumerate(assets)}
    for depth, frame in enumerate(frames):
        if frame.is_empty:
            continue
        rows = np.searchsorted(dates, frame.dates)
        columns = np.array(
            [asset_position[asset] for asset in frame.assets], dtype=np.intp
        )
        stacked[depth, rows[:, None], columns[None, :]] = frame.values

    stacked.flags.writeable = False
    return AlignedStack(dates=dates, assets=assets, values=stacked)


def carried(
    frame: WeightFrame, grid: Dates, *, max_age: Duration | None = None
) -> WeightFrame:
    """`frame`'s newest row at or before each grid row, missing where there is none.

    A position is unchanged between rebalances, so repeating the last row across
    the grid rows it covers is exact rather than an approximation. Rows before
    the frame's first, and rows further than `max_age` past the one that would
    carry, are NaN: a book not yet stated is absent, never flat.
    """
    columns = len(frame.assets)
    held: Matrix = np.full((grid.size, columns), np.nan)
    if frame.has_dates and grid.size:
        order = np.argsort(frame.dates)
        dates = frame.dates[order]
        values = frame.values[order]
        previous = np.searchsorted(dates, grid, side="right") - 1
        # A hole is "this asset is not held", which a fill cannot tell from the
        # blanks the grid itself introduces -- so gather whole rows rather than
        # forward-filling cells, and a closed position stays closed.
        stated = previous >= 0
        if max_age is not None:
            stated[stated] &= (grid[stated] - dates[previous[stated]]) <= max_age
        held[stated] = values[previous[stated]]
    return WeightFrame(dates=grid, assets=frame.assets, values=held)
