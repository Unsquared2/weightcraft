"""The container every `weightcraft` function speaks: dates x assets, missing is NaN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from weightcraft.arrays import Dates, Matrix, Vector

DATE_COLUMN = "date"
"""The leading column name every frame carries."""

_EXPECTED_RANK = 2


def _frozen_values(values: Matrix) -> Matrix:
    frozen: Matrix = np.array(values, dtype=np.float64, copy=True)
    frozen.flags.writeable = False
    return frozen


def _frozen_dates(dates: Dates) -> Dates:
    frozen: Dates = np.array(dates, dtype="datetime64[ns]", copy=True)
    frozen.flags.writeable = False
    return frozen


@dataclass(frozen=True, slots=True, eq=False)
class WeightFrame:
    """A rectangular block of weights, indexed by date and asset.

    The arrays are copied and marked read-only on construction, so a frame
    cannot be changed after it is built, or by whoever handed the data over.
    """

    dates: Dates
    assets: tuple[str, ...]
    values: Matrix

    def __post_init__(self) -> None:
        dates = _frozen_dates(self.dates)
        values = _frozen_values(self.values)
        assets = tuple(self.assets)
        if dates.ndim != 1:
            msg = f"dates must be one-dimensional, got shape {dates.shape}"
            raise ValueError(msg)
        if values.ndim != _EXPECTED_RANK:
            msg = f"values must be two-dimensional, got shape {values.shape}"
            raise ValueError(msg)
        if values.shape != (dates.size, len(assets)):
            msg = (
                f"values shape {values.shape} does not match "
                f"{dates.size} dates x {len(assets)} assets"
            )
            raise ValueError(msg)
        if len(set(assets)) != len(assets):
            msg = "assets must be unique"
            raise ValueError(msg)
        # `align` places rows by `searchsorted` against the union of every
        # frame's dates, so a repeated date is not a duplicate row -- it is one
        # row silently overwriting another.
        if np.unique(dates).size != dates.size:
            msg = "dates must be unique"
            raise ValueError(msg)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "assets", assets)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare labels and values, counting NaN in the same cell as equal."""
        if not isinstance(other, WeightFrame):
            return NotImplemented
        return (
            self.assets == other.assets
            and bool(np.array_equal(self.dates, other.dates))
            and bool(np.array_equal(self.values, other.values, equal_nan=True))
        )

    @override
    def __hash__(self) -> int:
        """Hash the labels only, which equal frames always share."""
        return hash((self.assets, self.dates.tobytes()))

    @override
    def __repr__(self) -> str:
        return f"WeightFrame({self.dates.size} dates x {len(self.assets)} assets)"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dates.size, len(self.assets))

    @property
    def is_empty(self) -> bool:
        """No cells at all -- either no dates, or no assets."""
        return self.values.size == 0

    @property
    def has_dates(self) -> bool:
        """Whether the frame covers any dates, regardless of how many assets."""
        return bool(self.dates.size)

    def with_values(self, values: Matrix) -> Self:
        """Return a new frame carrying the same labels and different values."""
        return type(self)(dates=self.dates, assets=self.assets, values=values)

    def last_row(self) -> Vector:
        if self.dates.size == 0:
            msg = "an empty frame has no last row"
            raise ValueError(msg)
        row: Vector = np.asarray(self.values[-1], dtype=np.float64)
        return row

    @classmethod
    def from_rows(
        cls,
        dates: Sequence[str] | Dates,
        assets: Sequence[str],
        values: Matrix,
    ) -> Self:
        return cls(
            dates=np.asarray(dates, dtype="datetime64[ns]"),
            assets=tuple(assets),
            values=values,
        )

    @classmethod
    def from_polars(cls, frame: pl.DataFrame) -> Self:
        """Read a frame with a leading `date` column and one column per asset."""
        if DATE_COLUMN not in frame.columns:
            msg = f"frame is missing its {DATE_COLUMN!r} column"
            raise ValueError(msg)
        assets = tuple(c for c in frame.columns if c != DATE_COLUMN)
        dates: Dates = frame.get_column(DATE_COLUMN).cast(pl.Datetime("ns")).to_numpy()
        values: Matrix = (
            frame.select(assets)
            .with_columns(pl.all().cast(pl.Float64))
            .to_numpy()
            .reshape(frame.height, len(assets))
        )
        return cls(dates=dates, assets=assets, values=values)

    def to_polars(self) -> pl.DataFrame:
        """Render as a polars frame with a leading `date` column, NaN as null."""
        columns: dict[str, pl.Series] = {
            DATE_COLUMN: pl.Series(DATE_COLUMN, self.dates, dtype=pl.Datetime("ns")),
        }
        for index, asset in enumerate(self.assets):
            columns[asset] = pl.Series(
                asset, self.values[:, index], dtype=pl.Float64
            ).fill_nan(None)
        return pl.DataFrame(columns)
