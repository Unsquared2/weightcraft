"""Named return series, in the spirit of empyrical's test fixtures.

A test that says "the flat line scores nothing" is legible in a way that one
built from an anonymous random array is not, and a shared vocabulary means a new
metric can be swept over every shape at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix, Vector


def _vector(values: list[float]) -> Vector:
    out: Vector = np.asarray(values, dtype=np.float64)
    return out


EMPTY: Vector = np.asarray([], dtype=np.float64)
ONE_RETURN = _vector([0.01])
FLAT_LINE = _vector([0.0] * 20)
FLAT_POSITIVE = _vector([0.01] * 20)
POSITIVE_LINE = _vector([0.01, 0.02, 0.03, 0.04, 0.05] * 4)
NEGATIVE_LINE = _vector([-0.01, -0.02, -0.03, -0.04, -0.05] * 4)
MIXED = _vector([0.05, -0.02, 0.03, -0.04, 0.01] * 4)
ALL_MISSING = _vector([np.nan] * 20)

_GENERATOR = np.random.default_rng(1729)
NOISE: Vector = _GENERATOR.normal(size=200) * 0.01
LOUD_NOISE: Vector = _GENERATOR.normal(size=200) * 0.05

SPARSE_NOISE: Vector = NOISE.copy()
SPARSE_NOISE[::7] = np.nan
"""Noise with gaps sprinkled through it -- empyrical's `sparse_noise`."""

LEADING_GAP: Vector = np.concatenate([np.full(5, np.nan), NOISE[:50]])
TRAILING_GAP: Vector = np.concatenate([NOISE[:50], np.full(5, np.nan)])

WITH_INFINITY: Vector = MIXED.copy()
WITH_INFINITY[3] = np.inf

EVERY_SERIES: dict[str, Vector] = {
    "empty": EMPTY,
    "one_return": ONE_RETURN,
    "flat_line": FLAT_LINE,
    "flat_positive": FLAT_POSITIVE,
    "positive_line": POSITIVE_LINE,
    "negative_line": NEGATIVE_LINE,
    "mixed": MIXED,
    "all_missing": ALL_MISSING,
    "noise": NOISE,
    "sparse_noise": SPARSE_NOISE,
    "leading_gap": LEADING_GAP,
    "trailing_gap": TRAILING_GAP,
    "with_infinity": WITH_INFINITY,
}
"""Every shape a series can take, for sweeping a function over all of them."""

NON_EMPTY = {name: series for name, series in EVERY_SERIES.items() if series.size}


def panel(*columns: Vector) -> Matrix:
    """A (dates, assets) panel from one series per asset."""
    return np.column_stack(columns)


def returns_panel(seed: int, rows: int, scales: list[float]) -> Matrix:
    """Independent normal returns, one column per scale."""
    generator = np.random.default_rng(seed)
    out: Matrix = generator.normal(size=(rows, len(scales))) * np.asarray(scales)
    return out


def correlated_panel(seed: int, rows: int, rho: float, scale: float = 0.02) -> Matrix:
    """Two columns with correlation `rho`, plus one independent of both."""
    generator = np.random.default_rng(seed)
    common = generator.normal(size=rows)
    first = common
    second = rho * common + np.sqrt(max(1.0 - rho * rho, 0.0)) * generator.normal(
        size=rows
    )
    third = generator.normal(size=rows)
    out: Matrix = np.column_stack([first, second, third]) * scale
    return out
