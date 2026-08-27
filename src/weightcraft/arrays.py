"""Shape-carrying array aliases, so a signature says what rank it expects.

`numpy.typing.NDArray` is `Any`-shaped, which loses the one thing worth stating
about a portfolio array: whether it is a series, a panel, or a stack of panels.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

type Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
type Matrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Cube = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
type Dates = np.ndarray[tuple[int], np.dtype[np.datetime64[dt.datetime]]]
type BoolMatrix = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
type IntVector = np.ndarray[tuple[int], np.dtype[np.intp]]
