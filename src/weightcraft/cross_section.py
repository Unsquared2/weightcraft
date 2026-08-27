"""Row-wise transforms: standardise, rank, select, neutralise."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from weightcraft.arrays import BoolMatrix, Cube, Matrix, Vector

_ORTHOGONALISATION_PASSES = 2
_RANK_TOLERANCE = 1e-12


def row_counts(values: Matrix) -> Vector:
    """How many cells each row actually observed."""
    counts: Vector = np.isfinite(values).sum(axis=1).astype(np.float64)
    return counts


def standardize_rows(values: Matrix) -> Matrix:
    """Row z-score with `ddof=0`; a constant row scores zero rather than NaN."""
    present = np.isfinite(values)
    filled = np.where(present, values, 0.0)
    counts = present.sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mean = np.where(counts > 0.0, filled.sum(axis=1, keepdims=True) / counts, 0.0)
        centred = np.where(present, values - mean, np.nan)
        variance = np.where(
            counts > 0.0,
            np.nansum(np.where(present, centred, 0.0) ** 2, axis=1, keepdims=True)
            / counts,
            0.0,
        )
        deviation = np.sqrt(variance)
        scaled: Matrix = np.where(deviation > 0.0, centred / deviation, centred * 0.0)
    return np.where(present, scaled, np.nan)


def row_rank_pct(values: Matrix) -> Matrix:
    """Per-row percentile rank in [0, 1], ties broken by first appearance."""
    out: Matrix = np.full(values.shape, np.nan)
    for row in range(values.shape[0]):
        present = np.flatnonzero(np.isfinite(values[row]))
        if present.size == 0:
            continue
        order = present[np.argsort(values[row, present], kind="stable")]
        out[row, order] = (np.arange(order.size, dtype=np.float64) + 1.0) / order.size
    return out


def top_n_mask(values: Matrix, top_n: int) -> BoolMatrix:
    """The `top_n` largest cells per row, ties broken toward the earlier column."""
    if top_n < 1:
        msg = f"top_n must be at least 1, got {top_n}"
        raise ValueError(msg)
    mask: BoolMatrix = np.zeros(values.shape, dtype=np.bool_)
    for row in range(values.shape[0]):
        present = np.flatnonzero(np.isfinite(values[row]))
        if present.size == 0:
            continue
        order = present[np.argsort(-values[row, present], kind="stable")]
        mask[row, order[:top_n]] = True
    return mask


def _orthonormal_basis(vectors: list[Vector]) -> list[Vector]:
    """Modified Gram-Schmidt, run twice, dropping directions that do not survive.

    One pass leaves enough of a control in the residual to matter once the
    controls are nearly collinear.
    """
    basis: list[Vector] = []
    with np.errstate(over="ignore", invalid="ignore"):
        for candidate in vectors:
            direction = candidate
            for _ in range(_ORTHOGONALISATION_PASSES):
                for existing in basis:
                    direction = direction - float(existing @ direction) * existing
            norm = float(np.linalg.norm(direction))
            if norm > _RANK_TOLERANCE and np.isfinite(direction).all():
                basis.append(direction / norm)
    return basis


def _residual_against(vector: Vector, basis: list[Vector]) -> Vector:
    residual = vector
    with np.errstate(over="ignore", invalid="ignore"):
        for existing in basis:
            residual = residual - float(existing @ residual) * existing
    return residual


def project_out_rows(
    target: Matrix, controls: Cube, eligible: BoolMatrix | None = None
) -> Matrix:
    """Per-row OLS residual of `target` on an intercept plus `controls`.

    `controls` is (controls, dates, assets). A row is answered only where the
    target and every control are present, so a gap costs one asset rather than
    the whole date.
    """
    if controls.ndim != _ORTHOGONALISATION_PASSES + 1:
        msg = f"controls must be (controls, dates, assets), got {controls.shape}"
        raise ValueError(msg)
    if controls.shape[1:] != target.shape:
        msg = f"controls {controls.shape[1:]} do not match target {target.shape}"
        raise ValueError(msg)

    usable = np.isfinite(target)
    if eligible is not None:
        usable &= eligible
    usable &= np.isfinite(controls).all(axis=0)

    out: Matrix = np.full(target.shape, np.nan)
    for row in range(target.shape[0]):
        columns = np.flatnonzero(usable[row])
        if columns.size == 0:
            continue
        regressors: list[Vector] = [np.ones(columns.size, dtype=np.float64)]
        regressors.extend(
            controls[control, row, columns].astype(np.float64)
            for control in range(controls.shape[0])
        )
        out[row, columns] = _residual_against(
            target[row, columns].astype(np.float64), _orthonormal_basis(regressors)
        )
    return out


def residualize_rows(
    signal: Matrix,
    controls: Cube,
    *,
    required_assets: int = 2,
    standardize: bool = True,
) -> Matrix:
    """`project_out_rows`, blanking rows too thin to regress and rescaling after."""
    residual = project_out_rows(signal, controls)
    thin = row_counts(residual) < required_assets
    residual[thin] = np.nan
    return standardize_rows(residual) if standardize else residual
