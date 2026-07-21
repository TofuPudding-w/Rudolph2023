"""Deterministic isometry-to-unitary completion for the Ran analytic layer."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import COMPLEX_DTYPE, as_complex128
from .linalg import unitarity_residual


def _double_mgs(
    vector: NDArray[np.complex128],
    basis: Sequence[NDArray[np.complex128]],
) -> NDArray[np.complex128]:
    residual = vector.copy()
    for _ in range(2):
        for column in basis:
            residual -= column * np.vdot(column, residual)
    return residual


def complete_unitary_from_columns(
    prescribed_columns: ArrayLike,
    column_indices: Sequence[int],
    *,
    dimension: int | None = None,
    atol: float = 1e-12,
) -> NDArray[np.complex128]:
    """Complete fixed orthonormal columns using computational-basis double MGS."""

    prescribed = as_complex128(prescribed_columns, copy=True)
    if prescribed.ndim != 2:
        raise ValueError("prescribed_columns must be a matrix.")
    inferred_dimension, count = prescribed.shape
    if dimension is None:
        dimension = inferred_dimension
    else:
        if isinstance(dimension, bool) or not isinstance(dimension, numbers.Integral):
            raise TypeError("dimension must be an integer or None.")
        dimension = int(dimension)
    if dimension < 1 or inferred_dimension != dimension:
        raise ValueError("Invalid completion dimension.")
    if not math.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be finite and positive.")
    raw_indices = tuple(column_indices)
    if any(
        isinstance(index, bool) or not isinstance(index, numbers.Integral)
        for index in raw_indices
    ):
        raise TypeError("column_indices must contain only integers.")
    indices = tuple(int(index) for index in raw_indices)
    if len(indices) != count or len(set(indices)) != count:
        raise ValueError("column_indices must uniquely match prescribed columns.")
    if any(index < 0 or index >= dimension for index in indices):
        raise ValueError("column_indices are outside the unitary dimension.")
    gram = prescribed.conj().T @ prescribed
    if not np.allclose(gram, np.eye(count), atol=atol, rtol=0.0):
        raise ValueError("Prescribed columns are not orthonormal.")

    filled = {index: prescribed[:, offset].copy() for offset, index in enumerate(indices)}
    basis = [prescribed[:, offset].copy() for offset in range(count)]
    missing = [index for index in range(dimension) if index not in filled]
    missing_offset = 0
    for anchor_index in range(dimension):
        if missing_offset == len(missing):
            break
        anchor = np.zeros(dimension, dtype=COMPLEX_DTYPE)
        anchor[anchor_index] = 1.0
        residual = _double_mgs(anchor, basis)
        norm = float(np.linalg.norm(residual))
        if norm <= atol:
            continue
        column = residual / norm
        pivot = int(np.argmax(np.abs(column)))
        column *= np.exp(-1j * np.angle(column[pivot]))
        filled[missing[missing_offset]] = column
        basis.append(column)
        missing_offset += 1
    if missing_offset != len(missing):
        raise np.linalg.LinAlgError("Could not span the unitary complement.")

    result = np.column_stack([filled[index] for index in range(dimension)])
    if unitarity_residual(result) > 32.0 * atol:
        raise ArithmeticError("Completed matrix is not unitary.")
    # Prescribed columns must remain bitwise/numerically unchanged.
    for offset, index in enumerate(indices):
        if not np.array_equal(result[:, index], prescribed[:, offset]):
            raise ArithmeticError("Completion changed a prescribed column.")
    return np.asarray(result, dtype=COMPLEX_DTYPE)


__all__ = ["complete_unitary_from_columns"]
