"""Audited small-matrix linear algebra used by analytic and optimized layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import COMPLEX_DTYPE, REAL_DTYPE, as_complex128, validate_real

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def unitarity_residual(matrix: ArrayLike) -> float:
    """Return ``||U^dagger U-I||_inf`` or infinity for nonsquare input."""

    array = np.asarray(matrix, dtype=COMPLEX_DTYPE)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return math.inf
    identity = np.eye(array.shape[0], dtype=COMPLEX_DTYPE)
    return float(np.linalg.norm(array.conj().T @ array - identity, ord=np.inf))


def validate_unitary(
    matrix: ArrayLike,
    *,
    dimension: int | None = None,
    atol: float = 1e-12,
) -> ComplexArray:
    """Return an owned read-only complex128 unitary after strict validation."""

    atol = validate_real(atol, name="atol", minimum=0.0)
    result = as_complex128(matrix, copy=True)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"Unitary must be square, got shape {result.shape}.")
    if dimension is not None and result.shape != (dimension, dimension):
        raise ValueError(f"Expected {dimension}x{dimension}, got {result.shape}.")
    residual = unitarity_residual(result)
    if residual > atol:
        raise ValueError(f"Matrix is not unitary; residual={residual:.3e}.")
    result.setflags(write=False)
    return result


def _phase_fix_svd_pair(
    u_matrix: ComplexArray,
    vh_matrix: ComplexArray,
    *,
    pivot_on: str,
) -> tuple[ComplexArray, ComplexArray]:
    fixed_u = np.array(u_matrix, dtype=COMPLEX_DTYPE, copy=True)
    fixed_vh = np.array(vh_matrix, dtype=COMPLEX_DTYPE, copy=True)
    for index in range(min(fixed_u.shape[1], fixed_vh.shape[0])):
        source = fixed_u[:, index] if pivot_on == "u" else fixed_vh[index, :]
        pivot = int(np.argmax(np.abs(source)))
        magnitude = abs(source[pivot])
        if magnitude == 0.0:
            continue
        phase = source[pivot] / magnitude
        if pivot_on == "u":
            fixed_u[:, index] *= np.conj(phase)
            fixed_vh[index, :] *= phase
        else:
            fixed_vh[index, :] *= np.conj(phase)
            fixed_u[:, index] *= phase
    return fixed_u, fixed_vh


def deterministic_svd(
    matrix: ArrayLike,
    *,
    full_matrices: bool = False,
    pivot_on: str = "u",
) -> tuple[ComplexArray, RealArray, ComplexArray]:
    """Compute SVD with paired-vector phase fixing.

    Exact degenerate singular subspaces remain a BLAS/LAPACK convention and
    are therefore recorded in environment provenance rather than hidden.
    """

    array = as_complex128(matrix)
    if array.ndim != 2 or min(array.shape, default=0) < 1:
        raise ValueError(f"SVD input must be a nonempty matrix, got {array.shape}.")
    if pivot_on not in {"u", "vh"}:
        raise ValueError("pivot_on must be 'u' or 'vh'.")
    u_matrix, singular_values, vh_matrix = np.linalg.svd(
        array, full_matrices=full_matrices
    )
    u_matrix, vh_matrix = _phase_fix_svd_pair(
        u_matrix, vh_matrix, pivot_on=pivot_on
    )
    return (
        np.asarray(u_matrix, dtype=COMPLEX_DTYPE),
        np.asarray(singular_values, dtype=REAL_DTYPE),
        np.asarray(vh_matrix, dtype=COMPLEX_DTYPE),
    )


def svd_numerical_rank(
    singular_values: ArrayLike,
    matrix_shape: tuple[int, int],
) -> int:
    """Return a guarded machine numerical rank, keeping at least one vector."""

    values = np.asarray(singular_values, dtype=REAL_DTYPE).reshape(-1)
    if values.size == 0 or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("singular_values must be finite, nonnegative, and nonempty.")
    if len(matrix_shape) != 2 or min(matrix_shape) < 1:
        raise ValueError("matrix_shape must contain two positive dimensions.")
    tolerance = (
        4.0
        * np.finfo(REAL_DTYPE).eps
        * max(matrix_shape)
        * float(values[0])
    )
    return max(1, int(np.count_nonzero(values > tolerance)))


def _orthogonal_complement(occupied: ComplexArray, *, dimension: int) -> ComplexArray:
    target_count = dimension - occupied.shape[1]
    if target_count == 0:
        return np.empty((dimension, 0), dtype=COMPLEX_DTYPE)
    basis: list[ComplexArray] = []
    tolerance = 128.0 * np.finfo(REAL_DTYPE).eps * dimension
    for anchor_index in range(dimension):
        vector = np.zeros(dimension, dtype=COMPLEX_DTYPE)
        vector[anchor_index] = 1.0
        for _ in range(2):
            if occupied.shape[1]:
                vector -= occupied @ (occupied.conj().T @ vector)
            for previous in basis:
                vector -= previous * np.vdot(previous, vector)
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            continue
        vector /= norm
        pivot = int(np.argmax(np.abs(vector)))
        vector *= np.exp(-1j * np.angle(vector[pivot]))
        basis.append(vector)
        if len(basis) == target_count:
            break
    if len(basis) != target_count:
        raise ArithmeticError("Could not complete the polar null space deterministically.")
    return np.column_stack(basis)


@dataclass(frozen=True, slots=True)
class PolarResult:
    """Diagnostics of ``argmax_U Re Tr(E^dagger U)``."""

    unitary: ComplexArray
    singular_values: RealArray
    nuclear_norm: float
    achieved_overlap: complex
    numerical_rank: int
    optimization_rank: int
    unitarity_residual: float


def polar_optimal_unitary(
    environment: ArrayLike,
    *,
    rank_rtol: float | None = None,
) -> PolarResult:
    """Return a deterministic unitary polar factor, including rank-deficient cases.

    ``rank_rtol`` controls the reported diagnostic rank only.  The optimizer
    must retain every machine-resolved nonzero singular direction; otherwise
    a user threshold could silently turn the result into a non-optimal polar
    factor.
    """

    matrix = as_complex128(environment, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Environment must be square, got {matrix.shape}.")
    if matrix.shape[0] != 4:
        raise ValueError("Rudolph optimization environments must be 4x4.")
    if rank_rtol is not None:
        rank_rtol = validate_real(rank_rtol, name="rank_rtol", minimum=0.0)

    magnitudes = np.abs(matrix)
    pivot = int(np.argmax(magnitudes))
    scale = float(magnitudes.flat[pivot])
    if scale == 0.0:
        phase = 1.0 + 0.0j
        canonical = matrix
    else:
        phase = complex(matrix.flat[pivot] / scale)
        canonical = matrix / (scale * phase)

    u_matrix, canonical_s, vh_matrix = np.linalg.svd(canonical, full_matrices=True)
    singular_values = np.asarray(canonical_s * scale, dtype=REAL_DTYPE)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    machine_threshold = np.finfo(REAL_DTYPE).eps * matrix.shape[0] * largest
    optimization_rank = int(np.count_nonzero(singular_values > machine_threshold))
    diagnostic_threshold = (
        machine_threshold if rank_rtol is None else rank_rtol * largest
    )
    rank = int(np.count_nonzero(singular_values > diagnostic_threshold))
    right_vectors = vh_matrix.conj().T
    left_support = np.asarray(
        u_matrix[:, :optimization_rank], dtype=COMPLEX_DTYPE
    )
    right_support = np.asarray(
        right_vectors[:, :optimization_rank], dtype=COMPLEX_DTYPE
    )
    support_factor = left_support @ right_support.conj().T
    left_null = _orthogonal_complement(left_support, dimension=matrix.shape[0])
    right_null = _orthogonal_complement(right_support, dimension=matrix.shape[0])
    unitary = np.asarray(
        phase * (support_factor + left_null @ right_null.conj().T),
        dtype=COMPLEX_DTYPE,
    )

    residual = unitarity_residual(unitary)
    nuclear_norm = float(np.sum(singular_values, dtype=REAL_DTYPE))
    achieved = complex(np.vdot(matrix, unitary))
    objective_scale = max(1.0, nuclear_norm)
    if abs(achieved.imag) > 1e-12 * objective_scale:
        raise ArithmeticError("Polar optimum has a non-negligible imaginary objective.")
    if abs(achieved.real - nuclear_norm) > 1e-12 * objective_scale:
        raise ArithmeticError("Polar objective does not equal the environment nuclear norm.")
    if residual > 1e-12:
        raise ArithmeticError(f"Polar factor lost unitarity: {residual:.3e}.")
    return PolarResult(
        unitary=unitary,
        singular_values=singular_values,
        nuclear_norm=nuclear_norm,
        achieved_overlap=achieved,
        numerical_rank=rank,
        optimization_rank=optimization_rank,
        unitarity_residual=residual,
    )


def fractional_unitary_power(
    unitary: ArrayLike,
    exponent: float,
    *,
    schur_atol: float = 1e-11,
) -> ComplexArray:
    """Return the principal-branch fractional power of a unitary.

    A complex Schur functional calculus is used.  Eigenphase ``-1`` is assigned
    principal phase ``+pi``.  The branch convention is part of provenance
    because the paper does not specify it.
    """

    exponent = validate_real(exponent, name="exponent", minimum=0.0, maximum=1.0)
    schur_atol = validate_real(schur_atol, name="schur_atol", minimum=0.0)
    matrix = validate_unitary(unitary)
    dimension = matrix.shape[0]
    if exponent == 0.0:
        return np.eye(dimension, dtype=COMPLEX_DTYPE)
    if exponent == 1.0:
        return np.array(matrix, copy=True)

    # Imported lazily so the exact analytic D_i path remains usable in a
    # NumPy-only diagnostic environment.
    from scipy.linalg import schur

    triangular, vectors = schur(matrix, output="complex")
    diagonal = np.diag(triangular)
    off_diagonal = triangular - np.diag(diagonal)
    if np.linalg.norm(off_diagonal, ord=np.inf) > schur_atol:
        raise ArithmeticError("Unitary Schur form is unexpectedly non-diagonal.")
    eigenvalues = diagonal / np.abs(diagonal)
    phases = np.angle(eigenvalues)
    # Freeze only points that are exactly on the cut.  An atol-based test would
    # incorrectly map a legitimate phase just below -pi to +pi, creating an
    # O(1) discontinuity in the fractional power.
    on_negative_real_axis = (eigenvalues.real < 0.0) & (eigenvalues.imag == 0.0)
    phases[on_negative_real_axis] = np.pi
    powered = vectors @ np.diag(np.exp(1j * exponent * phases)) @ vectors.conj().T
    residual = unitarity_residual(powered)
    if residual > 1e-11:
        raise ArithmeticError(f"Fractional power lost unitarity: {residual:.3e}.")
    return np.asarray(powered, dtype=COMPLEX_DTYPE)


@dataclass(frozen=True, slots=True)
class DampedUpdate:
    """Complete diagnostics for Rudolph equation (6)."""

    unitary: ComplexArray
    relative_unitary: ComplexArray
    fractional_step: ComplexArray
    learning_rate: float
    unitarity_residual: float


def damped_unitary_update(
    current: ArrayLike,
    optimal: ArrayLike,
    *,
    learning_rate: float = 0.6,
) -> DampedUpdate:
    """Apply ``U <- U (U^dagger U_opt)^r`` exactly as paper equation (6)."""

    rate = validate_real(
        learning_rate, name="learning_rate", minimum=0.0, maximum=1.0
    )
    current_u = validate_unitary(current, dimension=4)
    optimal_u = validate_unitary(optimal, dimension=4)
    relative = np.asarray(current_u.conj().T @ optimal_u, dtype=COMPLEX_DTYPE)
    step = fractional_unitary_power(relative, rate)
    updated = np.asarray(current_u @ step, dtype=COMPLEX_DTYPE)
    residual = unitarity_residual(updated)
    if residual > 1e-11:
        raise ArithmeticError(f"Damped update lost unitarity: {residual:.3e}.")
    return DampedUpdate(
        unitary=updated,
        relative_unitary=relative,
        fractional_step=step,
        learning_rate=rate,
        unitarity_residual=residual,
    )


__all__ = [
    "DampedUpdate",
    "PolarResult",
    "damped_unitary_update",
    "deterministic_svd",
    "fractional_unitary_power",
    "polar_optimal_unitary",
    "svd_numerical_rank",
    "unitarity_residual",
    "validate_unitary",
]
