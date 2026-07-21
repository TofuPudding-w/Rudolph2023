"""Numerical, basis, and global-phase conventions for the reproduction."""

from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray

COMPLEX_DTYPE = np.dtype(np.complex128)
REAL_DTYPE = np.dtype(np.float64)
QUBIT_ORDER = "q0-most-significant"
TWO_QUBIT_BASIS = ("00", "01", "10", "11")
MPS_AXIS_ORDER = ("chi_left", "physical", "chi_right")
LAYER_GATE_ORDER = "left-to-right adjacent U4: (0,1),...,(N-2,N-1)"


def as_complex128(array: ArrayLike, *, copy: bool = False) -> NDArray[np.complex128]:
    """Return an array in the only supported complex dtype."""

    if copy:
        result = np.array(array, dtype=COMPLEX_DTYPE, copy=True)
    else:
        result = np.asarray(array, dtype=COMPLEX_DTYPE)
    if not np.all(np.isfinite(result)):
        raise ValueError("Array must contain only finite values.")
    return result


def validate_integer(value: int, *, name: str, minimum: int) -> int:
    """Validate an integer while rejecting booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}.")
    return result


def validate_real(
    value: float,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate a finite real scalar and optional closed bounds."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {result}.")
    return result


def infer_num_qubits(state: ArrayLike) -> int:
    """Infer N from a nonempty power-of-two statevector length."""

    size = int(np.asarray(state).size)
    if size < 2 or size & (size - 1):
        raise ValueError(f"Statevector length must be a power of two >= 2, got {size}.")
    return size.bit_length() - 1


def normalized_statevector(state: ArrayLike) -> NDArray[np.complex128]:
    """Return a normalized finite statevector without fixing global phase."""

    vector = as_complex128(state, copy=True).reshape(-1)
    infer_num_qubits(vector)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Statevector must have a finite nonzero norm.")
    return vector / norm


def validate_normalized_statevector(
    state: ArrayLike,
    *,
    atol: float = 1e-12,
) -> NDArray[np.complex128]:
    """Validate, but do not silently normalize, a unit-norm statevector."""

    atol = validate_real(atol, name="atol", minimum=0.0)
    vector = as_complex128(state, copy=True).reshape(-1)
    infer_num_qubits(vector)
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > atol:
        raise ValueError(f"Statevector must be normalized; observed norm={norm:.17g}.")
    return vector


def phase_fix_state(state: ArrayLike) -> NDArray[np.complex128]:
    """Fix global phase at the first maximum-modulus amplitude.

    Rudolph's fractional principal-branch update is not invariant under an
    arbitrary target global phase.  This physical gauge makes the protocol
    deterministic: ties use the smallest computational-basis index.
    """

    vector = normalized_statevector(state)
    pivot = int(np.argmax(np.abs(vector)))
    magnitude = abs(vector[pivot])
    if magnitude == 0.0:  # unreachable for a normalized vector
        raise FloatingPointError("Cannot phase-fix a zero statevector.")
    vector *= np.exp(-1j * np.angle(vector[pivot]))
    vector[pivot] = complex(abs(vector[pivot]), 0.0)
    return vector


def phase_aligned_state(reference: ArrayLike, candidate: ArrayLike) -> NDArray[np.complex128]:
    """Align a candidate to a reference for diagnostic comparisons only."""

    ref = normalized_statevector(reference)
    cand = normalized_statevector(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"Statevector shape mismatch: {ref.shape} vs {cand.shape}.")
    overlap = np.vdot(ref, cand)
    if abs(overlap) > 0.0:
        cand *= np.exp(-1j * np.angle(overlap))
    return cand


def state_distance_up_to_global_phase(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return normalized L2 distance after optimal global-phase alignment."""

    ref = normalized_statevector(reference)
    return float(np.linalg.norm(ref - phase_aligned_state(ref, candidate)))


__all__ = [
    "COMPLEX_DTYPE",
    "LAYER_GATE_ORDER",
    "MPS_AXIS_ORDER",
    "QUBIT_ORDER",
    "REAL_DTYPE",
    "TWO_QUBIT_BASIS",
    "as_complex128",
    "infer_num_qubits",
    "normalized_statevector",
    "phase_aligned_state",
    "phase_fix_state",
    "state_distance_up_to_global_phase",
    "validate_integer",
    "validate_normalized_statevector",
    "validate_real",
]

