"""Open-boundary qubit MPS primitives and audited chi=2 TT-SVD truncation."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import COMPLEX_DTYPE, as_complex128, infer_num_qubits
from .linalg import deterministic_svd, svd_numerical_rank

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
MPSLike = "MPS | Sequence[ArrayLike]"


def _coerce_tensor(tensor: ArrayLike, *, copy: bool) -> ComplexArray:
    array = np.asarray(tensor)
    if array.ndim != 3:
        raise ValueError(f"MPS tensor must have three axes, got {array.shape}.")
    result = (
        np.array(array, dtype=COMPLEX_DTYPE, copy=True)
        if copy
        else np.asarray(array, dtype=COMPLEX_DTYPE)
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("MPS tensors must contain only finite values.")
    return result


def validate_mps(tensors: MPSLike) -> tuple[ComplexArray, ...]:
    """Validate an OBC qubit MPS in ``(chi_left, physical, chi_right)`` order."""

    raw = tensors.tensors if isinstance(tensors, MPS) else tuple(tensors)
    if not raw:
        raise ValueError("An MPS must contain at least one tensor.")
    checked = tuple(_coerce_tensor(tensor, copy=False) for tensor in raw)
    if checked[0].shape[0] != 1 or checked[-1].shape[2] != 1:
        raise ValueError("Open-boundary MPS endpoint bond dimensions must be one.")
    for site, tensor in enumerate(checked):
        if tensor.shape[1] != 2:
            raise ValueError(f"Site {site} must have physical dimension two.")
        if tensor.shape[0] < 1 or tensor.shape[2] < 1:
            raise ValueError(f"Site {site} has a nonpositive bond dimension.")
        if site and checked[site - 1].shape[2] != tensor.shape[0]:
            raise ValueError(f"Bond mismatch between sites {site - 1} and {site}.")
    return checked


@dataclass(frozen=True, init=False)
class MPS(Sequence[ComplexArray]):
    """Owned complex128 open-boundary qubit MPS container."""

    tensors: tuple[ComplexArray, ...]

    def __init__(self, tensors: Iterable[ArrayLike]) -> None:
        owned = tuple(_coerce_tensor(tensor, copy=True) for tensor in tensors)
        validate_mps(owned)
        object.__setattr__(self, "tensors", owned)

    def __len__(self) -> int:
        return len(self.tensors)

    def __iter__(self) -> Iterator[ComplexArray]:
        return iter(self.tensors)

    def __getitem__(self, item: int | slice) -> ComplexArray | tuple[ComplexArray, ...]:
        return self.tensors[item]

    @property
    def num_sites(self) -> int:
        return len(self.tensors)

    @property
    def bond_dimensions(self) -> tuple[int, ...]:
        return tuple(int(tensor.shape[2]) for tensor in self.tensors[:-1])

    @property
    def max_bond_dimension(self) -> int:
        return max(self.bond_dimensions, default=1)

    def copy(self) -> MPS:
        return type(self)(tensor.copy() for tensor in self.tensors)

    def to_dense(self) -> ComplexArray:
        return mps_to_dense(self)

    def norm(self) -> float:
        return mps_norm(self)


def as_mps(mps: MPSLike, *, copy: bool = False) -> MPS:
    """Return a validated MPS, with an optional deep copy."""

    if isinstance(mps, MPS):
        return mps.copy() if copy else mps
    return MPS(mps)


def mps_to_dense(mps: MPSLike) -> ComplexArray:
    """Contract an MPS into a q0-most-significant statevector."""

    tensors = validate_mps(mps)
    state = tensors[0][0, :, :]
    for tensor in tensors[1:]:
        state = np.tensordot(state, tensor, axes=([-1], [0]))
    return np.asarray(state[..., 0], dtype=COMPLEX_DTYPE).reshape(-1)


def dense_to_mps(
    state: ArrayLike,
    *,
    max_bond: int | None = None,
    normalize: bool = False,
) -> MPS:
    """Factor a state right-to-left into the Ran row-isometric gauge."""

    vector = as_complex128(state, copy=True).reshape(-1)
    num_qubits = infer_num_qubits(vector)
    raw_norm = float(np.linalg.norm(vector))
    if raw_norm <= 0.0 or not math.isfinite(raw_norm):
        raise ValueError("Statevector must have a finite nonzero norm.")
    if normalize:
        vector /= raw_norm
    if max_bond is not None:
        if isinstance(max_bond, bool) or not isinstance(max_bond, numbers.Integral):
            raise TypeError("max_bond must be an integer or None.")
        if max_bond < 1:
            raise ValueError("max_bond must be positive.")

    remaining = vector.reshape((2,) * num_qubits)
    right_bond = 1
    reversed_tensors: list[ComplexArray] = []
    for _site in range(num_qubits - 1, 0, -1):
        matrix = remaining.reshape(-1, 2 * right_bond)
        u_matrix, singular_values, vh_matrix = deterministic_svd(matrix, pivot_on="vh")
        rank = svd_numerical_rank(singular_values, matrix.shape)
        if max_bond is not None:
            rank = min(rank, int(max_bond))
        reversed_tensors.append(vh_matrix[:rank].reshape(rank, 2, right_bond))
        remaining = u_matrix[:, :rank] * singular_values[:rank]
        right_bond = rank
    first = remaining.reshape(1, 2, right_bond)
    result = MPS([first, *reversed(reversed_tensors)])
    return normalize_mps(result) if normalize else result


def mps_inner_product(bra: MPSLike, ket: MPSLike) -> complex:
    """Contract ``<bra|ket>`` without changing either norm."""

    bra_tensors = validate_mps(bra)
    ket_tensors = validate_mps(ket)
    if len(bra_tensors) != len(ket_tensors):
        raise ValueError("MPS site counts differ.")
    environment = np.ones((1, 1), dtype=COMPLEX_DTYPE)
    for bra_tensor, ket_tensor in zip(bra_tensors, ket_tensors, strict=True):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            environment = np.einsum(
                "ab,api,bpj->ij",
                environment,
                bra_tensor.conj(),
                ket_tensor,
                optimize=True,
            )
        if not np.all(np.isfinite(environment)):
            raise FloatingPointError("MPS contraction overflowed or became non-finite.")
    value = complex(environment[0, 0])
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise FloatingPointError("MPS inner product is non-finite.")
    return value


def mps_norm(mps: MPSLike) -> float:
    """Return the MPS norm after checking numerical reality and positivity."""

    value = mps_inner_product(mps, mps)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise FloatingPointError("MPS squared norm is non-finite.")
    scale = max(1.0, abs(value.real))
    if abs(value.imag) > 1e-12 * scale or value.real < -1e-13 * scale:
        raise FloatingPointError(f"Invalid MPS squared norm: {value}.")
    norm = math.sqrt(max(0.0, float(value.real)))
    if not math.isfinite(norm):
        raise FloatingPointError("MPS norm is non-finite.")
    return norm


def normalize_mps(mps: MPSLike) -> MPS:
    """Normalize an MPS by scaling only its first tensor."""

    state = as_mps(mps, copy=True)
    norm = mps_norm(state)
    if not math.isfinite(norm):
        raise FloatingPointError("Cannot normalize an MPS with non-finite norm.")
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero MPS.")
    tensors = list(state.tensors)
    tensors[0] = tensors[0] / norm
    return MPS(tensors)


def canonicalize_to_ran(mps: MPSLike, *, normalize: bool = True) -> MPS:
    """Sweep right-to-left so each tensor obeys ``A A^dagger=I``."""

    state = as_mps(mps, copy=True)
    raw_norm = mps_norm(state)
    if not math.isfinite(raw_norm):
        raise FloatingPointError("Cannot canonicalize an MPS with non-finite norm.")
    if raw_norm <= 0.0:
        raise ValueError("Cannot canonicalize a zero MPS.")
    tensors = [tensor.copy() for tensor in state.tensors]
    for site in range(state.num_sites - 1, 0, -1):
        tensor = tensors[site]
        matrix = tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])
        u_matrix, singular_values, vh_matrix = deterministic_svd(matrix, pivot_on="vh")
        rank = svd_numerical_rank(singular_values, matrix.shape)
        tensors[site] = vh_matrix[:rank].reshape(rank, tensor.shape[1], tensor.shape[2])
        transfer = u_matrix[:, :rank] * singular_values[:rank]
        tensors[site - 1] = np.einsum(
            "lpm,mr->lpr", tensors[site - 1], transfer, optimize=True
        )
    if normalize:
        tensors[0] /= raw_norm
    return MPS(tensors)


def ran_isometry_residuals(mps: MPSLike) -> tuple[float, ...]:
    """Return per-site normalized Frobenius residuals of ``A A^dagger=I``."""

    residuals: list[float] = []
    for tensor in validate_mps(mps):
        matrix = tensor.reshape(tensor.shape[0], -1)
        identity = np.eye(matrix.shape[0], dtype=COMPLEX_DTYPE)
        error = matrix @ matrix.conj().T - identity
        residuals.append(float(np.linalg.norm(error) / math.sqrt(identity.shape[0])))
    return tuple(residuals)


@dataclass(frozen=True, slots=True)
class TruncationResult:
    """A chi-limited MPS and complete TT-SVD diagnostics."""

    mps: MPS
    max_bond: int
    kept_ranks: tuple[int, ...]
    singular_values: tuple[RealArray, ...]
    discarded_weights: tuple[float, ...]
    input_norm: float
    truncated_norm_before_normalization: float
    output_norm: float
    normalized_overlap: float
    squared_fidelity: float
    isometry_residuals: tuple[float, ...]

    @property
    def total_discarded_weight(self) -> float:
        return float(sum(self.discarded_weights))


def truncate_mps(
    mps: MPSLike,
    *,
    max_bond: int,
    normalize: bool = True,
) -> TruncationResult:
    """Perform deterministic canonical TT-SVD rounding to ``max_bond``."""

    if isinstance(max_bond, bool) or not isinstance(max_bond, numbers.Integral):
        raise TypeError("max_bond must be an integer.")
    if max_bond < 1:
        raise ValueError("max_bond must be positive.")
    original = as_mps(mps)
    input_norm = mps_norm(original)
    if not math.isfinite(input_norm):
        raise FloatingPointError("Cannot truncate an MPS with non-finite norm.")
    right_canonical = canonicalize_to_ran(original, normalize=False)
    tensors = [tensor.copy() for tensor in right_canonical.tensors]
    singular_values_by_bond: list[RealArray] = []
    discarded_weights: list[float] = []
    kept_ranks: list[int] = []
    for site in range(original.num_sites - 1):
        tensor = tensors[site]
        matrix = tensor.reshape(tensor.shape[0] * 2, tensor.shape[2])
        u_matrix, singular_values, vh_matrix = deterministic_svd(matrix, pivot_on="u")
        rank = svd_numerical_rank(singular_values, matrix.shape)
        keep = min(int(max_bond), rank)
        singular_values_by_bond.append(singular_values.copy())
        discarded_weights.append(float(np.sum(np.square(singular_values[keep:]))))
        kept_ranks.append(keep)
        tensors[site] = u_matrix[:, :keep].reshape(tensor.shape[0], 2, keep)
        transfer = singular_values[:keep, None] * vh_matrix[:keep]
        tensors[site + 1] = np.einsum(
            "km,mpn->kpn", transfer, tensors[site + 1], optimize=True
        )

    truncated = MPS(tensors)
    truncated_norm = mps_norm(truncated)
    if not math.isfinite(truncated_norm):
        raise FloatingPointError("TT-SVD approximation has a non-finite norm.")
    if truncated_norm <= 0.0:
        raise FloatingPointError("TT-SVD produced a zero approximation.")
    output = canonicalize_to_ran(truncated, normalize=normalize)
    output_norm = mps_norm(output)
    if not math.isfinite(output_norm) or output_norm <= 0.0:
        raise FloatingPointError("TT-SVD output has an invalid norm.")
    overlap = mps_inner_product(original, output) / (input_norm * output_norm)
    amplitude = float(abs(overlap))
    return TruncationResult(
        mps=output,
        max_bond=int(max_bond),
        kept_ranks=tuple(kept_ranks),
        singular_values=tuple(singular_values_by_bond),
        discarded_weights=tuple(discarded_weights),
        input_norm=input_norm,
        truncated_norm_before_normalization=truncated_norm,
        output_norm=output_norm,
        normalized_overlap=amplitude,
        squared_fidelity=amplitude**2,
        isometry_residuals=ran_isometry_residuals(output),
    )


def open_boundary_bond_dimensions(num_qubits: int, max_bond: int) -> tuple[int, ...]:
    """Return the maximal feasible OBC bond profile capped at ``max_bond``."""

    if isinstance(num_qubits, bool) or not isinstance(num_qubits, numbers.Integral):
        raise TypeError("num_qubits must be an integer.")
    if isinstance(max_bond, bool) or not isinstance(max_bond, numbers.Integral):
        raise TypeError("max_bond must be an integer.")
    if num_qubits < 2 or max_bond < 1:
        raise ValueError("num_qubits must be >=2 and max_bond positive.")
    return tuple(
        min(1 << (bond + 1), int(max_bond), 1 << (num_qubits - bond - 1))
        for bond in range(num_qubits - 1)
    )


__all__ = [
    "MPS",
    "TruncationResult",
    "as_mps",
    "canonicalize_to_ran",
    "dense_to_mps",
    "mps_inner_product",
    "mps_norm",
    "mps_to_dense",
    "normalize_mps",
    "open_boundary_bond_dimensions",
    "ran_isometry_residuals",
    "truncate_mps",
    "validate_mps",
]
