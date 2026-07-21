"""Ran analytic ``D_i`` layer adapted to Rudolph's pure-U(4) ansatz."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .circuit import StaircaseLayer, apply_adjacent_u4, zero_state
from .completion import complete_unitary_from_columns
from .conventions import COMPLEX_DTYPE, infer_num_qubits, validate_normalized_statevector
from .linalg import unitarity_residual, validate_unitary
from .mps import (
    MPS,
    TruncationResult,
    dense_to_mps,
    mps_to_dense,
    truncate_mps,
    validate_mps,
)

ComplexArray = NDArray[np.complex128]


def _preparation_u4_from_tensor(
    tensor: ComplexArray,
    *,
    atol: float,
) -> ComplexArray:
    chi_left, _, chi_right = tensor.shape
    prescribed = np.zeros((4, chi_left), dtype=COMPLEX_DTYPE)
    for left_bond in range(chi_left):
        for physical in range(2):
            for right_bond in range(chi_right):
                prescribed[2 * physical + right_bond, left_bond] = tensor[
                    left_bond, physical, right_bond
                ]
    input_columns = tuple(2 * left_bond for left_bond in range(chi_left))
    return complete_unitary_from_columns(
        prescribed,
        input_columns,
        dimension=4,
        atol=atol,
    )


def _terminal_u2_from_tensor(tensor: ComplexArray, *, atol: float) -> ComplexArray:
    chi_left, _, chi_right = tensor.shape
    if chi_right != 1:
        raise ValueError("Terminal MPS tensor must have right bond dimension one.")
    prescribed = np.zeros((2, chi_left), dtype=COMPLEX_DTYPE)
    for left_bond in range(chi_left):
        prescribed[:, left_bond] = tensor[left_bond, :, 0]
    return complete_unitary_from_columns(
        prescribed,
        tuple(range(chi_left)),
        dimension=2,
        atol=atol,
    )


def fold_terminal_u2_into_last_u4(
    two_qubit_gates: tuple[ArrayLike, ...],
    terminal_u2: ArrayLike,
) -> tuple[ComplexArray, ...]:
    """Fold the terminal gate as ``(I2 kron U2) @ U4_last``.

    This turns Ran's ``N-1 U(4) + U(2)`` layer into exactly the ``N-1 U(4)``
    macro-layer counted by Rudolph et al.
    """

    gates = tuple(validate_unitary(gate, dimension=4) for gate in two_qubit_gates)
    if not gates:
        raise ValueError("At least one U(4) is required; Rudolph layers need N>=2.")
    terminal = validate_unitary(terminal_u2, dimension=2)
    folded = list(gates)
    folded[-1] = np.kron(np.eye(2, dtype=COMPLEX_DTYPE), terminal) @ folded[-1]
    if unitarity_residual(folded[-1]) > 1e-12:
        raise ArithmeticError("Folding the terminal U(2) lost unitarity.")
    return tuple(np.asarray(gate, dtype=COMPLEX_DTYPE) for gate in folded)


def _prepare_unfolded(
    num_qubits: int,
    two_qubit_gates: tuple[ComplexArray, ...],
    terminal_u2: ComplexArray,
) -> ComplexArray:
    state = zero_state(num_qubits)
    for left_site, gate in enumerate(two_qubit_gates):
        state = apply_adjacent_u4(
            state, gate, left_site, num_qubits=num_qubits, validate_matrix=False
        )
    tensor = state.reshape((2,) * num_qubits)
    leading = np.moveaxis(tensor, num_qubits - 1, 0)
    transformed = terminal_u2 @ leading.reshape(2, -1)
    return np.moveaxis(
        transformed.reshape(leading.shape), 0, num_qubits - 1
    ).reshape(-1)


@dataclass(frozen=True, slots=True)
class Chi2LayerEncoding:
    """Exact chi<=2 encoding and terminal-fold validation diagnostics."""

    layer: StaircaseLayer
    unfolded_u4: tuple[ComplexArray, ...]
    terminal_u2: ComplexArray
    exact_state_error: float
    fold_replay_error: float
    max_unitarity_residual: float


def chi2_mps_to_layer(
    mps: MPS,
    *,
    canonical_atol: float = 1e-12,
    replay_atol: float = 1e-10,
) -> Chi2LayerEncoding:
    """Map a normalized Ran-canonical chi<=2 MPS to one pure-U(4) layer."""

    if not math.isfinite(canonical_atol) or canonical_atol <= 0.0:
        raise ValueError("canonical_atol must be finite and positive.")
    if not math.isfinite(replay_atol) or replay_atol <= 0.0:
        raise ValueError("replay_atol must be finite and positive.")
    tensors = validate_mps(mps)
    if len(tensors) < 2:
        raise ValueError("Rudolph pure-U(4) layers require at least two qubits.")
    if any(max(tensor.shape[0], tensor.shape[2]) > 2 for tensor in tensors):
        raise ValueError("Single-layer exact encoding requires every bond <=2.")
    gram_infinity_residuals = []
    for tensor in tensors:
        matrix = tensor.reshape(tensor.shape[0], -1)
        gram_error = matrix @ matrix.conj().T - np.eye(
            matrix.shape[0], dtype=COMPLEX_DTYPE
        )
        gram_infinity_residuals.append(
            float(np.linalg.norm(gram_error, ord=np.inf))
        )
    max_canonical_error = max(gram_infinity_residuals)
    if max_canonical_error > canonical_atol:
        raise ValueError(
            f"MPS is not Ran canonical; infinity residual={max_canonical_error:.3e}."
        )

    unfolded = tuple(
        _preparation_u4_from_tensor(tensor, atol=canonical_atol)
        for tensor in tensors[:-1]
    )
    terminal = _terminal_u2_from_tensor(tensors[-1], atol=canonical_atol)
    folded = fold_terminal_u2_into_last_u4(unfolded, terminal)
    layer = StaircaseLayer(len(tensors), folded)

    target = mps_to_dense(mps)
    unfolded_state = _prepare_unfolded(len(tensors), unfolded, terminal)
    folded_state = layer.prepare()
    exact_error = float(np.max(np.abs(unfolded_state - target)))
    fold_error = float(np.max(np.abs(folded_state - unfolded_state)))
    if exact_error > replay_atol:
        raise ArithmeticError(f"Chi=2 layer does not reproduce its MPS: {exact_error:.3e}.")
    if fold_error > replay_atol:
        raise ArithmeticError(f"Terminal U(2) fold changed the state: {fold_error:.3e}.")
    max_unitarity = max(unitarity_residual(matrix) for matrix in folded)
    return Chi2LayerEncoding(
        layer=layer,
        unfolded_u4=unfolded,
        terminal_u2=terminal,
        exact_state_error=exact_error,
        fold_replay_error=fold_error,
        max_unitarity_residual=max_unitarity,
    )


@dataclass(frozen=True, slots=True)
class AnalyticStepResult:
    """One verified (D_i) extraction from an exact dense residual."""

    layer: StaircaseLayer
    compression: TruncationResult
    encoding: Chi2LayerEncoding
    compressed_state: ComplexArray
    next_residual: ComplexArray
    input_norm: float
    output_norm: float
    extracted_overlap: complex
    expected_overlap: complex
    extracted_overlap_error: float
    preparation_error: float


def analytic_step(
    residual_state: ArrayLike,
    *,
    validation_atol: float = 1e-10,
) -> AnalyticStepResult:
    """Extract one Ran (D_i) layer without truncating the residual itself."""

    if not math.isfinite(validation_atol) or validation_atol <= 0.0:
        raise ValueError("validation_atol must be finite and positive.")
    residual = validate_normalized_statevector(residual_state)
    num_qubits = infer_num_qubits(residual)
    if num_qubits < 2:
        raise ValueError("Rudolph pure-U(4) encoding requires N>=2.")
    exact_mps = dense_to_mps(residual, normalize=False)
    compression = truncate_mps(exact_mps, max_bond=2, normalize=True)
    encoding = chi2_mps_to_layer(
        compression.mps,
        canonical_atol=1e-12,
        replay_atol=validation_atol,
    )
    compressed_state = mps_to_dense(compression.mps)
    prepared_state = encoding.layer.prepare()
    preparation_error = float(np.max(np.abs(prepared_state - compressed_state)))
    if preparation_error > validation_atol:
        raise ArithmeticError(
            f"Analytic preparation differs from compressed MPS: {preparation_error:.3e}."
        )

    next_residual = encoding.layer.adjoint_apply(residual)
    input_norm = float(np.linalg.norm(residual))
    output_norm = float(np.linalg.norm(next_residual))
    if abs(input_norm - output_norm) > validation_atol:
        raise ArithmeticError("Analytic disentangler failed norm conservation.")
    extracted = complex(next_residual[0])
    expected = complex(np.vdot(compressed_state, residual))
    overlap_error = float(abs(extracted - expected))
    if overlap_error > validation_atol:
        raise ArithmeticError(
            "Analytic overlap identity <0|L^dagger|psi>=<psi_chi2|psi> failed."
        )
    return AnalyticStepResult(
        layer=encoding.layer,
        compression=compression,
        encoding=encoding,
        compressed_state=compressed_state,
        next_residual=next_residual,
        input_norm=input_norm,
        output_norm=output_norm,
        extracted_overlap=extracted,
        expected_overlap=expected,
        extracted_overlap_error=overlap_error,
        preparation_error=preparation_error,
    )


__all__ = [
    "AnalyticStepResult",
    "Chi2LayerEncoding",
    "analytic_step",
    "chi2_mps_to_layer",
    "fold_terminal_u2_into_last_u4",
]
