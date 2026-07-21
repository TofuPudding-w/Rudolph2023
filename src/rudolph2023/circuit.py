"""Pure-U(4) staircase circuit with explicit physical execution order."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import (
    COMPLEX_DTYPE,
    as_complex128,
    infer_num_qubits,
    validate_integer,
    validate_normalized_statevector,
)
from .linalg import validate_unitary

ComplexArray = NDArray[np.complex128]


def zero_state(num_qubits: int) -> ComplexArray:
    """Return ``|0>^N`` in q0-most-significant order."""

    num_qubits = validate_integer(num_qubits, name="num_qubits", minimum=2)
    state = np.zeros(1 << num_qubits, dtype=COMPLEX_DTYPE)
    state[0] = 1.0
    return state


def apply_adjacent_u4(
    state: ArrayLike,
    matrix: ArrayLike,
    left_site: int,
    *,
    num_qubits: int | None = None,
    validate_matrix: bool = True,
) -> ComplexArray:
    """Apply a row-output/column-input U(4) in local basis 00,01,10,11."""

    vector = as_complex128(state, copy=True).reshape(-1)
    inferred = infer_num_qubits(vector)
    num_qubits = inferred if num_qubits is None else validate_integer(
        num_qubits, name="num_qubits", minimum=2
    )
    if inferred != num_qubits:
        raise ValueError("num_qubits is inconsistent with statevector length.")
    left_site = validate_integer(left_site, name="left_site", minimum=0)
    if left_site >= num_qubits - 1:
        raise IndexError("Adjacent gate lies outside the register.")
    gate = (
        validate_unitary(matrix, dimension=4)
        if validate_matrix
        else np.asarray(matrix, dtype=COMPLEX_DTYPE)
    )
    if gate.shape != (4, 4):
        raise ValueError("Adjacent gate must be 4x4.")
    tensor = vector.reshape((2,) * num_qubits)
    leading = np.moveaxis(tensor, (left_site, left_site + 1), (0, 1))
    transformed = gate @ leading.reshape(4, -1)
    restored = np.moveaxis(
        transformed.reshape(leading.shape),
        (0, 1),
        (left_site, left_site + 1),
    )
    return np.asarray(restored.reshape(-1), dtype=COMPLEX_DTYPE)


def embed_adjacent_u4(matrix: ArrayLike, left_site: int, num_qubits: int) -> ComplexArray:
    """Embed an adjacent U(4) as a full matrix for independent small-N oracles."""

    num_qubits = validate_integer(num_qubits, name="num_qubits", minimum=2)
    left_site = validate_integer(left_site, name="left_site", minimum=0)
    if left_site >= num_qubits - 1:
        raise IndexError("Adjacent gate lies outside the register.")
    gate = validate_unitary(matrix, dimension=4)
    left_identity = np.eye(1 << left_site, dtype=COMPLEX_DTYPE)
    right_count = num_qubits - left_site - 2
    right_identity = np.eye(1 << right_count, dtype=COMPLEX_DTYPE)
    return np.kron(np.kron(left_identity, gate), right_identity)


@dataclass(frozen=True, slots=True)
class StaircaseLayer:
    """One left-to-right macro-layer containing exactly N-1 U(4) gates."""

    num_qubits: int
    matrices: tuple[ComplexArray, ...]

    def __post_init__(self) -> None:
        num_qubits = validate_integer(self.num_qubits, name="num_qubits", minimum=2)
        matrices = tuple(
            validate_unitary(matrix, dimension=4) for matrix in tuple(self.matrices)
        )
        if len(matrices) != num_qubits - 1:
            raise ValueError(f"A layer needs N-1={num_qubits - 1} U(4) gates.")
        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "matrices", matrices)

    @classmethod
    def identity(cls, num_qubits: int) -> StaircaseLayer:
        num_qubits = validate_integer(num_qubits, name="num_qubits", minimum=2)
        return cls(
            num_qubits,
            tuple(np.eye(4, dtype=COMPLEX_DTYPE) for _ in range(num_qubits - 1)),
        )

    def apply(self, state: ArrayLike) -> ComplexArray:
        result = as_complex128(state, copy=True).reshape(-1)
        if infer_num_qubits(result) != self.num_qubits:
            raise ValueError("State size does not match layer width.")
        for left_site, matrix in enumerate(self.matrices):
            result = apply_adjacent_u4(
                result, matrix, left_site, num_qubits=self.num_qubits, validate_matrix=False
            )
        return result

    def adjoint_apply(self, state: ArrayLike) -> ComplexArray:
        result = as_complex128(state, copy=True).reshape(-1)
        if infer_num_qubits(result) != self.num_qubits:
            raise ValueError("State size does not match layer width.")
        for left_site in range(self.num_qubits - 2, -1, -1):
            result = apply_adjacent_u4(
                result,
                self.matrices[left_site].conj().T,
                left_site,
                num_qubits=self.num_qubits,
                validate_matrix=False,
            )
        return result

    def prepare(self) -> ComplexArray:
        return self.apply(zero_state(self.num_qubits))

    def with_gate(self, position: int, matrix: ArrayLike) -> StaircaseLayer:
        position = validate_integer(position, name="position", minimum=0)
        if position >= len(self.matrices):
            raise IndexError("Layer gate position is out of range.")
        matrices = list(self.matrices)
        matrices[position] = validate_unitary(matrix, dimension=4)
        return StaircaseLayer(self.num_qubits, tuple(matrices))


@dataclass(frozen=True, slots=True)
class LocatedU4:
    """One flattened gate with dynamic physical layer and position."""

    flat_index: int
    layer_index: int
    position: int
    left_site: int
    matrix: ComplexArray


@dataclass(frozen=True, slots=True)
class StaircaseCircuit:
    """Layers and gates stored in physical execution order."""

    num_qubits: int
    layers: tuple[StaircaseLayer, ...]

    def __post_init__(self) -> None:
        num_qubits = validate_integer(self.num_qubits, name="num_qubits", minimum=2)
        layers = tuple(self.layers)
        if any(not isinstance(layer, StaircaseLayer) for layer in layers):
            raise TypeError("layers must contain StaircaseLayer objects.")
        if any(layer.num_qubits != num_qubits for layer in layers):
            raise ValueError("All layers must have the circuit width.")
        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "layers", layers)

    @classmethod
    def empty(cls, num_qubits: int) -> StaircaseCircuit:
        return cls(num_qubits, ())

    @classmethod
    def from_gate_array(cls, gate_array: ArrayLike) -> StaircaseCircuit:
        array = as_complex128(gate_array, copy=True)
        if array.ndim != 4 or array.shape[2:] != (4, 4):
            raise ValueError("gate_array must have shape (depth,N-1,4,4).")
        depth, gates_per_layer = array.shape[:2]
        num_qubits = gates_per_layer + 1
        layers = tuple(
            StaircaseLayer(num_qubits, tuple(array[layer])) for layer in range(depth)
        )
        return cls(num_qubits, layers)

    @property
    def depth(self) -> int:
        return len(self.layers)

    @property
    def num_gates(self) -> int:
        return self.depth * (self.num_qubits - 1)

    @property
    def flat_gates(self) -> tuple[LocatedU4, ...]:
        gates: list[LocatedU4] = []
        for layer_index, layer in enumerate(self.layers):
            for position, matrix in enumerate(layer.matrices):
                gates.append(
                    LocatedU4(
                        flat_index=len(gates),
                        layer_index=layer_index,
                        position=position,
                        left_site=position,
                        matrix=matrix,
                    )
                )
        return tuple(gates)

    def gate_array(self) -> ComplexArray:
        if self.depth == 0:
            return np.empty((0, self.num_qubits - 1, 4, 4), dtype=COMPLEX_DTYPE)
        return np.asarray(
            [[matrix for matrix in layer.matrices] for layer in self.layers],
            dtype=COMPLEX_DTYPE,
        )

    def prepend_input_layer(self, layer: StaircaseLayer) -> StaircaseCircuit:
        """Insert a new analytic layer at the beginning of physical execution."""

        if not isinstance(layer, StaircaseLayer) or layer.num_qubits != self.num_qubits:
            raise ValueError("New layer must match the circuit width.")
        return StaircaseCircuit(self.num_qubits, (layer, *self.layers))

    def with_flat_matrices(self, matrices: Sequence[ArrayLike]) -> StaircaseCircuit:
        matrices = tuple(matrices)
        if len(matrices) != self.num_gates:
            raise ValueError("Replacement matrix count does not match circuit gates.")
        stride = self.num_qubits - 1
        layers = tuple(
            StaircaseLayer(
                self.num_qubits,
                tuple(matrices[start : start + stride]),
            )
            for start in range(0, len(matrices), stride)
        )
        return StaircaseCircuit(self.num_qubits, layers)

    def apply(self, state: ArrayLike) -> ComplexArray:
        result = as_complex128(state, copy=True).reshape(-1)
        if infer_num_qubits(result) != self.num_qubits:
            raise ValueError("State size does not match circuit width.")
        for layer in self.layers:
            result = layer.apply(result)
        return result

    def adjoint_apply(self, state: ArrayLike) -> ComplexArray:
        result = as_complex128(state, copy=True).reshape(-1)
        if infer_num_qubits(result) != self.num_qubits:
            raise ValueError("State size does not match circuit width.")
        for layer in reversed(self.layers):
            result = layer.adjoint_apply(result)
        return result

    def prepare(self) -> ComplexArray:
        return self.apply(zero_state(self.num_qubits))

    def overlap(self, target: ArrayLike) -> complex:
        target_state = validate_normalized_statevector(target)
        if infer_num_qubits(target_state) != self.num_qubits:
            raise ValueError("Target size does not match circuit width.")
        return complex(np.vdot(target_state, self.prepare()))


def apply_flat_gate_sequence(
    state: ArrayLike,
    gates: Iterable[LocatedU4],
    *,
    num_qubits: int,
) -> ComplexArray:
    """Apply located gates in iterable order; used by independent replay tests."""

    num_qubits = validate_integer(num_qubits, name="num_qubits", minimum=2)
    result = as_complex128(state, copy=True).reshape(-1)
    if infer_num_qubits(result) != num_qubits:
        raise ValueError("num_qubits is inconsistent with statevector length.")
    for gate in gates:
        result = apply_adjacent_u4(
            result, gate.matrix, gate.left_site, num_qubits=num_qubits, validate_matrix=False
        )
    return result


__all__ = [
    "LocatedU4",
    "StaircaseCircuit",
    "StaircaseLayer",
    "apply_adjacent_u4",
    "apply_flat_gate_sequence",
    "embed_adjacent_u4",
    "zero_state",
]
