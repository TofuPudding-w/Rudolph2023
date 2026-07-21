from __future__ import annotations

import numpy as np
import pytest

from rudolph2023.circuit import (
    StaircaseCircuit,
    StaircaseLayer,
    apply_adjacent_u4,
    apply_flat_gate_sequence,
    embed_adjacent_u4,
    zero_state,
)
from rudolph2023.completion import complete_unitary_from_columns
from rudolph2023.conventions import normalized_statevector
from rudolph2023.linalg import unitarity_residual


def random_unitary(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q_matrix, _ = np.linalg.qr(matrix)
    return q_matrix


def test_deterministic_completion_preserves_columns_and_is_unitary() -> None:
    prescribed = np.array(
        [[1 / np.sqrt(2), 0], [0, 1], [1j / np.sqrt(2), 0], [0, 0]],
        dtype=np.complex128,
    )
    first = complete_unitary_from_columns(prescribed, (0, 2))
    second = complete_unitary_from_columns(prescribed, (0, 2))
    assert np.array_equal(first, second)
    assert np.array_equal(first[:, (0, 2)], prescribed)
    assert unitarity_residual(first) < 1e-12
    with pytest.raises(ValueError):
        complete_unitary_from_columns(np.ones((4, 2)), (0, 2))


def test_completion_strict_integer_validation_and_empty_completion() -> None:
    prescribed = np.eye(4, dtype=np.complex128)[:, :1]
    for bad_dimension in (4.0, True):
        with pytest.raises(TypeError):
            complete_unitary_from_columns(
                prescribed, (0,), dimension=bad_dimension  # type: ignore[arg-type]
            )
    for bad_index in (0.9, True):
        with pytest.raises(TypeError):
            complete_unitary_from_columns(
                prescribed, (bad_index,)  # type: ignore[arg-type]
            )
    empty = complete_unitary_from_columns(
        np.empty((3, 0), dtype=np.complex128), (), dimension=3
    )
    assert np.array_equal(empty, np.eye(3))


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
def test_adjacent_executor_matches_full_kronecker_oracle(num_qubits: int) -> None:
    state = normalized_statevector(
        np.arange(1, (1 << num_qubits) + 1) + 1j * np.arange(1 << num_qubits)
    )
    for left_site in range(num_qubits - 1):
        gate = random_unitary(10 + left_site)
        observed = apply_adjacent_u4(state, gate, left_site, num_qubits=num_qubits)
        expected = embed_adjacent_u4(gate, left_site, num_qubits) @ state
        assert np.max(np.abs(observed - expected)) < 2e-14


def test_layer_forward_adjoint_and_full_matrix_order() -> None:
    gates = tuple(random_unitary(seed) for seed in (1, 2, 3))
    layer = StaircaseLayer(4, gates)
    state = normalized_statevector(np.arange(1, 17) + 0.3j * np.arange(16))
    full = np.eye(16, dtype=np.complex128)
    for left_site, gate in enumerate(gates):
        full = embed_adjacent_u4(gate, left_site, 4) @ full
    assert np.max(np.abs(layer.apply(state) - full @ state)) < 3e-14
    assert np.max(np.abs(layer.adjoint_apply(layer.apply(state)) - state)) < 4e-14
    assert np.linalg.norm(layer.prepare()) == pytest.approx(1.0, abs=2e-14)


def test_circuit_prepend_gate_array_and_independent_flat_replay() -> None:
    old = StaircaseLayer(3, (random_unitary(1), random_unitary(2)))
    new = StaircaseLayer(3, (random_unitary(3), random_unitary(4)))
    circuit = StaircaseCircuit.empty(3).prepend_input_layer(old)
    grown = circuit.prepend_input_layer(new)
    state = zero_state(3)
    expected = old.apply(new.apply(state))
    assert np.max(np.abs(grown.prepare() - expected)) < 2e-14
    replay = apply_flat_gate_sequence(state, grown.flat_gates, num_qubits=3)
    assert np.max(np.abs(replay - expected)) < 2e-14
    assert grown.num_gates == 4
    assert grown.gate_array().shape == (2, 2, 4, 4)
    loaded = StaircaseCircuit.from_gate_array(grown.gate_array())
    assert np.max(np.abs(loaded.prepare() - expected)) < 2e-14
    assert np.max(np.abs(grown.adjoint_apply(grown.apply(state)) - state)) < 5e-14


def test_circuit_layout_validation() -> None:
    identity = StaircaseLayer.identity(3)
    assert np.array_equal(identity.prepare(), zero_state(3))
    changed = identity.with_gate(0, random_unitary(99))
    assert np.max(np.abs(changed.matrices[0] - identity.matrices[0])) > 1e-3
    with pytest.raises(ValueError):
        StaircaseLayer(4, (np.eye(4),))
    with pytest.raises(ValueError):
        StaircaseCircuit.from_gate_array(np.zeros((2, 3, 3, 3)))
    with pytest.raises(ValueError):
        StaircaseCircuit.empty(3).with_flat_matrices([np.eye(4)])
    assert StaircaseCircuit.empty(3).overlap(zero_state(3)) == pytest.approx(1.0)
    assert np.array_equal(
        apply_flat_gate_sequence(zero_state(3), (), num_qubits=3), zero_state(3)
    )
    with pytest.raises(ValueError, match="inconsistent"):
        apply_flat_gate_sequence(zero_state(2), (), num_qubits=3)
