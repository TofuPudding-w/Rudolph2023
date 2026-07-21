from __future__ import annotations

import numpy as np
import pytest

from rudolph2023.circuit import (
    StaircaseCircuit,
    StaircaseLayer,
    apply_adjacent_u4,
    zero_state,
)
from rudolph2023.conventions import normalized_statevector
from rudolph2023.linalg import damped_unitary_update, polar_optimal_unitary
from rudolph2023.optimizer import (
    build_suffix_states,
    environment_overlap_identity,
    forward_sweep,
    gate_environment,
    local_state_matrix,
    optimize_all,
)


def random_unitary(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q_matrix, _ = np.linalg.qr(matrix)
    return q_matrix


def random_circuit(num_qubits: int, depth: int, seed: int) -> StaircaseCircuit:
    rng = np.random.default_rng(seed)
    layers = []
    for _layer in range(depth):
        gates = []
        for _site in range(num_qubits - 1):
            matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            q_matrix, _ = np.linalg.qr(matrix)
            gates.append(q_matrix)
        layers.append(StaircaseLayer(num_qubits, tuple(gates)))
    return StaircaseCircuit(num_qubits, tuple(layers))


def random_state(num_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalized_statevector(
        rng.normal(size=1 << num_qubits) + 1j * rng.normal(size=1 << num_qubits)
    )


def test_local_state_matrix_matches_manual_axis_permutation() -> None:
    state = np.arange(16, dtype=np.complex128) + 1j * np.arange(16)[::-1]
    observed = local_state_matrix(state, 1, num_qubits=4)
    expected = np.moveaxis(state.reshape(2, 2, 2, 2), (1, 2), (0, 1)).reshape(4, -1)
    assert np.array_equal(observed, expected)


def test_suffix_cache_matches_independent_tail_replay() -> None:
    circuit = random_circuit(4, 2, 10)
    target = random_state(4, 11)
    suffixes = build_suffix_states(target, circuit)
    gates = circuit.flat_gates
    for start in range(len(gates) + 1):
        expected = target.copy()
        for gate in reversed(gates[start:]):
            expected = apply_adjacent_u4(
                expected,
                gate.matrix.conj().T,
                gate.left_site,
                num_qubits=4,
                validate_matrix=False,
            )
        assert np.max(np.abs(suffixes[start] - expected)) < 2e-14


@pytest.mark.parametrize("left_site", [0, 1, 2])
def test_environment_identity_equals_direct_global_overlap(left_site: int) -> None:
    prefix = random_state(4, 20)
    suffix = random_state(4, 21)
    gate = random_unitary(22)
    environment = gate_environment(
        prefix, suffix, left_site=left_site, num_qubits=4
    )
    via_environment, via_state, error = environment_overlap_identity(
        environment,
        gate,
        prefix,
        suffix,
        left_site=left_site,
        num_qubits=4,
    )
    assert error < 2e-14
    assert via_environment == pytest.approx(via_state, abs=2e-14)


def test_environment_every_matrix_element_matches_independent_full_replay() -> None:
    circuit = random_circuit(4, 2, 702)
    target = random_state(4, 703)
    gates = circuit.flat_gates
    gate_index = 3
    prefix = zero_state(4)
    for gate in gates[:gate_index]:
        prefix = apply_adjacent_u4(
            prefix,
            gate.matrix,
            gate.left_site,
            num_qubits=4,
            validate_matrix=False,
        )
    suffix = target.copy()
    for gate in reversed(gates[gate_index + 1 :]):
        suffix = apply_adjacent_u4(
            suffix,
            gate.matrix.conj().T,
            gate.left_site,
            num_qubits=4,
            validate_matrix=False,
        )
    environment = gate_environment(
        prefix,
        suffix,
        left_site=gates[gate_index].left_site,
        num_qubits=4,
    )
    for row in range(4):
        for column in range(4):
            basis = np.zeros((4, 4), dtype=np.complex128)
            basis[row, column] = 1.0
            replay = zero_state(4)
            for index, gate in enumerate(gates):
                matrix = basis if index == gate_index else gate.matrix
                replay = apply_adjacent_u4(
                    replay,
                    matrix,
                    gate.left_site,
                    num_qubits=4,
                    validate_matrix=False,
                )
            direct = np.vdot(target, replay)
            assert np.vdot(environment, basis) == pytest.approx(direct, abs=3e-14)


def naive_forward_sweep(
    target: np.ndarray,
    circuit: StaircaseCircuit,
    rate: float,
) -> StaircaseCircuit:
    old_gates = circuit.flat_gates
    updated: list[np.ndarray] = []
    for index, gate in enumerate(old_gates):
        prefix = zero_state(circuit.num_qubits)
        for previous, matrix in zip(old_gates[:index], updated, strict=True):
            prefix = apply_adjacent_u4(
                prefix,
                matrix,
                previous.left_site,
                num_qubits=circuit.num_qubits,
                validate_matrix=False,
            )
        suffix = target.copy()
        for future in reversed(old_gates[index + 1 :]):
            suffix = apply_adjacent_u4(
                suffix,
                future.matrix.conj().T,
                future.left_site,
                num_qubits=circuit.num_qubits,
                validate_matrix=False,
            )
        environment = gate_environment(
            prefix,
            suffix,
            left_site=gate.left_site,
            num_qubits=circuit.num_qubits,
        )
        optimal = polar_optimal_unitary(environment).unitary
        updated.append(
            damped_unitary_update(gate.matrix, optimal, learning_rate=rate).unitary
        )
    return circuit.with_flat_matrices(updated)


@pytest.mark.parametrize("rate", [0.0, 0.6, 1.0])
def test_cached_forward_sweep_matches_naive_gauss_seidel(rate: float) -> None:
    circuit = random_circuit(4, 2, 30)
    target = random_state(4, 31)
    cached, record = forward_sweep(target, circuit, learning_rate=rate)
    naive = naive_forward_sweep(target, circuit, rate)
    assert np.max(np.abs(cached.gate_array() - naive.gate_array())) < 5e-12
    assert np.max(np.abs(cached.prepare() - naive.prepare())) < 5e-12
    assert record.max_environment_identity_error < 5e-14
    assert record.replay_error < 5e-14
    assert len(record.gate_updates) == circuit.num_gates
    assert record.amplitude_gain == pytest.approx(
        record.amplitude_fidelity_after - record.amplitude_fidelity_before
    )
    assert record.squared_fidelity_gain == pytest.approx(
        record.squared_fidelity_after - record.squared_fidelity_before
    )


def test_learning_rate_zero_is_byte_stable_and_rate_one_reaches_each_local_polar() -> None:
    circuit = random_circuit(3, 1, 40)
    target = random_state(3, 41)
    unchanged, _ = forward_sweep(target, circuit, learning_rate=0.0)
    assert np.max(np.abs(unchanged.gate_array() - circuit.gate_array())) < 2e-14

    full, record = forward_sweep(target, circuit, learning_rate=1.0)
    assert full.num_gates == circuit.num_gates
    for update in record.gate_updates:
        assert update.amplitude_after == pytest.approx(
            update.environment_nuclear_norm, abs=2e-12
        )
        assert update.amplitude_after >= update.amplitude_before - 2e-12
    for previous, following in zip(
        record.gate_updates[:-1], record.gate_updates[1:], strict=True
    ):
        assert previous.overlap_after == pytest.approx(
            following.overlap_before, abs=3e-13
        )


def test_optimize_all_uses_fixed_budget_and_exact_update_count() -> None:
    circuit = random_circuit(5, 3, 50)
    target = random_state(5, 51)
    result = optimize_all(target, circuit, sweeps=4, learning_rate=0.6)
    assert len(result.sweeps) == 4
    assert result.total_gate_updates == 4 * 3 * 4
    assert result.requested_sweeps == 4
    assert result.optimized_circuit.depth == 3
    assert np.linalg.norm(result.optimized_circuit.prepare()) == pytest.approx(1.0, abs=1e-12)


def test_damped_updates_record_negative_gains_without_false_monotonic_assertion() -> None:
    # This fixed case proves principal-branch r=.6 is not a general monotone
    # line search; the implementation must record the loss rather than silently
    # replace Eq. (6) with a different phase-aligned update.
    seed = 127
    circuit = random_circuit(3, 1, 100 + seed)
    target = random_state(3, 500 + seed)
    _updated, record = forward_sweep(target, circuit, learning_rate=0.6)
    assert min(update.squared_fidelity_gain for update in record.gate_updates) < -0.1
