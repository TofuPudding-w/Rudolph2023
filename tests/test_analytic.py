from __future__ import annotations

import numpy as np
import pytest

from rudolph2023.analytic import (
    analytic_step,
    chi2_mps_to_layer,
    fold_terminal_u2_into_last_u4,
)
from rudolph2023.circuit import StaircaseLayer, apply_adjacent_u4, zero_state
from rudolph2023.conventions import normalized_statevector, state_distance_up_to_global_phase
from rudolph2023.linalg import unitarity_residual
from rudolph2023.mps import MPS, canonicalize_to_ran, dense_to_mps, mps_to_dense


def random_unitary(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q_matrix, _ = np.linalg.qr(matrix)
    return q_matrix


def apply_last_u2(state: np.ndarray, unitary: np.ndarray, num_qubits: int) -> np.ndarray:
    tensor = state.reshape((2,) * num_qubits)
    leading = np.moveaxis(tensor, num_qubits - 1, 0)
    output = unitary @ leading.reshape(2, -1)
    return np.moveaxis(output.reshape(leading.shape), 0, num_qubits - 1).reshape(-1)


def test_terminal_u2_fold_is_exact_for_arbitrary_input_not_only_zero_state() -> None:
    num_qubits = 4
    u4s = tuple(random_unitary(4, seed) for seed in (1, 2, 3))
    # Deliberately asymmetric complex U2 catches Kronecker and multiplication reversals.
    u2 = np.array(
        [[1.0, 1.0j], [1.0j, 1.0]], dtype=np.complex128
    ) / np.sqrt(2.0)
    folded = fold_terminal_u2_into_last_u4(u4s, u2)
    rng = np.random.default_rng(4)
    state = normalized_statevector(
        rng.normal(size=16) + 1j * rng.normal(size=16)
    )
    unfolded_state = state
    for left_site, gate in enumerate(u4s):
        unfolded_state = apply_adjacent_u4(
            unfolded_state, gate, left_site, num_qubits=num_qubits
        )
    unfolded_state = apply_last_u2(unfolded_state, u2, num_qubits)
    folded_state = StaircaseLayer(num_qubits, folded).apply(state)
    assert np.max(np.abs(folded_state - unfolded_state)) < 3e-14
    assert max(unitarity_residual(gate) for gate in folded) < 1e-12


@pytest.mark.parametrize("num_qubits", range(2, 8))
def test_chi2_mps_maps_exactly_to_one_pure_u4_layer(num_qubits: int) -> None:
    source_layer = StaircaseLayer(
        num_qubits,
        tuple(random_unitary(4, 1000 + site) for site in range(num_qubits - 1)),
    )
    state = source_layer.prepare()
    mps = canonicalize_to_ran(dense_to_mps(state), normalize=True)
    assert mps.max_bond_dimension <= 2
    encoding = chi2_mps_to_layer(mps)
    assert len(encoding.layer.matrices) == num_qubits - 1
    assert state_distance_up_to_global_phase(state, encoding.layer.prepare()) < 2e-12
    assert encoding.exact_state_error < 2e-12
    assert encoding.fold_replay_error < 2e-12


@pytest.mark.parametrize("num_qubits", range(2, 8))
def test_general_random_row_isometric_mps_maps_to_layer(num_qubits: int) -> None:
    rng = np.random.default_rng(3000 + num_qubits)
    bonds = [1, *(2 for _ in range(num_qubits - 1)), 1]
    tensors = []
    for site in range(num_qubits):
        left, right = bonds[site], bonds[site + 1]
        raw = rng.normal(size=(2 * right, left)) + 1j * rng.normal(
            size=(2 * right, left)
        )
        q_matrix, _ = np.linalg.qr(raw)
        tensors.append(q_matrix.conj().T.reshape(left, 2, right))
    mps = MPS(tensors)
    state = mps_to_dense(mps)
    assert np.linalg.norm(state) == pytest.approx(1.0, abs=2e-13)
    encoding = chi2_mps_to_layer(mps)
    assert np.max(np.abs(encoding.layer.prepare() - state)) < 3e-13


def test_analytic_layer_exact_on_basis_ghz_and_w_boundary_cases() -> None:
    cases = []
    for num_qubits in range(2, 7):
        for index in range(1 << num_qubits):
            state = np.zeros(1 << num_qubits, dtype=np.complex128)
            state[index] = 1.0
            cases.append(state)
    ghz = np.zeros(32, dtype=np.complex128)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    w_state = np.zeros(32, dtype=np.complex128)
    for site in range(5):
        w_state[1 << site] = 1 / np.sqrt(5)
    cases.extend((ghz, w_state))
    for state in cases:
        result = analytic_step(state)
        assert np.max(np.abs(result.layer.prepare() - state)) < 5e-13
        assert result.compression.squared_fidelity == pytest.approx(1.0, abs=1e-12)


def test_analytic_tolerance_arguments_are_strict_and_separate() -> None:
    mps = canonicalize_to_ran(dense_to_mps(zero_state(3)))
    with pytest.raises(ValueError, match="canonical_atol"):
        chi2_mps_to_layer(mps, canonical_atol=0.0)
    with pytest.raises(ValueError, match="replay_atol"):
        chi2_mps_to_layer(mps, replay_atol=0.0)


def test_product_and_bell_states_are_machine_precision_analytic_cases() -> None:
    product = zero_state(4)
    product_step = analytic_step(product)
    assert abs(product_step.extracted_overlap) == pytest.approx(1.0, abs=1e-13)
    assert np.max(np.abs(product_step.layer.prepare() - product)) < 1e-13

    bell = np.zeros(4, dtype=np.complex128)
    bell[0] = bell[3] = 1 / np.sqrt(2)
    bell_step = analytic_step(bell)
    assert abs(bell_step.extracted_overlap) == pytest.approx(1.0, abs=1e-13)
    assert np.max(np.abs(bell_step.layer.prepare() - bell)) < 1e-13


@pytest.mark.parametrize("seed", range(12))
def test_analytic_step_high_bond_invariants_under_stress(seed: int) -> None:
    rng = np.random.default_rng(seed)
    residual = normalized_statevector(
        rng.normal(size=64) + 1j * rng.normal(size=64)
    )
    result = analytic_step(residual)
    assert result.compression.mps.max_bond_dimension <= 2
    assert result.input_norm == pytest.approx(1.0, abs=2e-13)
    assert result.output_norm == pytest.approx(1.0, abs=2e-13)
    assert result.extracted_overlap_error < 2e-12
    assert result.preparation_error < 2e-12
    direct = result.layer.adjoint_apply(residual)
    assert np.max(np.abs(direct - result.next_residual)) < 1e-14
    assert direct[0] == pytest.approx(np.vdot(result.compressed_state, residual), abs=2e-12)


def test_analytic_step_rejects_non_normalized_input() -> None:
    with pytest.raises(ValueError):
        analytic_step(np.ones(4, dtype=np.complex128))
