from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from rudolph2023.conventions import phase_fix_state
from rudolph2023.metrics import (
    energy_error_per_site,
    energy_expectation,
    entanglement_entropy,
    fidelity_metrics,
)
from rudolph2023.mps import mps_to_dense, ran_isometry_residuals
from rudolph2023.targets import (
    PAULI_X,
    PAULI_Z,
    SPIN_X,
    SPIN_Y,
    SPIN_Z,
    bars_and_stripes_bitstrings,
    bars_and_stripes_state,
    exact_ground_state,
    grid_edges,
    heisenberg_grid_hamiltonian,
    random_real_mps,
    single_site_operator,
    tfim_hamiltonian,
    two_site_product_operator,
    xxz_hamiltonian,
)


def test_grid_edges_paper_4_by_3_oracle() -> None:
    edges = grid_edges(3, 4)
    horizontal = [edge for edge in edges if edge[1] - edge[0] == 1]
    vertical = [edge for edge in edges if edge[1] - edge[0] == 4]
    assert len(edges) == 17
    assert len(horizontal) == 9
    assert len(vertical) == 8
    assert len(set(edges)) == 17


def test_sparse_operator_embeddings_match_explicit_kronecker() -> None:
    observed_one = single_site_operator(PAULI_X, 1, 3).toarray()
    expected_one = np.kron(np.kron(np.eye(2), PAULI_X), np.eye(2))
    assert np.array_equal(observed_one, expected_one)
    observed_two = two_site_product_operator(PAULI_X, PAULI_Z, 0, 2, 3).toarray()
    expected_two = np.kron(np.kron(PAULI_X, np.eye(2)), PAULI_Z)
    assert np.array_equal(observed_two, expected_two)
    with pytest.raises(ValueError):
        two_site_product_operator(PAULI_X, PAULI_Z, 1, 1, 3)
    bad = PAULI_X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        two_site_product_operator(bad, PAULI_Z, 0, 1, 2)


def test_two_site_heisenberg_has_singlet_minus_three_quarters() -> None:
    hamiltonian = heisenberg_grid_hamiltonian(rows=1, columns=2)
    values = np.linalg.eigvalsh(hamiltonian.toarray())
    assert values[0] == pytest.approx(-0.75, abs=1e-14)
    assert values[1:] == pytest.approx([0.25, 0.25, 0.25], abs=1e-14)
    explicit = sum(
        two_site_product_operator(spin, spin, 0, 1, 2).toarray()
        for spin in (SPIN_X, SPIN_Y, SPIN_Z)
    )
    assert np.max(np.abs(hamiltonian.toarray() - explicit)) < 1e-15


def test_tfim_and_xxz_small_hamiltonian_definitions() -> None:
    tfim = tfim_hamiltonian(2, coupling_zz=1.2, field_x=0.3, field_z=0.2).toarray()
    explicit_tfim = (
        -1.2 * np.kron(PAULI_Z, PAULI_Z)
        - 0.3 * (np.kron(PAULI_X, np.eye(2)) + np.kron(np.eye(2), PAULI_X))
        - 0.2 * (np.kron(PAULI_Z, np.eye(2)) + np.kron(np.eye(2), PAULI_Z))
    )
    assert np.max(np.abs(tfim - explicit_tfim)) < 1e-15
    xxz = xxz_hamiltonian(2, delta=1.0).toarray()
    heisenberg = heisenberg_grid_hamiltonian(rows=1, columns=2).toarray()
    assert np.max(np.abs(xxz - heisenberg)) < 1e-15


def test_ground_state_dense_and_forced_sparse_solvers_cross_validate() -> None:
    hamiltonian = tfim_hamiltonian(5, coupling_zz=1.0, field_x=0.7)
    dense = exact_ground_state(hamiltonian, dense_dimension_cutoff=64)
    sparse = exact_ground_state(hamiltonian, dense_dimension_cutoff=2)
    assert dense.energy == pytest.approx(sparse.energy, abs=2e-11)
    assert abs(np.vdot(dense.state, sparse.state)) == pytest.approx(1.0, abs=2e-10)
    assert dense.residual < 1e-11
    assert sparse.residual < 1e-10
    exact_values = np.linalg.eigvalsh(hamiltonian.toarray())
    assert dense.first_excited_energy == pytest.approx(exact_values[1], abs=2e-12)
    assert dense.gap == pytest.approx(exact_values[1] - exact_values[0], abs=2e-12)


def test_ground_state_rejects_nonfinite_and_degenerate_targets() -> None:
    nonfinite = csr_matrix(np.diag([0.0, 1.0, 2.0, np.nan]).astype(np.complex128))
    with pytest.raises(ValueError, match="finite"):
        exact_ground_state(nonfinite)
    with pytest.raises(ValueError, match="degenerate"):
        exact_ground_state(np.eye(4, dtype=np.complex128))


def test_bars_and_stripes_support_and_uniform_amplitudes() -> None:
    patterns = bars_and_stripes_bitstrings(2, 6)
    state = bars_and_stripes_state(2, 6)
    support = np.flatnonzero(np.abs(state) > 0.0)
    assert len(patterns) == 66
    assert len(set(patterns)) == 66
    assert len(support) == 66
    assert set(support) == {int(pattern, 2) for pattern in patterns}
    assert np.unique(state[support]).size == 1
    assert state[support[0]].real == pytest.approx(1 / np.sqrt(66), abs=1e-15)
    assert np.linalg.norm(state) == pytest.approx(1.0, abs=1e-15)
    # Independent logical predicate, not the implementation's enumerator.
    for pattern in patterns:
        image = np.asarray(list(pattern), dtype=int).reshape(2, 6)
        bars = all(np.unique(image[:, column]).size == 1 for column in range(6))
        stripes = all(np.unique(image[row, :]).size == 1 for row in range(2))
        assert bars or stripes
    assert "010101101010" not in patterns
    assert "000000111111" in patterns
    assert "010101010101" in patterns


def test_random_real_mps_is_deterministic_real_normal_and_canonical() -> None:
    first = random_real_mps(8, max_bond=16, seed=123)
    second = random_real_mps(8, max_bond=16, seed=123)
    assert first.bond_dimensions == (2, 4, 8, 16, 8, 4, 2)
    for first_tensor, second_tensor in zip(first.raw_tensors, second.raw_tensors, strict=True):
        assert np.array_equal(first_tensor, second_tensor)
        assert np.max(np.abs(first_tensor.imag)) == 0.0
    assert np.max(np.abs(first.state - second.state)) == 0.0
    assert np.linalg.norm(first.state) == pytest.approx(1.0, abs=2e-13)
    assert np.max(np.abs(mps_to_dense(first.canonical_mps) - first.state)) < 2e-13
    assert max(ran_isometry_residuals(first.canonical_mps)) < 2e-13
    rng = np.random.default_rng(123)
    dimensions = (1, *first.bond_dimensions, 1)
    independently_drawn = tuple(
        rng.normal(
            0.0,
            1.0,
            size=(dimensions[site], 2, dimensions[site + 1]),
        )
        for site in range(8)
    )
    for observed, expected in zip(first.raw_tensors, independently_drawn, strict=True):
        assert np.array_equal(observed.real, expected)
    raw_dense = mps_to_dense(first.raw_tensors)
    assert np.max(np.abs(phase_fix_state(raw_dense) - first.state)) < 3e-13


def test_fidelity_metrics_distinguish_paper_and_squared_infidelity() -> None:
    target = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    candidate = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.complex128)
    metrics = fidelity_metrics(target, candidate)
    assert metrics.amplitude_fidelity == pytest.approx(0.8)
    assert metrics.paper_amplitude_infidelity == pytest.approx(0.2)
    assert metrics.squared_state_fidelity == pytest.approx(0.64)
    assert metrics.squared_infidelity == pytest.approx(0.36)
    phased = fidelity_metrics(np.exp(0.8j) * target, candidate)
    assert phased.amplitude_fidelity == pytest.approx(metrics.amplitude_fidelity)


def test_fidelity_roundoff_norm_drift_never_produces_negative_infidelity() -> None:
    state = np.array([1.0 + 9e-13, 0.0, 0.0, 0.0], dtype=np.complex128)
    metrics = fidelity_metrics(state, state)
    assert metrics.amplitude_fidelity == 1.0
    assert metrics.squared_state_fidelity == 1.0
    assert metrics.paper_amplitude_infidelity == 0.0
    assert metrics.squared_infidelity == 0.0


def test_energy_metrics_and_entanglement_entropy_oracles() -> None:
    hamiltonian = tfim_hamiltonian(2, coupling_zz=1.0, field_x=0.0)
    ground = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    assert energy_expectation(ground, hamiltonian) == pytest.approx(-1.0)
    assert energy_error_per_site(ground, hamiltonian, -1.0) == 0.0
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    assert entanglement_entropy(bell, 1) == pytest.approx(np.log(2), abs=1e-14)
    with pytest.raises(ArithmeticError, match="below"):
        energy_error_per_site(ground, hamiltonian, 0.0)
    bad_hamiltonian = hamiltonian.copy()
    bad_hamiltonian.data[0] = np.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        energy_expectation(ground, bad_hamiltonian)
