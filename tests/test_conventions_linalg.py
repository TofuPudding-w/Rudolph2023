from __future__ import annotations

import numpy as np
import pytest

from rudolph2023.conventions import (
    as_complex128,
    infer_num_qubits,
    normalized_statevector,
    phase_aligned_state,
    phase_fix_state,
    state_distance_up_to_global_phase,
    validate_integer,
    validate_normalized_statevector,
    validate_real,
)
from rudolph2023.linalg import (
    damped_unitary_update,
    deterministic_svd,
    fractional_unitary_power,
    polar_optimal_unitary,
    svd_numerical_rank,
    unitarity_residual,
    validate_unitary,
)


def random_unitary(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q_matrix, r_matrix = np.linalg.qr(matrix)
    phases = np.diag(r_matrix)
    phases = np.where(np.abs(phases) > 0, phases / np.abs(phases), 1.0)
    return q_matrix @ np.diag(np.conj(phases))


def test_scalar_and_state_validation() -> None:
    assert validate_integer(np.int64(3), name="n", minimum=2) == 3
    assert validate_real(np.float64(0.6), name="r", minimum=0, maximum=1) == 0.6
    with pytest.raises(TypeError):
        validate_integer(True, name="n", minimum=0)
    with pytest.raises(ValueError):
        validate_real(np.inf, name="x")
    with pytest.raises(ValueError):
        infer_num_qubits(np.ones(3))
    array = as_complex128([1.0, 1j])
    assert array.dtype == np.complex128
    state = normalized_statevector([1.0, 1.0])
    assert np.linalg.norm(state) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        validate_normalized_statevector([1.0, 1.0])


def test_global_phase_utilities_are_physical_and_deterministic() -> None:
    reference = normalized_statevector([1.0, 2.0j, -0.5, 0.0])
    candidate = np.exp(0.73j) * reference
    assert state_distance_up_to_global_phase(reference, candidate) < 1e-14
    assert np.max(np.abs(phase_aligned_state(reference, candidate) - reference)) < 1e-14
    fixed = phase_fix_state(candidate)
    pivot = int(np.argmax(np.abs(fixed)))
    assert fixed[pivot].imag == 0.0
    assert fixed[pivot].real > 0.0
    assert np.max(np.abs(fixed - phase_fix_state(reference))) < 1e-14


def test_validate_unitary_and_residual() -> None:
    unitary = random_unitary(4, 1)
    assert unitarity_residual(unitary) < 1e-12
    checked = validate_unitary(unitary, dimension=4)
    assert not checked.flags.writeable
    with pytest.raises(ValueError):
        validate_unitary(np.ones((4, 4)), dimension=4)
    assert np.isinf(unitarity_residual(np.ones((2, 3))))


@pytest.mark.parametrize("shape", [(7, 4), (4, 7), (4, 4)])
def test_deterministic_svd_reconstructs(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(sum(shape))
    matrix = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    u_matrix, values, vh_matrix = deterministic_svd(matrix, pivot_on="u")
    reconstructed = (u_matrix * values) @ vh_matrix
    assert np.max(np.abs(reconstructed - matrix)) < 2e-14
    for column in range(min(shape)):
        pivot = int(np.argmax(np.abs(u_matrix[:, column])))
        assert abs(u_matrix[pivot, column].imag) < 2e-15


def test_svd_numerical_rank_discards_only_machine_null_space() -> None:
    values = np.array([1.0, 1e-3, 0.0])
    assert svd_numerical_rank(values, (3, 3)) == 2
    assert svd_numerical_rank(np.zeros(3), (3, 3)) == 1
    with pytest.raises(ValueError):
        svd_numerical_rank([], (0, 0))


@pytest.mark.parametrize("rank", [0, 1, 2, 4])
def test_polar_optimum_handles_rank_deficiency(rank: int) -> None:
    left = random_unitary(4, 10)
    right = random_unitary(4, 11)
    singular = np.zeros(4)
    singular[:rank] = np.linspace(2.0, 0.5, rank) if rank else []
    environment = left @ np.diag(singular) @ right.conj().T
    result = polar_optimal_unitary(environment)
    assert unitarity_residual(result.unitary) < 1e-12
    assert result.nuclear_norm == pytest.approx(float(np.sum(singular)), abs=1e-12)
    assert result.achieved_overlap.real == pytest.approx(result.nuclear_norm, abs=1e-12)
    assert abs(result.achieved_overlap.imag) < 1e-12
    assert result.numerical_rank == rank


def test_polar_is_at_least_as_good_as_random_unitaries() -> None:
    rng = np.random.default_rng(22)
    environment = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    result = polar_optimal_unitary(environment)
    random_objectives = [
        np.vdot(environment, random_unitary(4, seed)).real for seed in range(30)
    ]
    assert result.achieved_overlap.real >= max(random_objectives) - 1e-12


def test_polar_rank_rtol_is_diagnostic_and_never_discards_objective() -> None:
    left = random_unitary(4, 901)
    right = random_unitary(4, 902)
    singular = np.array([1.0, 1e-3, 1e-4, 0.0])
    environment = left @ np.diag(singular) @ right.conj().T
    result = polar_optimal_unitary(environment, rank_rtol=1e-2)
    assert result.numerical_rank == 1
    assert result.optimization_rank == 3
    assert result.achieved_overlap.real == pytest.approx(np.sum(singular), abs=2e-12)
    assert unitarity_residual(result.unitary) < 1e-12


def test_fractional_unitary_power_endpoints_diagonal_and_minus_one_branch() -> None:
    phases = np.array([0.2, -0.7, np.pi, 1.4])
    unitary = np.diag(np.exp(1j * phases))
    assert np.array_equal(fractional_unitary_power(unitary, 0.0), np.eye(4))
    assert np.array_equal(fractional_unitary_power(unitary, 1.0), unitary)
    square_root = fractional_unitary_power(unitary, 0.5)
    expected = np.diag(np.exp(0.5j * phases))
    assert np.max(np.abs(square_root - expected)) < 1e-13
    minus_identity_root = fractional_unitary_power(-np.eye(4), 0.5)
    assert np.max(np.abs(minus_identity_root - 1j * np.eye(4))) < 1e-13
    assert unitarity_residual(square_root) < 1e-12


def test_fractional_power_respects_both_sides_of_negative_axis_cut() -> None:
    epsilon = 5e-15
    phases = np.array([-np.pi + epsilon, np.pi - epsilon, -0.4, 0.7])
    unitary = np.diag(np.exp(1j * phases))
    observed = fractional_unitary_power(unitary, 0.6)
    expected = np.diag(np.exp(0.6j * phases))
    assert np.max(np.abs(observed - expected)) < 5e-13
    assert np.angle(observed[0, 0]) < 0.0
    assert np.angle(observed[1, 1]) > 0.0


@pytest.mark.parametrize("seed", range(10))
def test_fractional_power_random_unitary_matches_independent_eigen_oracle(seed: int) -> None:
    vectors = random_unitary(4, seed)
    phases = np.array([-2.2, -0.5, 0.8, 2.1])
    unitary = vectors @ np.diag(np.exp(1j * phases)) @ vectors.conj().T
    observed = fractional_unitary_power(unitary, 0.6)
    expected = vectors @ np.diag(np.exp(0.6j * phases)) @ vectors.conj().T
    assert np.max(np.abs(observed - expected)) < 5e-13


def test_damped_update_is_exact_equation_6_and_has_correct_endpoints() -> None:
    current = random_unitary(4, 30)
    optimal = random_unitary(4, 31)
    update = damped_unitary_update(current, optimal, learning_rate=0.6)
    expected = current @ fractional_unitary_power(current.conj().T @ optimal, 0.6)
    assert np.max(np.abs(update.unitary - expected)) < 1e-14
    assert np.max(
        np.abs(damped_unitary_update(current, optimal, learning_rate=0.0).unitary - current)
    ) < 1e-14
    assert np.max(
        np.abs(damped_unitary_update(current, optimal, learning_rate=1.0).unitary - optimal)
    ) < 2e-13
    assert update.unitarity_residual < 1e-11
