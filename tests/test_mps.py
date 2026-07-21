from __future__ import annotations

import numpy as np
import pytest

from rudolph2023.conventions import normalized_statevector, phase_aligned_state
from rudolph2023.mps import (
    MPS,
    canonicalize_to_ran,
    dense_to_mps,
    mps_inner_product,
    mps_norm,
    mps_to_dense,
    normalize_mps,
    open_boundary_bond_dimensions,
    ran_isometry_residuals,
    truncate_mps,
    validate_mps,
)


def random_state(num_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalized_statevector(
        rng.normal(size=1 << num_qubits) + 1j * rng.normal(size=1 << num_qubits)
    )


def test_mps_validation_rejects_invalid_boundaries_bonds_and_physical_dimension() -> None:
    with pytest.raises(ValueError):
        validate_mps([])
    with pytest.raises(ValueError):
        MPS([np.ones((2, 2, 1))])
    with pytest.raises(ValueError):
        MPS([np.ones((1, 3, 1))])
    with pytest.raises(ValueError):
        MPS([np.ones((1, 2, 2)), np.ones((3, 2, 1))])


def test_mps_norm_pipeline_rejects_finite_values_that_overflow_contraction() -> None:
    huge = MPS([np.array([[[1e308], [1e308]]], dtype=np.complex128)])
    with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
        mps_inner_product(huge, huge)
    with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
        mps_norm(huge)
    with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
        normalize_mps(huge)
    with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
        canonicalize_to_ran(huge)
    with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
        truncate_mps(huge, max_bond=1)


def test_mps_norm_pipeline_rejects_underflowed_zero_norm() -> None:
    tiny = MPS([np.array([[[1e-300], [0.0]]], dtype=np.complex128)])
    assert mps_norm(tiny) == 0.0
    with pytest.raises(ValueError, match="zero MPS"):
        normalize_mps(tiny)


@pytest.mark.parametrize("num_qubits", range(2, 8))
def test_dense_mps_exact_round_trip(num_qubits: int) -> None:
    state = random_state(num_qubits, 100 + num_qubits)
    mps = dense_to_mps(state)
    assert np.max(np.abs(mps_to_dense(mps) - state)) < 2e-14
    assert mps_norm(mps) == pytest.approx(1.0, abs=2e-14)
    assert mps.norm() == pytest.approx(1.0, abs=2e-14)


def test_mps_inner_product_matches_numpy_vdot() -> None:
    first = random_state(5, 1)
    second = random_state(5, 2)
    first_mps = dense_to_mps(first)
    second_mps = dense_to_mps(second)
    assert len(first_mps) == 5
    assert sum(1 for _ in first_mps) == 5
    assert first_mps[0].shape[0] == 1
    assert first_mps.copy() is not first_mps
    assert mps_inner_product(first_mps, second_mps) == pytest.approx(
        np.vdot(first, second), abs=2e-14
    )


def test_normalize_and_ran_canonicalization_preserve_state() -> None:
    state = 3.7 * random_state(6, 7)
    mps = dense_to_mps(state)
    normalized = normalize_mps(mps)
    canonical = canonicalize_to_ran(normalized)
    assert np.max(np.abs(mps_to_dense(normalized) - state / np.linalg.norm(state))) < 3e-14
    assert np.max(np.abs(mps_to_dense(canonical) - mps_to_dense(normalized))) < 5e-14
    assert max(ran_isometry_residuals(canonical)) < 3e-14


def test_chi2_truncation_has_bond_cap_and_self_consistent_metrics() -> None:
    state = random_state(8, 8)
    source = dense_to_mps(state)
    result = truncate_mps(source, max_bond=2)
    assert result.mps.max_bond_dimension <= 2
    dense = mps_to_dense(result.mps)
    amplitude = abs(np.vdot(state, dense))
    assert amplitude == pytest.approx(result.normalized_overlap, abs=2e-13)
    assert amplitude**2 == pytest.approx(result.squared_fidelity, abs=3e-13)
    assert result.output_norm == pytest.approx(1.0, abs=2e-13)
    assert max(result.isometry_residuals) < 2e-13
    assert all(weight >= 0.0 for weight in result.discarded_weights)
    assert result.total_discarded_weight == pytest.approx(
        sum(result.discarded_weights), abs=0.0
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_truncation_of_unnormalized_input_reports_consistent_norms(
    normalize: bool,
) -> None:
    dense = 3.25 * random_state(6, 808)
    source = dense_to_mps(dense)
    result = truncate_mps(source, max_bond=2, normalize=normalize)
    output = mps_to_dense(result.mps)
    expected_amplitude = abs(np.vdot(dense, output)) / (
        np.linalg.norm(dense) * np.linalg.norm(output)
    )
    assert result.input_norm == pytest.approx(3.25, abs=5e-13)
    assert result.output_norm == pytest.approx(np.linalg.norm(output), abs=5e-13)
    assert result.normalized_overlap == pytest.approx(expected_amplitude, abs=5e-13)
    if normalize:
        assert result.output_norm == pytest.approx(1.0, abs=5e-13)


def test_single_bond_truncation_matches_eckart_young_oracle() -> None:
    state = random_state(4, 19)
    result = truncate_mps(dense_to_mps(state), max_bond=2)
    # At the central 4x4 bipartition the optimal rank-2 normalized state is explicit.
    u_matrix, singular, vh_matrix = np.linalg.svd(state.reshape(4, 4), full_matrices=False)
    rank_two = (u_matrix[:, :2] * singular[:2]) @ vh_matrix[:2]
    rank_two = normalized_statevector(rank_two.reshape(-1))
    # Multi-bond TT-SVD also restricts the edge bonds, which are already <=2.
    observed = phase_aligned_state(rank_two, result.mps.to_dense())
    assert np.linalg.norm(observed - rank_two) < 1e-12


def test_open_boundary_bond_profile() -> None:
    assert open_boundary_bond_dimensions(12, 64) == (
        2,
        4,
        8,
        16,
        32,
        64,
        32,
        16,
        8,
        4,
        2,
    )
    with pytest.raises(ValueError):
        open_boundary_bond_dimensions(1, 2)
