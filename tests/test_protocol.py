from __future__ import annotations

import numpy as np
import pytest

from rudolph2023.circuit import StaircaseLayer, embed_adjacent_u4
from rudolph2023.conventions import normalized_statevector
from rudolph2023.protocol import (
    expected_cumulative_gate_updates,
    run_iter_di_o_all,
)
from rudolph2023.targets import random_real_mps


def test_update_count_paper_oracle() -> None:
    assert expected_cumulative_gate_updates(12, 8, 10) == 3960
    assert expected_cumulative_gate_updates(12, 8, 100) == 39600
    assert expected_cumulative_gate_updates(12, 8, 10) != 4320


def test_depth_zero_is_identity_baseline() -> None:
    target = random_real_mps(4, max_bond=4, seed=0).state
    result = run_iter_di_o_all(target, max_depth=0, sweeps_per_depth=3)
    assert result.final_circuit.depth == 0
    assert result.total_gate_updates == 0
    assert result.depth_records == ()
    assert np.array_equal(result.final_prepared_state, np.eye(1, 16, 0).reshape(-1))
    with pytest.raises(ValueError):
        run_iter_di_o_all(2.0 * target, max_depth=0, sweeps_per_depth=0)
    with pytest.raises(ValueError, match="strictly positive"):
        run_iter_di_o_all(
            target,
            max_depth=0,
            sweeps_per_depth=0,
            validation_atol=0.0,
        )


def test_chi2_target_is_exact_at_depth_one_without_optimization() -> None:
    rng = np.random.default_rng(1)
    gates = []
    for _site in range(4):
        matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        gates.append(np.linalg.qr(matrix)[0])
    target = StaircaseLayer(5, tuple(gates)).prepare()
    result = run_iter_di_o_all(target, max_depth=1, sweeps_per_depth=0)
    assert result.final_metrics.amplitude_fidelity == pytest.approx(1.0, abs=2e-12)
    assert result.final_metrics.paper_amplitude_infidelity < 2e-12


def test_each_iteration_recomputes_residual_after_optimization_and_prepends_layer() -> None:
    target = random_real_mps(5, max_bond=8, seed=2).state
    result = run_iter_di_o_all(
        target, max_depth=3, sweeps_per_depth=2, learning_rate=0.6
    )
    assert result.total_gate_updates == expected_cumulative_gate_updates(5, 3, 2)
    previous_circuit = None
    for record in result.depth_records:
        assert record.optimized_circuit.depth == record.depth
        assert record.optimized_circuit.num_gates == record.depth * 4
        assert record.candidate_overlap_identity_error < 1e-10
        assert record.residual_overlap_identity_error < 1e-10
        assert record.cumulative_gate_updates == expected_cumulative_gate_updates(
            5, record.depth, 2
        )
        if previous_circuit is not None:
            expected_residual = previous_circuit.adjoint_apply(result.canonical_target)
            assert np.max(np.abs(record.residual_before - expected_residual)) < 1e-12
            # Candidate executes the newly extracted layer first, then all old layers.
            for candidate_layer, old_layer in zip(
                record.candidate_circuit.layers[1:],
                previous_circuit.layers,
                strict=True,
            ):
                assert np.max(
                    np.abs(
                        np.asarray(candidate_layer.matrices)
                        - np.asarray(old_layer.matrices)
                    )
                ) < 1e-14
            previous_record = result.depth_records[record.depth - 2]
            assert np.max(
                np.abs(previous_record.residual_after - record.residual_before)
            ) < 2e-12
        previous_circuit = record.optimized_circuit


def test_multilayer_order_matches_independent_full_matrix_oracle() -> None:
    target = random_real_mps(4, max_bond=8, seed=711).state
    result = run_iter_di_o_all(target, max_depth=3, sweeps_per_depth=0)
    full = np.eye(16, dtype=np.complex128)
    for layer in result.final_circuit.layers:
        for left_site, gate in enumerate(layer.matrices):
            full = embed_adjacent_u4(gate, left_site, 4) @ full
    zero = np.zeros(16, dtype=np.complex128)
    zero[0] = 1.0
    assert np.max(np.abs(full @ zero - result.final_prepared_state)) < 4e-13
    assert np.max(
        np.abs(full.conj().T @ result.canonical_target - result.depth_records[-1].residual_after)
    ) < 5e-13


def test_protocol_canonical_target_phase_removes_input_global_phase_dependence() -> None:
    target = random_real_mps(4, max_bond=4, seed=3).state
    first = run_iter_di_o_all(target, max_depth=2, sweeps_per_depth=2)
    second = run_iter_di_o_all(
        np.exp(0.73j) * target, max_depth=2, sweeps_per_depth=2
    )
    assert np.max(np.abs(first.canonical_target - second.canonical_target)) < 2e-15
    # Rank-deficient SVD/completion gauges can change individual gate matrices
    # while leaving the prepared physical state invariant.
    assert abs(
        np.vdot(first.final_prepared_state, second.final_prepared_state)
    ) ** 2 > 1.0 - 1e-12
    assert np.max(
        np.abs(first.final_prepared_state - second.final_prepared_state)
    ) < 2e-12
    assert first.final_metrics.amplitude_fidelity == pytest.approx(
        second.final_metrics.amplitude_fidelity, abs=2e-13
    )


@pytest.mark.parametrize("seed", range(5))
def test_protocol_multiseed_end_to_end_invariants(seed: int) -> None:
    rng = np.random.default_rng(seed)
    target = normalized_statevector(
        rng.normal(size=16) + 1j * rng.normal(size=16)
    )
    result = run_iter_di_o_all(target, max_depth=2, sweeps_per_depth=1)
    assert result.final_circuit.gate_array().shape == (2, 3, 4, 4)
    assert np.linalg.norm(result.final_prepared_state) == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= result.final_metrics.amplitude_fidelity <= 1.0 + 1e-12
    assert result.total_gate_updates == 9
