"""Verified orchestration of the paper's main ``Iter[D_i O_all]`` protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .analytic import AnalyticStepResult, analytic_step
from .circuit import StaircaseCircuit
from .conventions import (
    infer_num_qubits,
    phase_fix_state,
    validate_integer,
    validate_normalized_statevector,
    validate_real,
)
from .metrics import FidelityMetrics, fidelity_metrics
from .optimizer import OptimizationResult, optimize_all

ComplexArray = NDArray[np.complex128]


def expected_cumulative_gate_updates(
    num_qubits: int,
    depth: int,
    sweeps_per_depth: int,
) -> int:
    """Return ``(N-1) T sum(k,1..K)`` for sequential growth."""

    num_qubits = validate_integer(num_qubits, name="num_qubits", minimum=2)
    depth = validate_integer(depth, name="depth", minimum=0)
    sweeps_per_depth = validate_integer(
        sweeps_per_depth, name="sweeps_per_depth", minimum=0
    )
    return (num_qubits - 1) * sweeps_per_depth * depth * (depth + 1) // 2


@dataclass(frozen=True, slots=True)
class DepthRecord:
    """One grow-then-optimize iteration and its cross-checked identities."""

    depth: int
    residual_before: ComplexArray
    analytic: AnalyticStepResult
    candidate_circuit: StaircaseCircuit
    candidate_metrics: FidelityMetrics
    candidate_overlap_identity_error: float
    optimization: OptimizationResult
    optimized_circuit: StaircaseCircuit
    optimized_metrics: FidelityMetrics
    residual_after: ComplexArray
    residual_overlap_identity_error: float
    gate_updates_this_depth: int
    cumulative_gate_updates: int


@dataclass(frozen=True, slots=True)
class IterDiOAllResult:
    """Complete main-protocol result with target gauge and depth trajectory."""

    input_target: ComplexArray
    canonical_target: ComplexArray
    final_circuit: StaircaseCircuit
    final_prepared_state: ComplexArray
    final_metrics: FidelityMetrics
    max_depth: int
    sweeps_per_depth: int
    learning_rate: float
    depth_records: tuple[DepthRecord, ...]

    @property
    def total_gate_updates(self) -> int:
        return sum(record.gate_updates_this_depth for record in self.depth_records)


def run_iter_di_o_all(
    target_state: ArrayLike,
    *,
    max_depth: int,
    sweeps_per_depth: int,
    learning_rate: float = 0.6,
    validation_atol: float = 1e-10,
) -> IterDiOAllResult:
    """Run the main Rudolph protocol with exact dense residuals.

    At every depth the new analytic layer is prepended, all existing gates are
    optimized for a fixed sweep budget, and the residual is recomputed from
    the optimized complete circuit before the next ``D_i`` extraction.
    """

    max_depth = validate_integer(max_depth, name="max_depth", minimum=0)
    sweeps_per_depth = validate_integer(
        sweeps_per_depth, name="sweeps_per_depth", minimum=0
    )
    rate = validate_real(
        learning_rate, name="learning_rate", minimum=0.0, maximum=1.0
    )
    validation_atol = validate_real(
        validation_atol, name="validation_atol", minimum=0.0
    )
    if validation_atol == 0.0:
        raise ValueError("validation_atol must be strictly positive.")
    input_target = validate_normalized_statevector(target_state)
    canonical_target = phase_fix_state(input_target)
    num_qubits = infer_num_qubits(canonical_target)
    if num_qubits < 2:
        raise ValueError("Rudolph pure-U(4) protocol requires N>=2.")

    circuit = StaircaseCircuit.empty(num_qubits)
    records: list[DepthRecord] = []
    cumulative_updates = 0

    for depth in range(1, max_depth + 1):
        # This must be recomputed after O_all changed every existing gate.
        residual_before = circuit.adjoint_apply(canonical_target)
        analytic = analytic_step(residual_before, validation_atol=validation_atol)
        candidate = circuit.prepend_input_layer(analytic.layer)
        candidate_prepared = candidate.prepare()
        candidate_metrics = fidelity_metrics(canonical_target, candidate_prepared)

        # <target|C L|0> = <C^dagger target|L|0>
        expected_candidate_overlap = complex(
            np.vdot(residual_before, analytic.compressed_state)
        )
        candidate_identity_error = float(
            abs(candidate_metrics.overlap - expected_candidate_overlap)
        )
        if candidate_identity_error > validation_atol:
            raise ArithmeticError(
                "Prepending the analytic layer violated the circuit/residual overlap identity."
            )
        if candidate.depth != depth or candidate.num_gates != depth * (num_qubits - 1):
            raise ArithmeticError("Sequential growth produced an invalid gate count.")

        optimization = optimize_all(
            canonical_target,
            candidate,
            sweeps=sweeps_per_depth,
            learning_rate=rate,
            environment_atol=max(1e-12, validation_atol),
        )
        circuit = optimization.optimized_circuit
        prepared = circuit.prepare()
        optimized_metrics = fidelity_metrics(canonical_target, prepared)
        residual_after = circuit.adjoint_apply(canonical_target)
        # residual[0]=<prepared|target>=conj(<target|prepared>)
        residual_identity_error = float(
            abs(np.conj(residual_after[0]) - optimized_metrics.overlap)
        )
        if residual_identity_error > validation_atol:
            raise ArithmeticError("Optimized residual overlap identity failed.")
        if abs(np.linalg.norm(residual_after) - 1.0) > validation_atol:
            raise ArithmeticError("Optimized circuit inverse did not conserve target norm.")

        updates_this_depth = optimization.total_gate_updates
        expected_this_depth = sweeps_per_depth * depth * (num_qubits - 1)
        if updates_this_depth != expected_this_depth:
            raise ArithmeticError(
                f"Update count mismatch: expected {expected_this_depth}, got {updates_this_depth}."
            )
        cumulative_updates += updates_this_depth
        expected_cumulative = expected_cumulative_gate_updates(
            num_qubits, depth, sweeps_per_depth
        )
        if cumulative_updates != expected_cumulative:
            raise ArithmeticError("Cumulative update count identity failed.")

        records.append(
            DepthRecord(
                depth=depth,
                residual_before=residual_before,
                analytic=analytic,
                candidate_circuit=candidate,
                candidate_metrics=candidate_metrics,
                candidate_overlap_identity_error=candidate_identity_error,
                optimization=optimization,
                optimized_circuit=circuit,
                optimized_metrics=optimized_metrics,
                residual_after=residual_after,
                residual_overlap_identity_error=residual_identity_error,
                gate_updates_this_depth=updates_this_depth,
                cumulative_gate_updates=cumulative_updates,
            )
        )

    final_prepared = circuit.prepare()
    final_metrics = fidelity_metrics(canonical_target, final_prepared)
    expected_total = expected_cumulative_gate_updates(
        num_qubits, max_depth, sweeps_per_depth
    )
    if cumulative_updates != expected_total:
        raise ArithmeticError("Final update count identity failed.")
    if not math.isclose(
        float(np.linalg.norm(final_prepared)),
        1.0,
        rel_tol=0.0,
        abs_tol=validation_atol,
    ):
        raise ArithmeticError("Final circuit state is not normalized.")
    return IterDiOAllResult(
        input_target=input_target,
        canonical_target=canonical_target,
        final_circuit=circuit,
        final_prepared_state=final_prepared,
        final_metrics=final_metrics,
        max_depth=max_depth,
        sweeps_per_depth=sweeps_per_depth,
        learning_rate=rate,
        depth_records=tuple(records),
    )


__all__ = [
    "DepthRecord",
    "IterDiOAllResult",
    "expected_cumulative_gate_updates",
    "run_iter_di_o_all",
]
