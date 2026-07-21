"""Rudolph environment/polar forward sweeps with equation-(6) damping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .circuit import (
    StaircaseCircuit,
    apply_adjacent_u4,
    zero_state,
)
from .conventions import (
    COMPLEX_DTYPE,
    as_complex128,
    infer_num_qubits,
    validate_integer,
    validate_normalized_statevector,
    validate_real,
)
from .linalg import damped_unitary_update, polar_optimal_unitary

ComplexArray = NDArray[np.complex128]


def local_state_matrix(
    state: ArrayLike,
    left_site: int,
    *,
    num_qubits: int,
) -> ComplexArray:
    """Reshape a state as ``(local 00/01/10/11, all other qubits)``."""

    vector = as_complex128(state, copy=True).reshape(-1)
    num_qubits = validate_integer(num_qubits, name="num_qubits", minimum=2)
    if infer_num_qubits(vector) != num_qubits:
        raise ValueError("State size does not match num_qubits.")
    left_site = validate_integer(left_site, name="left_site", minimum=0)
    if left_site >= num_qubits - 1:
        raise IndexError("Adjacent pair is outside the register.")
    tensor = vector.reshape((2,) * num_qubits)
    leading = np.moveaxis(tensor, (left_site, left_site + 1), (0, 1))
    return np.asarray(leading.reshape(4, -1), dtype=COMPLEX_DTYPE)


def build_suffix_states(
    target_state: ArrayLike,
    circuit: StaircaseCircuit,
) -> tuple[ComplexArray, ...]:
    """Return suffix[k] with future gates k..M-1 removed by adjoints."""

    if not isinstance(circuit, StaircaseCircuit):
        raise TypeError("circuit must be StaircaseCircuit.")
    target = validate_normalized_statevector(target_state)
    if infer_num_qubits(target) != circuit.num_qubits:
        raise ValueError("Target width differs from circuit width.")
    gates = circuit.flat_gates
    suffixes: list[ComplexArray | None] = [None] * (len(gates) + 1)
    suffixes[-1] = target
    running = target
    for index in range(len(gates) - 1, -1, -1):
        gate = gates[index]
        running = apply_adjacent_u4(
            running,
            gate.matrix.conj().T,
            gate.left_site,
            num_qubits=circuit.num_qubits,
            validate_matrix=False,
        )
        suffixes[index] = running
    if any(state is None for state in suffixes):
        raise AssertionError("Suffix construction left an empty entry.")
    return tuple(state for state in suffixes if state is not None)


def gate_environment(
    prefix_state: ArrayLike,
    suffix_state: ArrayLike,
    *,
    left_site: int,
    num_qubits: int,
) -> ComplexArray:
    """Return E such that ``Tr(E^dagger U)=<suffix|U|prefix>``."""

    prefix_local = local_state_matrix(
        prefix_state, left_site, num_qubits=num_qubits
    )
    suffix_local = local_state_matrix(
        suffix_state, left_site, num_qubits=num_qubits
    )
    return np.asarray(suffix_local @ prefix_local.conj().T, dtype=COMPLEX_DTYPE)


def environment_overlap_identity(
    environment: ArrayLike,
    gate: ArrayLike,
    prefix_state: ArrayLike,
    suffix_state: ArrayLike,
    *,
    left_site: int,
    num_qubits: int,
) -> tuple[complex, complex, float]:
    """Evaluate the environment identity through two independent paths."""

    environment_matrix = as_complex128(environment)
    gate_matrix = as_complex128(gate)
    via_environment = complex(np.vdot(environment_matrix, gate_matrix))
    local_output = apply_adjacent_u4(
        prefix_state,
        gate_matrix,
        left_site,
        num_qubits=num_qubits,
        validate_matrix=False,
    )
    via_state = complex(np.vdot(as_complex128(suffix_state), local_output))
    return via_environment, via_state, float(abs(via_environment - via_state))


@dataclass(frozen=True, slots=True)
class GateUpdateRecord:
    """Mathematical audit trail for one gate update."""

    sweep_index: int
    flat_gate_index: int
    layer_index: int
    position: int
    left_site: int
    overlap_before: complex
    overlap_after: complex
    amplitude_before: float
    amplitude_after: float
    squared_fidelity_before: float
    squared_fidelity_after: float
    amplitude_gain: float
    squared_fidelity_gain: float
    environment_identity_error_before: float
    environment_identity_error_after: float
    environment_nuclear_norm: float
    environment_rank: int
    polar_unitarity_residual: float
    updated_unitarity_residual: float
    learning_rate: float


@dataclass(frozen=True, slots=True)
class SweepRecord:
    """One complete forward Gauss-Seidel sweep."""

    sweep_index: int
    overlap_before: complex
    overlap_after: complex
    amplitude_fidelity_before: float
    amplitude_fidelity_after: float
    squared_fidelity_before: float
    squared_fidelity_after: float
    max_environment_identity_error: float
    max_unitarity_residual: float
    replay_error: float
    gate_updates: tuple[GateUpdateRecord, ...]

    @property
    def amplitude_gain(self) -> float:
        return self.amplitude_fidelity_after - self.amplitude_fidelity_before

    @property
    def squared_fidelity_gain(self) -> float:
        return self.squared_fidelity_after - self.squared_fidelity_before


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Fixed-budget optimization result and full sweep trajectory."""

    initial_circuit: StaircaseCircuit
    optimized_circuit: StaircaseCircuit
    learning_rate: float
    requested_sweeps: int
    sweeps: tuple[SweepRecord, ...]
    initial_overlap: complex
    final_overlap: complex
    initial_amplitude_fidelity: float
    final_amplitude_fidelity: float
    initial_squared_fidelity: float
    final_squared_fidelity: float

    @property
    def total_gate_updates(self) -> int:
        return sum(len(sweep.gate_updates) for sweep in self.sweeps)


def forward_sweep(
    target_state: ArrayLike,
    circuit: StaircaseCircuit,
    *,
    learning_rate: float = 0.6,
    sweep_index: int = 0,
    environment_atol: float = 1e-11,
) -> tuple[StaircaseCircuit, SweepRecord]:
    """Perform one cached forward sweep exactly following paper Algorithm 2.

    The damped principal-branch update is not universally monotone for an
    arbitrary environment.  Gains are therefore recorded, not asserted.
    """

    if not isinstance(circuit, StaircaseCircuit):
        raise TypeError("circuit must be StaircaseCircuit.")
    sweep_index = validate_integer(sweep_index, name="sweep_index", minimum=0)
    rate = validate_real(
        learning_rate, name="learning_rate", minimum=0.0, maximum=1.0
    )
    environment_atol = validate_real(
        environment_atol, name="environment_atol", minimum=0.0
    )
    target = validate_normalized_statevector(target_state)
    if infer_num_qubits(target) != circuit.num_qubits:
        raise ValueError("Target width differs from circuit width.")

    old_gates = circuit.flat_gates
    suffixes = build_suffix_states(target, circuit)
    initial_prepared = circuit.prepare()
    overlap_before = complex(np.vdot(target, initial_prepared))
    prefix = zero_state(circuit.num_qubits)
    updated_matrices: list[ComplexArray] = []
    records: list[GateUpdateRecord] = []

    for flat_index, old_gate in enumerate(old_gates):
        suffix = suffixes[flat_index + 1]
        environment = gate_environment(
            prefix,
            suffix,
            left_site=old_gate.left_site,
            num_qubits=circuit.num_qubits,
        )
        env_before, direct_before, error_before = environment_overlap_identity(
            environment,
            old_gate.matrix,
            prefix,
            suffix,
            left_site=old_gate.left_site,
            num_qubits=circuit.num_qubits,
        )
        if error_before > environment_atol * max(1.0, abs(direct_before)):
            raise ArithmeticError(
                f"Environment identity failed before gate {flat_index}: {error_before:.3e}."
            )

        polar = polar_optimal_unitary(environment)
        damped = damped_unitary_update(
            old_gate.matrix,
            polar.unitary,
            learning_rate=rate,
        )
        env_after, direct_after, error_after = environment_overlap_identity(
            environment,
            damped.unitary,
            prefix,
            suffix,
            left_site=old_gate.left_site,
            num_qubits=circuit.num_qubits,
        )
        if error_after > environment_atol * max(1.0, abs(direct_after)):
            raise ArithmeticError(
                f"Environment identity failed after gate {flat_index}: {error_after:.3e}."
            )

        prefix = apply_adjacent_u4(
            prefix,
            damped.unitary,
            old_gate.left_site,
            num_qubits=circuit.num_qubits,
            validate_matrix=False,
        )
        updated_matrices.append(damped.unitary)
        amplitude_before = float(abs(direct_before))
        amplitude_after = float(abs(direct_after))
        records.append(
            GateUpdateRecord(
                sweep_index=sweep_index,
                flat_gate_index=flat_index,
                layer_index=old_gate.layer_index,
                position=old_gate.position,
                left_site=old_gate.left_site,
                overlap_before=env_before,
                overlap_after=env_after,
                amplitude_before=amplitude_before,
                amplitude_after=amplitude_after,
                squared_fidelity_before=amplitude_before**2,
                squared_fidelity_after=amplitude_after**2,
                amplitude_gain=amplitude_after - amplitude_before,
                squared_fidelity_gain=amplitude_after**2 - amplitude_before**2,
                environment_identity_error_before=error_before,
                environment_identity_error_after=error_after,
                environment_nuclear_norm=polar.nuclear_norm,
                environment_rank=polar.numerical_rank,
                polar_unitarity_residual=polar.unitarity_residual,
                updated_unitarity_residual=damped.unitarity_residual,
                learning_rate=rate,
            )
        )

    optimized = circuit.with_flat_matrices(updated_matrices)
    overlap_after = complex(np.vdot(target, prefix))
    replayed = optimized.prepare()
    replay_error = float(np.max(np.abs(prefix - replayed)))
    if replay_error > environment_atol:
        raise ArithmeticError(f"End-of-sweep cached prefix replay failed: {replay_error:.3e}.")
    direct_final_overlap = complex(np.vdot(target, replayed))
    if abs(overlap_after - direct_final_overlap) > environment_atol:
        raise ArithmeticError("End-of-sweep overlap replay failed.")

    max_environment_error = max(
        (
            max(record.environment_identity_error_before, record.environment_identity_error_after)
            for record in records
        ),
        default=0.0,
    )
    max_unitarity = max(
        (record.updated_unitarity_residual for record in records), default=0.0
    )
    record = SweepRecord(
        sweep_index=sweep_index,
        overlap_before=overlap_before,
        overlap_after=overlap_after,
        amplitude_fidelity_before=float(abs(overlap_before)),
        amplitude_fidelity_after=float(abs(overlap_after)),
        squared_fidelity_before=float(abs(overlap_before) ** 2),
        squared_fidelity_after=float(abs(overlap_after) ** 2),
        max_environment_identity_error=max_environment_error,
        max_unitarity_residual=max_unitarity,
        replay_error=replay_error,
        gate_updates=tuple(records),
    )
    return optimized, record


def optimize_all(
    target_state: ArrayLike,
    initial_circuit: StaircaseCircuit,
    *,
    sweeps: int,
    learning_rate: float = 0.6,
    environment_atol: float = 1e-11,
) -> OptimizationResult:
    """Run an exact fixed number of forward sweeps; no convergence early stop."""

    if not isinstance(initial_circuit, StaircaseCircuit):
        raise TypeError("initial_circuit must be StaircaseCircuit.")
    sweeps = validate_integer(sweeps, name="sweeps", minimum=0)
    rate = validate_real(
        learning_rate, name="learning_rate", minimum=0.0, maximum=1.0
    )
    target = validate_normalized_statevector(target_state)
    initial_overlap = complex(np.vdot(target, initial_circuit.prepare()))
    circuit = initial_circuit
    records: list[SweepRecord] = []
    for sweep_index in range(sweeps):
        circuit, record = forward_sweep(
            target,
            circuit,
            learning_rate=rate,
            sweep_index=sweep_index,
            environment_atol=environment_atol,
        )
        records.append(record)
    final_overlap = complex(np.vdot(target, circuit.prepare()))
    return OptimizationResult(
        initial_circuit=initial_circuit,
        optimized_circuit=circuit,
        learning_rate=rate,
        requested_sweeps=sweeps,
        sweeps=tuple(records),
        initial_overlap=initial_overlap,
        final_overlap=final_overlap,
        initial_amplitude_fidelity=float(abs(initial_overlap)),
        final_amplitude_fidelity=float(abs(final_overlap)),
        initial_squared_fidelity=float(abs(initial_overlap) ** 2),
        final_squared_fidelity=float(abs(final_overlap) ** 2),
    )


__all__ = [
    "GateUpdateRecord",
    "OptimizationResult",
    "SweepRecord",
    "build_suffix_states",
    "environment_overlap_identity",
    "forward_sweep",
    "gate_environment",
    "local_state_matrix",
    "optimize_all",
]
