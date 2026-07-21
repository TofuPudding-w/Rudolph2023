"""Convention-safe Qiskit adapter for the Rudolph pure-U(4) staircase.

The paper reproduction itself deliberately has no Qiskit dependency.  Every
Qiskit import in this module is therefore lazy, so importing
``rudolph2023.qiskit_adapter`` remains safe in the NumPy/SciPy-only
environment.

The ordering bridge is

``project site i (q0 most significant) -> Qiskit qubit N - 1 - i``.

For a project-local U(4) whose basis is ``|left,right>`` this additionally
requires passing Qiskit qargs as ``[mapped_right, mapped_left]``.  With both
steps, Qiskit's statevector array has exactly the same computational-basis
ordering as the project's q0-MSB dense statevector; no bit reversal is needed.
"""

from __future__ import annotations

import importlib
import importlib.util
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .circuit import StaircaseCircuit

ComplexArray = NDArray[np.complex128]
NATIVE_BASIS = frozenset({"rz", "sx", "x", "cx"})


class QiskitUnavailableError(ImportError):
    """Raised when an explicitly requested optional Qiskit operation is unavailable."""


@dataclass(frozen=True, slots=True)
class CompilationDiagnostics:
    """Ideal logical replay and exact native-synthesis diagnostics.

    All fidelities are normalized squared pure-state fidelities.  A phase-
    aligned error is the maximum absolute statevector difference after
    removing the candidate-over-reference global phase.
    """

    num_qubits: int
    depth: int
    num_u4_gates: int
    logical_replay_fidelity: float
    native_replay_fidelity: float
    compile_fidelity: float
    logical_phase_over_dense: float
    native_phase_over_logical: float
    logical_phase_aligned_max_abs_error: float
    native_phase_aligned_max_abs_error: float
    native_operation_counts: tuple[tuple[str, int], ...]
    native_basis_only: bool


def qiskit_available() -> bool:
    """Return whether a top-level :mod:`qiskit` package is discoverable."""

    try:
        return importlib.util.find_spec("qiskit") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def require_qiskit() -> Any:
    """Import Qiskit or raise one actionable optional-dependency error."""

    try:
        return importlib.import_module("qiskit")
    except (ImportError, ModuleNotFoundError) as exc:
        raise QiskitUnavailableError(
            "Qiskit is required for circuit translation and native synthesis; "
            "install the project's optional noise dependencies."
        ) from exc


def _require_staircase(circuit: StaircaseCircuit) -> StaircaseCircuit:
    if not isinstance(circuit, StaircaseCircuit):
        raise TypeError(
            "circuit must be a rudolph2023.circuit.StaircaseCircuit, "
            f"got {type(circuit).__name__}."
        )
    return circuit


def _validated_num_qubits(num_qubits: int) -> int:
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, numbers.Integral):
        raise TypeError("num_qubits must be an integer.")
    result = int(num_qubits)
    if result < 2:
        raise ValueError("num_qubits must be at least two.")
    return result


def _validated_site(site: int, *, num_qubits: int) -> int:
    if isinstance(site, bool) or not isinstance(site, numbers.Integral):
        raise TypeError("site must be an integer.")
    result = int(site)
    if result < 0 or result >= num_qubits:
        raise ValueError(f"site={result} is outside an N={num_qubits} register.")
    return result


def project_site_to_qiskit(site: int, *, num_qubits: int) -> int:
    """Map one q0-MSB project site to its Qiskit little-endian qubit index."""

    width = _validated_num_qubits(num_qubits)
    validated = _validated_site(site, num_qubits=width)
    return width - 1 - validated


def project_sites_to_qiskit_qargs(
    sites: tuple[int, int],
    *,
    num_qubits: int,
) -> tuple[int, int]:
    """Map one ordered adjacent project pair while preserving its U(4) basis.

    Qiskit treats its first qarg as the least-significant local subsystem.
    The project lists the most-significant local site first, hence the
    intentional reversal after mapping the two project sites.
    """

    width = _validated_num_qubits(num_qubits)
    if not isinstance(sites, tuple):
        raise TypeError("sites must be a tuple containing one ordered adjacent pair.")
    if len(sites) != 2:
        raise ValueError("sites must contain exactly two sites.")
    left = _validated_site(sites[0], num_qubits=width)
    right = _validated_site(sites[1], num_qubits=width)
    if right != left + 1:
        raise ValueError("sites must be an ordered adjacent pair (left,left+1).")
    mapped = (
        project_site_to_qiskit(left, num_qubits=width),
        project_site_to_qiskit(right, num_qubits=width),
    )
    return mapped[1], mapped[0]


def to_qiskit_logical(
    circuit: StaircaseCircuit,
    *,
    insert_barriers: bool = False,
) -> Any:
    """Translate the physical U(4) execution order into exact Qiskit blocks."""

    circuit = _require_staircase(circuit)
    if not isinstance(insert_barriers, bool):
        raise TypeError("insert_barriers must be bool.")
    require_qiskit()
    from qiskit import QuantumCircuit

    translated = QuantumCircuit(circuit.num_qubits, name="rudolph2023_logical")
    for flat_index, gate in enumerate(circuit.flat_gates):
        qargs = project_sites_to_qiskit_qargs(
            (gate.left_site, gate.left_site + 1),
            num_qubits=circuit.num_qubits,
        )
        translated.unitary(
            np.asarray(gate.matrix, dtype=np.complex128),
            list(qargs),
            label=f"u4_l{gate.layer_index}_p{gate.position}",
        )
        if insert_barriers and flat_index + 1 < circuit.num_gates:
            translated.barrier()
    return translated


def remove_barriers(qiskit_circuit: Any) -> Any:
    """Return a circuit copy with barriers removed and global phase retained."""

    require_qiskit()
    from qiskit import QuantumCircuit
    from qiskit.transpiler.passes import RemoveBarriers

    if not isinstance(qiskit_circuit, QuantumCircuit):
        raise TypeError("qiskit_circuit must be a qiskit.QuantumCircuit.")
    return RemoveBarriers()(qiskit_circuit.copy())


def compile_staircase_native(
    circuit: StaircaseCircuit,
    *,
    insert_barriers: bool = False,
    remove_inserted_barriers: bool = True,
) -> Any:
    """Exactly synthesize every U(4) independently into ``rz/sx/x/cx``.

    Independent block synthesis preserves the Rudolph macro-layer resource
    boundary.  ``basis_fidelity=1`` and ``approximate=False`` forbid an
    approximate hardware-aware replacement at this stage.
    """

    circuit = _require_staircase(circuit)
    if not isinstance(insert_barriers, bool):
        raise TypeError("insert_barriers must be bool.")
    if not isinstance(remove_inserted_barriers, bool):
        raise TypeError("remove_inserted_barriers must be bool.")

    require_qiskit()
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import CXGate
    try:
        from qiskit.synthesis import TwoQubitBasisDecomposer
    except ImportError:  # Qiskit 0.45 compatibility; Qiskit 2.x uses the path above.
        from qiskit.quantum_info.synthesis import TwoQubitBasisDecomposer

    decomposer = TwoQubitBasisDecomposer(
        CXGate(),
        basis_fidelity=1.0,
        euler_basis="ZSX",
        pulse_optimize=False,
    )
    native = QuantumCircuit(circuit.num_qubits, name="rudolph2023_native")
    for flat_index, gate in enumerate(circuit.flat_gates):
        block = decomposer(
            np.asarray(gate.matrix, dtype=np.complex128),
            basis_fidelity=1.0,
            approximate=False,
        )
        qargs = project_sites_to_qiskit_qargs(
            (gate.left_site, gate.left_site + 1),
            num_qubits=circuit.num_qubits,
        )
        native.compose(block, list(qargs), inplace=True)
        if insert_barriers and flat_index + 1 < circuit.num_gates:
            native.barrier()

    if insert_barriers and remove_inserted_barriers:
        native = remove_barriers(native)
    unexpected = set(native.count_ops()) - NATIVE_BASIS - {"barrier"}
    if unexpected:
        joined = ", ".join(sorted(unexpected))
        raise RuntimeError(f"Exact synthesis emitted non-native operations: {joined}.")
    return native


def qiskit_statevector_q0_msb(qiskit_circuit: Any) -> ComplexArray:
    """Ideally execute a Qiskit circuit in the project's direct q0-MSB array order."""

    require_qiskit()
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    if not isinstance(qiskit_circuit, QuantumCircuit):
        raise TypeError("qiskit_circuit must be a qiskit.QuantumCircuit.")
    state = np.asarray(Statevector.from_instruction(qiskit_circuit).data, dtype=np.complex128)
    if state.shape != (1 << qiskit_circuit.num_qubits,):
        raise ArithmeticError("Qiskit returned an unexpected statevector shape.")
    if not np.all(np.isfinite(state)):
        raise ArithmeticError("Qiskit returned a non-finite statevector.")
    return np.array(state, dtype=np.complex128, copy=True)


def statevector_fidelity(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return normalized squared pure-state fidelity without numerical clipping."""

    reference_array = np.asarray(reference, dtype=np.complex128).reshape(-1)
    candidate_array = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "Statevectors must have the same shape; "
            f"got {reference_array.shape} and {candidate_array.shape}."
        )
    if reference_array.size == 0:
        raise ValueError("Statevectors must not be empty.")
    if not np.all(np.isfinite(reference_array)) or not np.all(np.isfinite(candidate_array)):
        raise ValueError("Statevectors must contain only finite values.")
    reference_norm_sq = float(np.vdot(reference_array, reference_array).real)
    candidate_norm_sq = float(np.vdot(candidate_array, candidate_array).real)
    if reference_norm_sq <= 0.0 or candidate_norm_sq <= 0.0:
        raise ValueError("Statevectors must have nonzero norm.")
    overlap = complex(np.vdot(reference_array, candidate_array))
    return float(abs(overlap) ** 2 / (reference_norm_sq * candidate_norm_sq))


def _phase_aligned_comparison(
    reference: ArrayLike,
    candidate: ArrayLike,
) -> tuple[float, float]:
    reference_array = np.asarray(reference, dtype=np.complex128).reshape(-1)
    candidate_array = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    # Reuse the strict validation, including equal shape and nonzero norms.
    statevector_fidelity(reference_array, candidate_array)
    reference_unit = reference_array / np.linalg.norm(reference_array)
    candidate_unit = candidate_array / np.linalg.norm(candidate_array)
    overlap = complex(np.vdot(reference_unit, candidate_unit))
    phase = float(np.angle(overlap)) if abs(overlap) > 0.0 else 0.0
    aligned = candidate_unit * np.exp(-1j * phase)
    return phase, float(np.max(np.abs(reference_unit - aligned)))


def diagnose_qiskit_compilation(
    circuit: StaircaseCircuit,
    native_circuit: Any | None = None,
) -> CompilationDiagnostics:
    """Cross-check project-dense, logical-Qiskit, and native-Qiskit replays."""

    circuit = _require_staircase(circuit)
    logical_circuit = to_qiskit_logical(circuit)
    if native_circuit is None:
        native_circuit = compile_staircase_native(circuit)

    require_qiskit()
    from qiskit import QuantumCircuit

    if not isinstance(native_circuit, QuantumCircuit):
        raise TypeError("native_circuit must be a qiskit.QuantumCircuit.")
    if native_circuit.num_qubits != circuit.num_qubits:
        raise ValueError(
            "native_circuit qubit count does not match staircase: "
            f"{native_circuit.num_qubits} != {circuit.num_qubits}."
        )

    dense_state = circuit.prepare()
    logical_state = qiskit_statevector_q0_msb(logical_circuit)
    native_state = qiskit_statevector_q0_msb(native_circuit)
    logical_phase, logical_error = _phase_aligned_comparison(dense_state, logical_state)
    native_phase, native_error = _phase_aligned_comparison(logical_state, native_state)
    counts = tuple(
        sorted((str(name), int(count)) for name, count in native_circuit.count_ops().items())
    )
    operation_names = {name for name, _ in counts}
    return CompilationDiagnostics(
        num_qubits=circuit.num_qubits,
        depth=circuit.depth,
        num_u4_gates=circuit.num_gates,
        logical_replay_fidelity=statevector_fidelity(dense_state, logical_state),
        native_replay_fidelity=statevector_fidelity(dense_state, native_state),
        compile_fidelity=statevector_fidelity(logical_state, native_state),
        logical_phase_over_dense=logical_phase,
        native_phase_over_logical=native_phase,
        logical_phase_aligned_max_abs_error=logical_error,
        native_phase_aligned_max_abs_error=native_error,
        native_operation_counts=counts,
        native_basis_only=(operation_names - {"barrier"}) <= NATIVE_BASIS,
    )


__all__ = [
    "CompilationDiagnostics",
    "NATIVE_BASIS",
    "QiskitUnavailableError",
    "compile_staircase_native",
    "diagnose_qiskit_compilation",
    "project_site_to_qiskit",
    "project_sites_to_qiskit_qargs",
    "qiskit_available",
    "qiskit_statevector_q0_msb",
    "remove_barriers",
    "require_qiskit",
    "statevector_fidelity",
    "to_qiskit_logical",
]
