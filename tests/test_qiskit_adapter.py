from __future__ import annotations

import importlib
import sys
from types import ModuleType

import numpy as np
import pytest

import rudolph2023.qiskit_adapter as adapter
from rudolph2023.circuit import (
    StaircaseCircuit,
    StaircaseLayer,
    embed_adjacent_u4,
)
from rudolph2023.qiskit_adapter import (
    NATIVE_BASIS,
    compile_staircase_native,
    diagnose_qiskit_compilation,
    project_site_to_qiskit,
    project_sites_to_qiskit_qargs,
    qiskit_statevector_q0_msb,
    statevector_fidelity,
    to_qiskit_logical,
)


def _haar(dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q_matrix, r_matrix = np.linalg.qr(matrix)
    diagonal = np.diag(r_matrix)
    phases = np.where(np.abs(diagonal) == 0.0, 1.0, diagonal / np.abs(diagonal))
    return np.asarray(q_matrix @ np.diag(phases.conj()), dtype=np.complex128)


def _random_circuit(num_qubits: int, depth: int, *, seed: int) -> StaircaseCircuit:
    rng = np.random.default_rng(seed)
    layers = tuple(
        StaircaseLayer(
            num_qubits,
            tuple(_haar(4, rng) for _ in range(num_qubits - 1)),
        )
        for _ in range(depth)
    )
    return StaircaseCircuit(num_qubits, layers)


def test_adapter_import_is_lazy_and_unavailable_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = "qiskit" in sys.modules
    importlib.reload(adapter)
    assert ("qiskit" in sys.modules) is before

    assert isinstance(adapter.qiskit_available(), bool)
    monkeypatch.setattr(adapter.importlib.util, "find_spec", lambda _name: None)
    assert not adapter.qiskit_available()

    def unavailable(_name: str) -> ModuleType:
        raise ImportError("intentionally unavailable")

    monkeypatch.setattr(adapter.importlib, "import_module", unavailable)
    with pytest.raises(adapter.QiskitUnavailableError, match="optional noise dependencies"):
        adapter.require_qiskit()


def test_site_mapping_preserves_ordered_local_u4_basis() -> None:
    assert tuple(project_site_to_qiskit(site, num_qubits=4) for site in range(4)) == (
        3,
        2,
        1,
        0,
    )
    assert project_sites_to_qiskit_qargs((0, 1), num_qubits=4) == (2, 3)
    assert project_sites_to_qiskit_qargs((1, 2), num_qubits=4) == (1, 2)
    assert project_sites_to_qiskit_qargs((2, 3), num_qubits=4) == (0, 1)


def test_mapping_and_statevector_fidelity_reject_ambiguous_inputs() -> None:
    for width, error in ((True, TypeError), (1, ValueError), (3.0, TypeError)):
        with pytest.raises(error):
            project_site_to_qiskit(0, num_qubits=width)  # type: ignore[arg-type]
    for site in (True, 1.5):
        with pytest.raises(TypeError):
            project_site_to_qiskit(site, num_qubits=3)  # type: ignore[arg-type]
    for site in (-1, 3):
        with pytest.raises(ValueError, match="outside"):
            project_site_to_qiskit(site, num_qubits=3)
    with pytest.raises(TypeError, match="tuple"):
        project_sites_to_qiskit_qargs([0, 1], num_qubits=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly two"):
        project_sites_to_qiskit_qargs((0,), num_qubits=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ordered adjacent"):
        project_sites_to_qiskit_qargs((1, 0), num_qubits=3)
    with pytest.raises(ValueError, match="ordered adjacent"):
        project_sites_to_qiskit_qargs((0, 2), num_qubits=3)

    reference = np.array([1.0, 2.0j], dtype=np.complex128)
    assert statevector_fidelity(reference, -3.5j * reference) == pytest.approx(
        1.0, abs=2e-15
    )
    with pytest.raises(ValueError, match="same shape"):
        statevector_fidelity([1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="empty"):
        statevector_fidelity([], [])
    with pytest.raises(ValueError, match="finite"):
        statevector_fidelity([np.nan, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="nonzero"):
        statevector_fidelity([0.0, 0.0], [1.0, 0.0])


def test_wrong_circuit_types_are_rejected_before_optional_work() -> None:
    with pytest.raises(TypeError, match="StaircaseCircuit"):
        to_qiskit_logical(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StaircaseCircuit"):
        compile_staircase_native(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StaircaseCircuit"):
        diagnose_qiskit_compilation(object())  # type: ignore[arg-type]


def test_asymmetric_u4_logical_circuit_matches_full_operator_oracle() -> None:
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    circuit = _random_circuit(3, 2, seed=734)
    translated = to_qiskit_logical(circuit)
    observed_state = qiskit_statevector_q0_msb(translated)
    expected_operator = np.eye(1 << circuit.num_qubits, dtype=np.complex128)
    for gate in circuit.flat_gates:
        expected_operator = (
            embed_adjacent_u4(gate.matrix, gate.left_site, circuit.num_qubits)
            @ expected_operator
        )

    np.testing.assert_allclose(observed_state, circuit.prepare(), rtol=0.0, atol=3e-13)
    np.testing.assert_allclose(
        np.asarray(Operator(translated).data),
        expected_operator,
        rtol=0.0,
        atol=3e-13,
    )
    assert translated.count_ops() == {"unitary": circuit.num_gates}


@pytest.mark.parametrize("seed", [4, 19, 907])
def test_multilayer_native_synthesis_matches_dense_executor(seed: int) -> None:
    pytest.importorskip("qiskit")
    circuit = _random_circuit(4, 3, seed=seed)

    native = compile_staircase_native(circuit)
    native_state = qiskit_statevector_q0_msb(native)
    diagnostics = diagnose_qiskit_compilation(circuit, native)

    assert set(native.count_ops()) <= NATIVE_BASIS
    assert "unitary" not in native.count_ops()
    assert "barrier" not in native.count_ops()
    assert statevector_fidelity(circuit.prepare(), native_state) > 1.0 - 3e-13
    assert diagnostics.num_qubits == 4
    assert diagnostics.depth == 3
    assert diagnostics.num_u4_gates == 9
    assert diagnostics.logical_replay_fidelity > 1.0 - 3e-14
    assert diagnostics.native_replay_fidelity > 1.0 - 3e-13
    assert diagnostics.compile_fidelity > 1.0 - 3e-13
    assert diagnostics.logical_phase_aligned_max_abs_error < 3e-13
    assert diagnostics.native_phase_aligned_max_abs_error < 3e-13
    assert diagnostics.native_basis_only


def test_exact_synthesis_preserves_accumulated_global_phase() -> None:
    pytest.importorskip("qiskit")
    first_phase = 0.37
    second_phase = -0.19
    circuit = StaircaseCircuit(
        3,
        (
            StaircaseLayer(
                3,
                (
                    np.exp(1j * first_phase) * np.eye(4, dtype=np.complex128),
                    np.exp(1j * second_phase) * np.eye(4, dtype=np.complex128),
                ),
            ),
        ),
    )
    logical = qiskit_statevector_q0_msb(to_qiskit_logical(circuit))
    native_circuit = compile_staircase_native(circuit)
    native = qiskit_statevector_q0_msb(native_circuit)
    expected_phase = first_phase + second_phase

    assert abs(np.angle(logical[0]) - expected_phase) < 3e-13
    assert abs(np.angle(native[0]) - expected_phase) < 3e-13
    np.testing.assert_allclose(native, logical, rtol=0.0, atol=3e-13)
    assert set(native_circuit.count_ops()) <= NATIVE_BASIS


def test_barriers_can_be_retained_or_removed_without_changing_state() -> None:
    pytest.importorskip("qiskit")
    circuit = _random_circuit(3, 2, seed=81)
    retained = compile_staircase_native(
        circuit,
        insert_barriers=True,
        remove_inserted_barriers=False,
    )
    removed = compile_staircase_native(
        circuit,
        insert_barriers=True,
        remove_inserted_barriers=True,
    )

    assert retained.count_ops()["barrier"] == circuit.num_gates - 1
    assert "barrier" not in removed.count_ops()
    assert diagnose_qiskit_compilation(circuit, retained).native_basis_only
    np.testing.assert_allclose(
        qiskit_statevector_q0_msb(retained),
        qiskit_statevector_q0_msb(removed),
        rtol=0.0,
        atol=3e-13,
    )


def test_qiskit_type_width_and_option_error_branches() -> None:
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    circuit = _random_circuit(3, 1, seed=12)
    with pytest.raises(TypeError, match="insert_barriers"):
        to_qiskit_logical(circuit, insert_barriers=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="insert_barriers"):
        compile_staircase_native(circuit, insert_barriers=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="remove_inserted_barriers"):
        compile_staircase_native(
            circuit,
            remove_inserted_barriers=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="QuantumCircuit"):
        adapter.remove_barriers(object())
    with pytest.raises(TypeError, match="QuantumCircuit"):
        qiskit_statevector_q0_msb(object())
    with pytest.raises(TypeError, match="native_circuit"):
        diagnose_qiskit_compilation(circuit, object())
    with pytest.raises(ValueError, match="qubit count"):
        diagnose_qiskit_compilation(circuit, QuantumCircuit(2))


def test_empty_staircase_translates_to_identity_without_spurious_gates() -> None:
    pytest.importorskip("qiskit")
    circuit = StaircaseCircuit.empty(3)
    logical = to_qiskit_logical(circuit, insert_barriers=True)
    native = compile_staircase_native(circuit, insert_barriers=True)
    diagnostics = diagnose_qiskit_compilation(circuit, native)

    assert logical.count_ops() == {}
    assert native.count_ops() == {}
    assert diagnostics.num_u4_gates == 0
    assert diagnostics.compile_fidelity == pytest.approx(1.0, abs=2e-15)
    assert diagnostics.native_basis_only
