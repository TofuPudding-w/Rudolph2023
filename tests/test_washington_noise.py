"""Independent contracts for the optional FakeWashingtonV2 noise layer.

These tests deliberately exercise the compact calibrated backend rather than
attempting to simulate all 127 Washington qubits.  They are kept separate from
the noiseless Rudolph reproduction and skip cleanly when the optional Qiskit
stack is not installed.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

pytestmark = pytest.mark.qiskit

qiskit = pytest.importorskip("qiskit")
fake_provider = pytest.importorskip("qiskit.providers.fake_provider")
quantum_info = pytest.importorskip("qiskit.quantum_info")
aer_noise = pytest.importorskip("qiskit_aer.noise")
washington_noise = pytest.importorskip("rudolph2023.washington_noise")

QuantumCircuit = qiskit.QuantumCircuit
GenericBackendV2 = fake_provider.GenericBackendV2
Statevector = quantum_info.Statevector
NoiseModel = aer_noise.NoiseModel
depolarizing_error = aer_noise.depolarizing_error


def _bidirectional(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [directed for left, right in edges for directed in ((left, right), (right, left))]


def _backend(
    num_qubits: int,
    edges: list[tuple[int, int]],
    *,
    seed: int = 81,
) -> GenericBackendV2:
    return GenericBackendV2(
        num_qubits=num_qubits,
        basis_gates=["id", "rz", "sx", "x", "cx"],
        coupling_map=_bidirectional(edges),
        seed=seed,
        noise_info=True,
    )


def _haar_unitary(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q_matrix, r_matrix = np.linalg.qr(matrix)
    diagonal = np.diag(r_matrix)
    return q_matrix @ np.diag(np.conj(diagonal / np.abs(diagonal)))


def test_generic_backend_path_selection_is_deterministic_and_validated() -> None:
    backend = _backend(6, [(0, 1), (0, 2), (1, 3), (2, 4), (4, 5)])

    # DFS must backtrack from the too-short 0-1 branch to 0-2-4-5.
    assert washington_noise.select_physical_path(backend, 4) == (0, 2, 4, 5)
    assert washington_noise.validate_physical_path(backend, [5, 4, 2, 0]) == (
        5,
        4,
        2,
        0,
    )
    with pytest.raises(ValueError, match="repeat"):
        washington_noise.validate_physical_path(backend, [0, 2, 0])
    with pytest.raises(ValueError, match="not contiguous"):
        washington_noise.validate_physical_path(backend, [0, 5])


def test_compaction_preserves_calibration_mapping_and_snapshot() -> None:
    backend = _backend(5, [(0, 1), (1, 2), (2, 3), (3, 4)], seed=7)
    device = washington_noise.build_compact_washington_device(
        backend=backend,
        physical_path=(1, 2, 3),
        readout_error=False,
    )

    assert (
        washington_noise.COMPACT_DEVICE_SCHEMA
        == "rudolph2023-compact-backend-noise-v1"
    )
    assert device.num_qubits == 3
    assert device.compact_to_physical == (1, 2, 3)
    assert device.physical_to_compact == ((1, 0), (2, 1), (3, 2))
    # Project q0 is most significant, whereas Qiskit compact wire 0 is least
    # significant.  This reversed physical ledger must therefore be explicit.
    assert device.project_site_to_physical_q0_msb == (3, 2, 1)
    assert set(device.target["cx"]) == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert all(
        max(item.compact_qubits, default=0) < 3
        for item in device.instruction_calibrations
    )

    for compact, physical in enumerate(device.physical_path):
        source = backend.target.qubit_properties[physical]
        copied = device.target.qubit_properties[compact]
        assert copied.t1 == source.t1
        assert copied.t2 == source.t2
        assert copied.frequency == source.frequency
    source_cx = backend.target["cx"][(1, 2)]
    compact_cx = device.target["cx"][(0, 1)]
    assert compact_cx.duration == source_cx.duration
    assert compact_cx.error == source_cx.error

    repeated = washington_noise.build_compact_washington_device(
        backend=backend,
        physical_path=(1, 2, 3),
        readout_error=False,
    )
    assert repeated.snapshot_sha256 == device.snapshot_sha256
    assert device.provenance_dict()["snapshot_sha256"] == device.snapshot_sha256
    compact_view = washington_noise._CompactBackendV2View(device.target, "audit-view")
    assert compact_view.num_qubits == device.num_qubits


def test_native_compilation_is_exact_native_and_swap_free() -> None:
    backend = _backend(3, [(0, 1), (1, 2)], seed=11)
    device = washington_noise.build_compact_washington_device(
        backend=backend,
        physical_path=(0, 1, 2),
        gate_error=False,
        thermal_relaxation=False,
    )
    logical = QuantumCircuit(3)
    logical.unitary(_haar_unitary(4, 3), [0, 1])
    logical.x(2)
    expected = np.asarray(Statevector.from_instruction(logical).data)

    compilation = washington_noise.compile_to_native_target(
        logical,
        device,
        optimization_level=0,
        seed_transpiler=19,
    )
    observed = np.asarray(Statevector.from_instruction(compilation.circuit).data)

    assert compilation.unitary_count == 0
    assert compilation.swap_count == 0
    assert set(dict(compilation.gate_counts)) <= set(device.target.operation_names)
    assert compilation.initial_index_layout == (0, 1, 2)
    assert compilation.final_index_layout == (0, 1, 2)
    assert abs(np.vdot(expected, observed)) ** 2 > 1.0 - 2e-12


def test_zero_noise_recovers_fidelity_energy_and_physical_density() -> None:
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.cx(0, 1)
    target = np.asarray(Statevector.from_instruction(circuit).data)
    hamiltonian = -np.outer(target, target.conj())

    result = washington_noise.evaluate_native_density_matrix(
        target,
        target,
        circuit,
        noise_model=NoiseModel(),
        hamiltonian=hamiltonian,
        ground_energy=-1.0,
        num_sites=2,
    )

    assert (
        washington_noise.NOISY_RESULT_SCHEMA
        == "rudolph2023-native-density-matrix-v1"
    )
    assert result.as_dict()["schema_version"] == washington_noise.NOISY_RESULT_SCHEMA
    assert result.logical_target_fidelity == pytest.approx(1.0, abs=2e-14)
    assert result.compilation_fidelity == pytest.approx(1.0, abs=2e-14)
    assert result.native_target_fidelity == pytest.approx(1.0, abs=2e-14)
    assert result.noisy_target_fidelity == pytest.approx(1.0, abs=2e-14)
    assert result.noisy_native_retention_fidelity == pytest.approx(1.0, abs=2e-14)
    assert result.purity == pytest.approx(1.0, abs=2e-14)
    assert result.density_trace_error < 2e-14
    assert result.hermiticity_residual < 2e-14
    assert result.minimum_eigenvalue is not None
    assert result.minimum_eigenvalue > -2e-14
    assert result.noisy_energy == pytest.approx(-1.0, abs=2e-14)
    assert result.energy_error_per_site == pytest.approx(0.0, abs=2e-14)
    assert result.simulator_method == "density_matrix"


@pytest.mark.parametrize("strength", [0.02, 0.3, 0.8])
def test_depolarizing_channel_matches_closed_form(strength: float) -> None:
    circuit = QuantumCircuit(1)
    circuit.x(0)
    target = np.array([0.0, 1.0], dtype=np.complex128)
    noise_model = NoiseModel()
    noise_model.add_quantum_error(depolarizing_error(strength, 1), "x", [0])

    result = washington_noise.evaluate_native_density_matrix(
        target,
        target,
        circuit,
        noise_model=noise_model,
    )

    # Aer defines E(rho)=(1-lambda)rho+lambda I/2 for one qubit.
    expected = 1.0 - strength / 2.0
    assert result.noisy_target_fidelity == pytest.approx(expected, abs=2e-13)
    assert result.noisy_native_retention_fidelity == pytest.approx(expected, abs=2e-13)
    assert result.minimum_eigenvalue == pytest.approx(strength / 2.0, abs=2e-13)


def test_density_matrix_width_guard_precedes_state_validation() -> None:
    too_wide = QuantumCircuit(13)
    with pytest.raises(ValueError, match="Refusing a 13-qubit full density matrix"):
        washington_noise.evaluate_native_density_matrix(
            [1.0],
            [1.0],
            too_wide,
            noise_model=NoiseModel(),
            max_density_matrix_qubits=12,
        )


def test_boundary_inputs_are_rejected_without_silent_coercion() -> None:
    backend = _backend(3, [(0, 1), (1, 2)])
    with pytest.raises(TypeError, match="integer qubit"):
        washington_noise.validate_physical_path(backend, [0, 1.5])
    with pytest.raises(TypeError, match="instruction name"):
        washington_noise.validate_physical_path(backend, [0, 1], two_qubit_gate="")
    with pytest.raises(TypeError, match="gate_error"):
        washington_noise.build_compact_washington_device(
            backend=backend,
            physical_path=(0, 1),
            gate_error=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="nonnegative"):
        washington_noise.build_compact_washington_device(
            backend=backend,
            physical_path=(0, 1),
            temperature_millikelvin=-0.1,
        )
    with pytest.raises(TypeError, match="num_qubits"):
        washington_noise.build_compact_washington_device(
            num_qubits=2.5,  # type: ignore[arg-type]
            backend=backend,
        )

    device = washington_noise.build_compact_washington_device(
        backend=backend,
        physical_path=(0, 1),
        gate_error=False,
        thermal_relaxation=False,
    )
    circuit = QuantumCircuit(2)
    with pytest.raises(TypeError, match="logical_circuit"):
        washington_noise.compile_to_native_target(object(), device)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed_transpiler"):
        washington_noise.compile_to_native_target(circuit, device, seed_transpiler=1.5)
    with pytest.raises(TypeError, match="max_density_matrix_qubits"):
        washington_noise.evaluate_native_density_matrix(
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            circuit,
            noise_model=NoiseModel(),
            max_density_matrix_qubits=2.5,
        )
    with pytest.raises(ValueError, match="either device or noise_model"):
        washington_noise.evaluate_native_density_matrix(
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            circuit,
            device=device,
            noise_model=NoiseModel(),
        )


def _provider_module(name: str, backend_type: type[object] | None) -> types.ModuleType:
    module = types.ModuleType(name)
    if backend_type is not None:
        module.FakeWashingtonV2 = backend_type
    return module


def test_loader_prefers_runtime_public_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    class RuntimeWashington:
        pass

    class LegacyWashington:
        def __init__(self) -> None:
            raise AssertionError("legacy provider must not be used")

    monkeypatch.setitem(
        sys.modules,
        "qiskit_ibm_runtime.fake_provider",
        _provider_module("qiskit_ibm_runtime.fake_provider", RuntimeWashington),
    )
    monkeypatch.setitem(
        sys.modules,
        "qiskit.providers.fake_provider",
        _provider_module("qiskit.providers.fake_provider", LegacyWashington),
    )
    assert isinstance(washington_noise.load_fake_washington_v2(), RuntimeWashington)


def test_loader_uses_legacy_provider_as_explicit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyWashington:
        pass

    monkeypatch.setitem(
        sys.modules,
        "qiskit_ibm_runtime.fake_provider",
        _provider_module("qiskit_ibm_runtime.fake_provider", None),
    )
    monkeypatch.setitem(
        sys.modules,
        "qiskit.providers.fake_provider",
        _provider_module("qiskit.providers.fake_provider", LegacyWashington),
    )
    assert isinstance(washington_noise.load_fake_washington_v2(), LegacyWashington)


def test_loader_never_substitutes_generic_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "qiskit_ibm_runtime.fake_provider",
        _provider_module("qiskit_ibm_runtime.fake_provider", None),
    )
    monkeypatch.setitem(
        sys.modules,
        "qiskit.providers.fake_provider",
        _provider_module("qiskit.providers.fake_provider", None),
    )
    with pytest.raises(washington_noise.QiskitNoiseDependencyError, match="not available"):
        washington_noise.load_fake_washington_v2()


def test_real_fake_washington_compact_density_matrix_smoke() -> None:
    """Exercise the real calibration object when the runtime extra is present."""

    try:
        backend = washington_noise.load_fake_washington_v2()
    except washington_noise.QiskitNoiseDependencyError:
        pytest.skip("A supported FakeWashingtonV2 provider is not installed.")

    device = washington_noise.build_compact_washington_device(
        2,
        backend=backend,
        readout_error=False,
    )
    logical = QuantumCircuit(2)
    logical.x(0)
    logical.cx(0, 1)
    logical_state = np.asarray(Statevector.from_instruction(logical).data)
    compilation = washington_noise.compile_to_native_target(
        logical,
        device,
        optimization_level=0,
        seed_transpiler=2026,
    )
    result = washington_noise.evaluate_native_density_matrix(
        logical_state,
        logical_state,
        compilation,
        device=device,
        seed_simulator=2026,
        max_density_matrix_qubits=2,
    )

    assert device.source_num_qubits == 127
    assert len(device.physical_path) == 2
    assert len(device.snapshot_sha256) == 64
    assert compilation.swap_count == 0
    assert compilation.unitary_count == 0
    assert result.compilation_fidelity == pytest.approx(1.0, abs=2e-12)
    assert 0.0 <= result.noisy_target_fidelity <= 1.0
    assert result.density_trace_error < 2e-8
    assert result.hermiticity_residual < 2e-8
