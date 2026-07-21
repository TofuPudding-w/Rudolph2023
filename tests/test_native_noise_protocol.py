from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import rudolph2023.cli as cli
import rudolph2023.native_noise_protocol as protocol
from rudolph2023.export import file_sha256, save_reproduction_artifact
from rudolph2023.native_noise_protocol import (
    EncodingNoiseEvaluation,
    LoadedReproductionArtifact,
    evaluate_circuit_under_washington,
    load_reproduction_artifact,
    run_artifact_native_noise_protocol,
)
from rudolph2023.protocol import run_iter_di_o_all
from rudolph2023.targets import exact_ground_state, random_real_mps, tfim_hamiltonian
from rudolph2023.washington_noise import (
    CompactWashingtonDevice,
    NativeCompilation,
    NoisyEncodingResult,
)


class _Target:
    operation_names = {"rz", "sx", "x", "cx"}


class _Circuit:
    pass


def _device(num_qubits: int) -> CompactWashingtonDevice:
    path = tuple(range(num_qubits))
    return CompactWashingtonDevice(
        source_backend_name="offline",
        source_backend_class="tests.OfflineBackend",
        source_num_qubits=num_qubits,
        physical_path=path,
        two_qubit_gate="cx",
        target=_Target(),
        noise_model=object(),
        qubit_calibrations=(),
        instruction_calibrations=(),
        include_gate_error=True,
        include_thermal_relaxation=True,
        include_readout_error=False,
        temperature_millikelvin=0.0,
        snapshot_sha256="a" * 64,
    )


def _diagnostics(num_qubits: int, depth: int) -> protocol.CompilationDiagnostics:
    return protocol.CompilationDiagnostics(
        num_qubits=num_qubits,
        depth=depth,
        num_u4_gates=depth * (num_qubits - 1),
        logical_replay_fidelity=1.0,
        native_replay_fidelity=1.0,
        compile_fidelity=1.0,
        logical_phase_over_dense=0.0,
        native_phase_over_logical=0.0,
        logical_phase_aligned_max_abs_error=0.0,
        native_phase_aligned_max_abs_error=0.0,
        native_operation_counts=(("cx", depth), ("rz", depth)),
        native_basis_only=True,
    )


def _compilation(num_qubits: int) -> NativeCompilation:
    identity = tuple(range(num_qubits))
    return NativeCompilation(
        circuit=_Circuit(),
        gate_counts=(("cx", 1), ("rz", 2)),
        depth=3,
        size=3,
        swap_count=0,
        unitary_count=0,
        initial_index_layout=identity,
        final_index_layout=identity,
        optimization_level=0,
        seed_transpiler=2026,
    )


def _noisy(num_qubits: int, *, logical_fidelity: float) -> NoisyEncodingResult:
    identity = tuple(range(num_qubits))
    return NoisyEncodingResult(
        num_qubits=num_qubits,
        logical_target_fidelity=logical_fidelity,
        compilation_fidelity=1.0,
        native_target_fidelity=logical_fidelity,
        noisy_target_fidelity=0.91,
        noisy_native_retention_fidelity=0.92,
        purity=0.83,
        density_trace_real=1.0,
        density_trace_imag=0.0,
        density_trace_error=0.0,
        hermiticity_residual=0.0,
        minimum_eigenvalue=0.0,
        eigenvalue_check_performed=True,
        noisy_energy=-1.0,
        energy_error_per_site=0.02,
        gate_counts=(("cx", 1), ("rz", 2)),
        native_depth=3,
        native_size=3,
        initial_index_layout=identity,
        final_index_layout=identity,
        physical_path=identity,
        backend_snapshot_sha256="a" * 64,
        simulator_method="density_matrix",
        seed_simulator=2026,
        runtime_seconds=0.01,
        qiskit_version="test",
        qiskit_aer_version="test",
    )


def _artifact(tmp_path: Path, *, with_energy: bool = False) -> Path:
    if with_energy:
        hamiltonian = tfim_hamiltonian(2, coupling_zz=1.0, field_x=0.7)
        ground = exact_ground_state(hamiltonian)
        target = ground.state
    else:
        hamiltonian = None
        ground = None
        target = random_real_mps(2, max_bond=2, seed=4).state
    result = run_iter_di_o_all(target, max_depth=1, sweeps_per_depth=0)
    output = tmp_path / ("energy_source" if with_energy else "source")
    save_reproduction_artifact(
        result,
        output,
        target_name="test",
        hamiltonian=hamiltonian,
        ground_energy=None if ground is None else ground.energy,
    )
    return output


def test_load_artifact_replays_circuit_and_reconstructs_sparse_hamiltonian(
    tmp_path: Path,
) -> None:
    loaded = load_reproduction_artifact(_artifact(tmp_path, with_energy=True))
    assert isinstance(loaded, LoadedReproductionArtifact)
    assert loaded.circuit.num_qubits == 2
    assert loaded.source_compatibility["recorded_core_files_verified"] is True
    np.testing.assert_allclose(loaded.circuit.prepare(), loaded.prepared_state, atol=1e-13)
    assert loaded.hamiltonian.shape == (4, 4)
    assert loaded.ground_energy is not None
    assert set(loaded.input_file_sha256) == {
        "manifest.json",
        "result_arrays.npz",
        "SHA256SUMS.json",
    }
    with pytest.raises(FileNotFoundError):
        load_reproduction_artifact(tmp_path / "missing")


def test_source_extension_compatibility_rejects_core_changes() -> None:
    package = Path(protocol.__file__).resolve().parent
    files = {"circuit.py": file_sha256(package / "circuit.py")}
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = protocol._source_compatibility(
        {"algorithm": "sha256", "files": files, "aggregate_sha256": aggregate}
    )
    assert result["recorded_core_files_verified"] is True
    assert "native_noise_protocol.py" in result["new_source_files"]

    bad_files = {"circuit.py": "0" * 64}
    bad_aggregate = hashlib.sha256(
        json.dumps(bad_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="core sources"):
        protocol._source_compatibility(
            {
                "algorithm": "sha256",
                "files": bad_files,
                "aggregate_sha256": bad_aggregate,
            }
        )
    with pytest.raises(ValueError, match="malformed"):
        protocol._source_compatibility({})


def test_hamiltonian_loader_covers_none_dense_and_invalid_storage() -> None:
    assert protocol._load_hamiltonian({"energy_evaluation": None}, {}) == (None, None)
    dense = np.eye(4, dtype=np.complex128)
    observed, energy = protocol._load_hamiltonian(
        {"energy_evaluation": {"storage": "dense", "ground_energy": -1.0}},
        {"hamiltonian_dense": dense},
    )
    np.testing.assert_array_equal(observed, dense)
    assert energy == -1.0
    with pytest.raises(ValueError, match="Unknown"):
        protocol._load_hamiltonian(
            {"energy_evaluation": {"storage": "other", "ground_energy": 0.0}},
            {},
        )


def test_evaluation_chain_enforces_both_compilation_oracles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_reproduction_artifact(_artifact(tmp_path))
    device = _device(2)
    recorded = float(loaded.manifest["final_metrics"]["squared_state_fidelity"])
    diagnostics = _diagnostics(2, 1)
    compilation = _compilation(2)
    noisy = _noisy(2, logical_fidelity=recorded)
    monkeypatch.setattr(protocol, "compile_staircase_native", lambda _circuit: _Circuit())
    monkeypatch.setattr(
        protocol, "diagnose_qiskit_compilation", lambda _circuit, _native: diagnostics
    )
    monkeypatch.setattr(
        protocol, "compile_to_native_target", lambda _native, _device, **_kwargs: compilation
    )
    monkeypatch.setattr(
        protocol, "evaluate_native_density_matrix", lambda *_args, **_kwargs: noisy
    )
    result = evaluate_circuit_under_washington(loaded, device)
    assert isinstance(result, EncodingNoiseEvaluation)
    assert result.block_synthesis.compile_fidelity == 1.0
    assert protocol._diagnostics_dict(diagnostics)["native_operation_counts"] == {
        "cx": 1,
        "rz": 1,
    }

    with pytest.raises(TypeError, match="LoadedReproductionArtifact"):
        evaluate_circuit_under_washington(object(), device)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_circuit_under_washington(
            loaded, device, compilation_fidelity_atol=0.0
        )
    bad = replace(diagnostics, compile_fidelity=0.5)
    monkeypatch.setattr(
        protocol, "diagnose_qiskit_compilation", lambda _circuit, _native: bad
    )
    with pytest.raises(ArithmeticError, match="synthesis failed"):
        evaluate_circuit_under_washington(loaded, device)


def test_atomic_export_and_cli_summary_with_offline_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _artifact(tmp_path)
    loaded = load_reproduction_artifact(source)
    device = _device(2)
    recorded = float(loaded.manifest["final_metrics"]["squared_state_fidelity"])
    evaluation = EncodingNoiseEvaluation(
        _diagnostics(2, 1),
        _compilation(2),
        _noisy(2, logical_fidelity=recorded),
    )
    monkeypatch.setattr(protocol, "build_compact_washington_device", lambda *_a, **_k: device)
    monkeypatch.setattr(protocol, "evaluate_circuit_under_washington", lambda *_a, **_k: evaluation)

    class _Qpy:
        @staticmethod
        def dump(_circuit: object, handle: object) -> None:
            handle.write(b"offline-qpy")  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "qiskit", SimpleNamespace(qpy=_Qpy))
    output = tmp_path / "noise_output"
    exported = run_artifact_native_noise_protocol(
        source,
        output,
        physical_path=(0, 1),
        provenance={"test": True},
    )
    assert exported.manifest_path.exists()
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == protocol.SCHEMA_VERSION
    assert manifest["scope"].startswith("read-only")
    assert manifest["noise_evaluation"]["fidelity"]["noisy_target"] == 0.91
    assert manifest["provenance"] == {"test": True}
    assert protocol._noise_source_provenance()["aggregate_sha256"]
    assert protocol._package_version("definitely-not-a-real-distribution") is None
    for line in exported.checksums_path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        assert file_sha256(output / name) == digest
    with pytest.raises(FileExistsError):
        run_artifact_native_noise_protocol(source, output)

    monkeypatch.setattr(
        cli,
        "run_artifact_native_noise_protocol",
        lambda *_a, **_k: exported,
    )
    assert (
        cli.main(
            [
                "native-noise",
                "--artifact",
                str(source),
                "--output",
                str(tmp_path / "ignored"),
                "--physical-path",
                "0",
                "1",
                "--skip-minimum-eigenvalue",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "completed"
    assert summary["noisy_target_fidelity"] == 0.91


def test_json_writer_and_protocol_boundary_validation(tmp_path: Path) -> None:
    payload = protocol._json_bytes({"value": 1})
    path = tmp_path / "payload.json"
    protocol._write_bytes(path, payload)
    assert json.loads(path.read_text()) == {"value": 1}
    with pytest.raises(FileExistsError):
        protocol._write_bytes(path, payload)
    with pytest.raises(ValueError):
        protocol._json_bytes({"value": float("nan")})
    with pytest.raises(TypeError, match="mapping"):
        run_artifact_native_noise_protocol(
            tmp_path / "missing",
            tmp_path / "output",
            provenance=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="max_density_matrix_qubits"):
        run_artifact_native_noise_protocol(
            tmp_path / "missing",
            tmp_path / "output",
            max_density_matrix_qubits=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="positive"):
        run_artifact_native_noise_protocol(
            tmp_path / "missing",
            tmp_path / "output",
            max_density_matrix_qubits=0,
        )
