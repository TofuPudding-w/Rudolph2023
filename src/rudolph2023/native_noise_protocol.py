"""Read-only FakeWashingtonV2 post-processing for Rudolph encodings.

The paper method and the hardware study are intentionally separated. This
module consumes an already exported Iter[D_i O_all] artifact, exactly
synthesizes its ordered U(4) blocks, compiles the result to a
calibration-preserving compact FakeWashingtonV2 path, and evaluates the noisy
density matrix. It never reruns or modifies the optimizer.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import numbers
import os
import platform
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from .circuit import StaircaseCircuit
from .export import file_sha256, verify_reproduction_artifact
from .qiskit_adapter import (
    CompilationDiagnostics,
    compile_staircase_native,
    diagnose_qiskit_compilation,
)
from .washington_noise import (
    CompactWashingtonDevice,
    NativeCompilation,
    NoisyEncodingResult,
    build_compact_washington_device,
    compile_to_native_target,
    evaluate_native_density_matrix,
)

SCHEMA_VERSION = "rudolph2023-fake-washington-native-noise-v1"
MANIFEST_FILENAME = "native_noise_manifest.json"
BACKEND_FILENAME = "fake_washington_compact_snapshot.json"
QPY_FILENAME = "native_circuit.qpy"
CHECKSUMS_FILENAME = "SHA256SUMS"


@dataclass(frozen=True, slots=True)
class LoadedReproductionArtifact:
    """Verified, replayable inputs from one noiseless Rudolph artifact."""

    directory: Path
    manifest: dict[str, Any]
    circuit: StaircaseCircuit
    target_state: np.ndarray
    prepared_state: np.ndarray
    hamiltonian: Any | None
    ground_energy: float | None
    input_file_sha256: dict[str, str]
    source_compatibility: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EncodingNoiseEvaluation:
    """One ideal U(4), native-target, and noisy evaluation chain."""

    block_synthesis: CompilationDiagnostics
    target_compilation: NativeCompilation
    noisy: NoisyEncodingResult


@dataclass(frozen=True, slots=True)
class NativeNoiseExport:
    """Paths and hashes of a completed single-artifact noise evaluation."""

    directory: Path
    manifest_path: Path
    backend_snapshot_path: Path
    native_qpy_path: Path
    checksums_path: Path
    manifest_sha256: str


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _source_compatibility(recorded: Mapping[str, Any]) -> dict[str, Any]:
    """Verify old artifact core files while allowing the new noise layer."""

    files = recorded.get("files")
    aggregate = recorded.get("aggregate_sha256")
    if recorded.get("algorithm") != "sha256" or not isinstance(files, Mapping):
        raise ValueError("Artifact source provenance is malformed.")
    normalized: dict[str, str] = {}
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Artifact source file hashes are malformed.")
        normalized[name] = digest
    expected_aggregate = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if aggregate != expected_aggregate:
        raise ValueError("Artifact source-provenance aggregate hash is invalid.")

    package_root = Path(__file__).resolve().parent
    changed: list[str] = []
    missing: list[str] = []
    for name, expected in sorted(normalized.items()):
        path = package_root / name
        if not path.is_file():
            missing.append(name)
        elif file_sha256(path) != expected:
            changed.append(name)
    allowed_changed = {
        "cli.py",
        "native_noise_protocol.py",
        "qiskit_adapter.py",
        "washington_noise.py",
    }
    disallowed = sorted(set(changed) - allowed_changed)
    if missing or disallowed:
        details = {"missing": missing, "changed_core": disallowed}
        raise ValueError(f"Recorded Rudolph core sources no longer match: {details}.")
    current_names = {path.name for path in package_root.glob("*.py")}
    return {
        "strict_source_tree_match": not changed and current_names == set(normalized),
        "recorded_core_files_verified": True,
        "allowed_changed_integration_files": changed,
        "new_source_files": sorted(current_names - set(normalized)),
        "recorded_aggregate_sha256": aggregate,
    }


def _load_hamiltonian(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> tuple[Any | None, float | None]:
    record = manifest.get("energy_evaluation")
    if record is None:
        return None, None
    if not isinstance(record, Mapping):
        raise ValueError("energy_evaluation must be an object or null.")
    storage = record.get("storage")
    if storage == "csr":
        required = (
            "hamiltonian_csr_data",
            "hamiltonian_csr_indices",
            "hamiltonian_csr_indptr",
            "hamiltonian_shape",
        )
        if any(name not in arrays for name in required):
            raise ValueError("CSR Hamiltonian arrays are incomplete.")
        shape = tuple(int(value) for value in arrays["hamiltonian_shape"])
        if len(shape) != 2:
            raise ValueError("Hamiltonian shape must have two entries.")
        hamiltonian = csr_matrix(
            (
                arrays["hamiltonian_csr_data"],
                arrays["hamiltonian_csr_indices"],
                arrays["hamiltonian_csr_indptr"],
            ),
            shape=shape,
        )
    elif storage == "dense":
        if "hamiltonian_dense" not in arrays:
            raise ValueError("Dense Hamiltonian array is missing.")
        hamiltonian = arrays["hamiltonian_dense"]
    else:
        raise ValueError("Unknown Hamiltonian storage mode.")
    ground_energy = record.get("ground_energy")
    if (
        isinstance(ground_energy, bool)
        or not isinstance(ground_energy, int | float)
        or not math.isfinite(float(ground_energy))
    ):
        raise ValueError("ground_energy must be a finite real scalar.")
    return hamiltonian, float(ground_energy)


def load_reproduction_artifact(
    artifact_directory: str | Path,
) -> LoadedReproductionArtifact:
    """Integrity-check and load an existing noiseless Rudolph artifact."""

    directory = Path(artifact_directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {directory}.")
    manifest_path = directory / "manifest.json"
    arrays_path = directory / "result_arrays.npz"
    checksums_path = directory / "SHA256SUMS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        verification = verify_reproduction_artifact(directory)
        source_compatibility = {
            "strict_source_tree_match": True,
            "recorded_core_files_verified": True,
            "artifact_verification": verification,
        }
    except ValueError as exc:
        if str(exc) != "Current rudolph2023 source tree differs from artifact provenance.":
            raise
        source_compatibility = _source_compatibility(manifest.get("source_provenance", {}))
        source_compatibility["artifact_verification"] = {
            "status": "verified_payload_with_compatible_noise_extension",
            "strict_source_tree_match": False,
        }

    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    circuit = StaircaseCircuit.from_gate_array(arrays["ordered_u4"])
    prepared = circuit.prepare()
    if not np.allclose(prepared, arrays["prepared_state"], rtol=0.0, atol=1e-11):
        raise ValueError("Loaded U(4) gates do not replay the stored prepared state.")
    hamiltonian, ground_energy = _load_hamiltonian(manifest, arrays)
    input_hashes = {
        path.name: file_sha256(path)
        for path in (manifest_path, arrays_path, checksums_path)
    }
    return LoadedReproductionArtifact(
        directory=directory,
        manifest=manifest,
        circuit=circuit,
        target_state=np.asarray(arrays["target_state"], dtype=np.complex128),
        prepared_state=np.asarray(prepared, dtype=np.complex128),
        hamiltonian=hamiltonian,
        ground_energy=ground_energy,
        input_file_sha256=input_hashes,
        source_compatibility=source_compatibility,
    )


def evaluate_circuit_under_washington(
    artifact: LoadedReproductionArtifact,
    device: CompactWashingtonDevice,
    *,
    optimization_level: int = 0,
    seed_transpiler: int = 2026,
    seed_simulator: int = 2026,
    max_density_matrix_qubits: int = 12,
    compute_minimum_eigenvalue: bool | None = None,
    compilation_fidelity_atol: float = 2e-10,
) -> EncodingNoiseEvaluation:
    """Exactly synthesize, target-compile, and noise-evaluate one artifact."""

    if not isinstance(artifact, LoadedReproductionArtifact):
        raise TypeError("artifact must be a LoadedReproductionArtifact.")
    if not isinstance(device, CompactWashingtonDevice):
        raise TypeError("device must be a CompactWashingtonDevice.")
    if artifact.circuit.num_qubits != device.num_qubits:
        raise ValueError(
            f"artifact has N={artifact.circuit.num_qubits}, device has N={device.num_qubits}."
        )
    if (
        isinstance(compilation_fidelity_atol, bool)
        or not isinstance(compilation_fidelity_atol, numbers.Real)
        or not math.isfinite(float(compilation_fidelity_atol))
        or float(compilation_fidelity_atol) <= 0.0
    ):
        raise ValueError("compilation_fidelity_atol must be finite and positive.")
    tolerance = float(compilation_fidelity_atol)

    block_native = compile_staircase_native(artifact.circuit)
    diagnostics = diagnose_qiskit_compilation(artifact.circuit, block_native)
    if not diagnostics.native_basis_only:
        raise ArithmeticError("Exact U(4) synthesis emitted a non-native operation.")
    if 1.0 - diagnostics.compile_fidelity > tolerance:
        raise ArithmeticError(
            "Exact U(4) synthesis failed its ideal fidelity oracle: "
            f"F={diagnostics.compile_fidelity:.17g}."
        )
    target_compilation = compile_to_native_target(
        block_native,
        device,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    noisy = evaluate_native_density_matrix(
        artifact.target_state,
        artifact.prepared_state,
        target_compilation,
        device=device,
        hamiltonian=artifact.hamiltonian,
        ground_energy=artifact.ground_energy,
        num_sites=artifact.circuit.num_qubits,
        seed_simulator=seed_simulator,
        max_density_matrix_qubits=max_density_matrix_qubits,
        compute_minimum_eigenvalue=compute_minimum_eigenvalue,
    )
    recorded_fidelity = float(
        artifact.manifest["final_metrics"]["squared_state_fidelity"]
    )
    if abs(noisy.logical_target_fidelity - recorded_fidelity) > 2e-10:
        raise ArithmeticError(
            "Qiskit logical replay disagrees with the source artifact: "
            f"{noisy.logical_target_fidelity:.17g} vs {recorded_fidelity:.17g}."
        )
    if 1.0 - noisy.compilation_fidelity > tolerance:
        raise ArithmeticError(
            "FakeWashington target compilation failed its fidelity oracle: "
            f"F={noisy.compilation_fidelity:.17g}."
        )
    return EncodingNoiseEvaluation(diagnostics, target_compilation, noisy)


def _diagnostics_dict(value: CompilationDiagnostics) -> dict[str, Any]:
    result = asdict(value)
    result["native_operation_counts"] = dict(value.native_operation_counts)
    return result


def _noise_source_provenance() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    names = (
        "native_noise_protocol.py",
        "qiskit_adapter.py",
        "washington_noise.py",
    )
    files = {name: file_sha256(package_root / name) for name in names}
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"algorithm": "sha256", "files": files, "aggregate_sha256": aggregate}


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_artifact_native_noise_protocol(
    artifact_directory: str | Path,
    output_directory: str | Path,
    *,
    physical_path: Sequence[int] | None = None,
    backend: Any | None = None,
    optimization_level: int = 0,
    seed_transpiler: int = 2026,
    seed_simulator: int = 2026,
    max_density_matrix_qubits: int = 12,
    compute_minimum_eigenvalue: bool | None = None,
    compilation_fidelity_atol: float = 2e-10,
    gate_error: bool = True,
    thermal_relaxation: bool = True,
    temperature_millikelvin: float = 0.0,
    provenance: Mapping[str, Any] | None = None,
) -> NativeNoiseExport:
    """Run and atomically export one provenance-complete noise evaluation."""

    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping or None.")
    if isinstance(max_density_matrix_qubits, bool) or not isinstance(
        max_density_matrix_qubits, numbers.Integral
    ):
        raise TypeError("max_density_matrix_qubits must be an integer.")
    max_density_matrix_qubits = int(max_density_matrix_qubits)
    if max_density_matrix_qubits < 1:
        raise ValueError("max_density_matrix_qubits must be positive.")
    artifact = load_reproduction_artifact(artifact_directory)
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}.")
    source = artifact.directory
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("output_directory must be separate from the read-only source artifact.")
    if artifact.circuit.num_qubits > max_density_matrix_qubits:
        raise ValueError(
            f"Artifact N={artifact.circuit.num_qubits} exceeds "
            f"max_density_matrix_qubits={max_density_matrix_qubits}."
        )
    device = build_compact_washington_device(
        artifact.circuit.num_qubits,
        backend=backend,
        physical_path=None if physical_path is None else tuple(physical_path),
        gate_error=gate_error,
        thermal_relaxation=thermal_relaxation,
        readout_error=False,
        temperature_millikelvin=temperature_millikelvin,
    )
    started = time.perf_counter()
    evaluation = evaluate_circuit_under_washington(
        artifact,
        device,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        seed_simulator=seed_simulator,
        max_density_matrix_qubits=max_density_matrix_qubits,
        compute_minimum_eigenvalue=compute_minimum_eigenvalue,
        compilation_fidelity_atol=compilation_fidelity_atol,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    ).resolve()
    try:
        _write_bytes(staging / BACKEND_FILENAME, _json_bytes(device.provenance_dict()))
        from qiskit import qpy

        with (staging / QPY_FILENAME).open("xb") as handle:
            qpy.dump(evaluation.target_compilation.circuit, handle)
            handle.flush()
            os.fsync(handle.fileno())
        source_metrics = artifact.manifest["final_metrics"]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "created_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "scope": (
                "read-only post-processing of an existing Rudolph Iter[D_i O_all] "
                "artifact; no reoptimization, pruning, approximation, or measurement"
            ),
            "source_artifact": {
                "directory": str(artifact.directory),
                "schema_version": artifact.manifest["schema_version"],
                "target_name": artifact.manifest["target_name"],
                "num_qubits": artifact.circuit.num_qubits,
                "depth": artifact.circuit.depth,
                "sweeps_per_depth": artifact.manifest["sweeps_per_depth"],
                "input_file_sha256": artifact.input_file_sha256,
                "source_compatibility": artifact.source_compatibility,
                "recorded_paper_amplitude_fidelity": source_metrics[
                    "amplitude_fidelity"
                ],
                "recorded_paper_amplitude_infidelity": source_metrics[
                    "paper_amplitude_infidelity"
                ],
                "recorded_squared_state_fidelity": source_metrics[
                    "squared_state_fidelity"
                ],
            },
            "native_compilation_protocol": {
                "block_synthesis": "exact independent U4 to rz/sx/x/cx",
                "basis_fidelity": 1.0,
                "approximation": False,
                "target_optimization_level": int(optimization_level),
                "routing_method": "none",
                "initial_layout": list(range(artifact.circuit.num_qubits)),
                "seed_transpiler": int(seed_transpiler),
                "block_synthesis_diagnostics": _diagnostics_dict(
                    evaluation.block_synthesis
                ),
                "target_compilation": evaluation.target_compilation.provenance_dict(),
            },
            "noise_protocol": {
                "backend": "FakeWashingtonV2",
                "compact_backend_snapshot": BACKEND_FILENAME,
                "backend_snapshot_sha256": device.snapshot_sha256,
                "physical_path": list(device.physical_path),
                "project_site_to_physical_q0_msb": list(
                    device.project_site_to_physical_q0_msb
                ),
                "simulator": "AerSimulator(method='density_matrix')",
                "gate_error": bool(gate_error),
                "thermal_relaxation": bool(thermal_relaxation),
                "readout_error": False,
                "temperature_millikelvin": float(temperature_millikelvin),
                "shots": 1,
                "shots_semantics": "deterministic density-matrix channel output",
                "seed_simulator": int(seed_simulator),
                "max_density_matrix_qubits": int(max_density_matrix_qubits),
                "raw_density_matrix_exported": False,
            },
            "metric_definitions": {
                "paper_amplitude_fidelity": "|<psi_target|psi_logical>|",
                "logical_target_fidelity": "|<psi_target|psi_logical>|^2",
                "compilation_fidelity": "|<psi_logical|psi_native>|^2",
                "noisy_target_fidelity": "<psi_target|rho_noisy|psi_target>",
                "noisy_native_retention_fidelity": (
                    "<psi_native|rho_noisy|psi_native>"
                ),
                "energy_error_per_site": "(Tr[H rho_noisy]-E0)/N",
                "fidelity_clipping": False,
            },
            "noise_evaluation": evaluation.noisy.as_dict(),
            "native_qpy": {
                "path": QPY_FILENAME,
                "sha256": file_sha256(staging / QPY_FILENAME),
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "qiskit": _package_version("qiskit"),
                "qiskit_aer": _package_version("qiskit-aer"),
                "qiskit_ibm_runtime": _package_version("qiskit-ibm-runtime"),
            },
            "noise_source_provenance": _noise_source_provenance(),
            "runtime_seconds": time.perf_counter() - started,
            "provenance": {} if provenance is None else dict(provenance),
        }
        manifest_payload = _json_bytes(manifest)
        _write_bytes(staging / MANIFEST_FILENAME, manifest_payload)
        hashes = {
            BACKEND_FILENAME: file_sha256(staging / BACKEND_FILENAME),
            MANIFEST_FILENAME: file_sha256(staging / MANIFEST_FILENAME),
            QPY_FILENAME: file_sha256(staging / QPY_FILENAME),
        }
        checksum_payload = "".join(
            f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
        ).encode("ascii")
        _write_bytes(staging / CHECKSUMS_FILENAME, checksum_payload)
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return NativeNoiseExport(
        directory=output,
        manifest_path=output / MANIFEST_FILENAME,
        backend_snapshot_path=output / BACKEND_FILENAME,
        native_qpy_path=output / QPY_FILENAME,
        checksums_path=output / CHECKSUMS_FILENAME,
        manifest_sha256=file_sha256(output / MANIFEST_FILENAME),
    )


__all__ = [
    "BACKEND_FILENAME",
    "CHECKSUMS_FILENAME",
    "MANIFEST_FILENAME",
    "QPY_FILENAME",
    "SCHEMA_VERSION",
    "EncodingNoiseEvaluation",
    "LoadedReproductionArtifact",
    "NativeNoiseExport",
    "evaluate_circuit_under_washington",
    "load_reproduction_artifact",
    "run_artifact_native_noise_protocol",
]
