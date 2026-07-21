"""Provenance-complete ordered-U(4) artifact export and integrity checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, isspmatrix

from .circuit import StaircaseCircuit
from .conventions import (
    QUBIT_ORDER,
    TWO_QUBIT_BASIS,
    validate_integer,
    validate_normalized_statevector,
    validate_real,
)
from .metrics import energy_error_per_site, energy_expectation, fidelity_metrics
from .protocol import IterDiOAllResult, expected_cumulative_gate_updates

SCHEMA_VERSION = "rudolph2023-iter-di-o-all-v2"


def semantic_array_sha256(array: ArrayLike) -> str:
    """Hash dtype, shape, and C-order bytes of an array."""

    materialized = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(materialized.dtype).encode("utf-8"))
    digest.update(json.dumps(materialized.shape).encode("utf-8"))
    digest.update(materialized.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return SHA-256 of one file without following any manifest convention."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _complex_json(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _source_tree_provenance() -> dict[str, Any]:
    """Fingerprint the exact Python package sources used for this artifact."""

    package_root = Path(__file__).resolve().parent
    file_hashes = {
        path.name: file_sha256(path)
        for path in sorted(package_root.glob("*.py"), key=lambda item: item.name)
    }
    aggregate = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "algorithm": "sha256",
        "package": "rudolph2023",
        "files": file_hashes,
        "aggregate_sha256": aggregate,
    }


def _require_close(name: str, observed: float, expected: float, *, atol: float) -> None:
    if not math.isfinite(observed) or not math.isfinite(expected):
        raise ValueError(f"{name} must be finite.")
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=atol):
        raise ValueError(
            f"{name} mismatch: observed={observed:.17g}, expected={expected:.17g}."
        )


def save_reproduction_artifact(
    result: IterDiOAllResult,
    output_directory: str | Path,
    *,
    target_name: str,
    target_metadata: dict[str, Any] | None = None,
    hamiltonian: ArrayLike | Any | None = None,
    ground_energy: float | None = None,
) -> Path:
    """Save ordered gates, states, metrics, environment, and checksums."""

    if not isinstance(result, IterDiOAllResult):
        raise TypeError("result must be IterDiOAllResult.")
    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError("target_name must be nonempty.")
    output = Path(output_directory).resolve()

    if (hamiltonian is None) != (ground_energy is None):
        raise ValueError("hamiltonian and ground_energy must be provided together.")
    gates = result.final_circuit.gate_array()
    arrays_path = output / "result_arrays.npz"
    arrays: dict[str, np.ndarray] = {
        "ordered_u4": gates,
        "target_state": result.canonical_target,
        "prepared_state": result.final_prepared_state,
    }
    energy_evaluation: dict[str, Any] | None = None
    if hamiltonian is not None and ground_energy is not None:
        if not isinstance(ground_energy, int | float | np.floating):
            raise TypeError("ground_energy must be real.")
        ground_energy = float(ground_energy)
        if not math.isfinite(ground_energy):
            raise ValueError("ground_energy must be finite.")
        if isspmatrix(hamiltonian):
            sparse_h = hamiltonian.tocsr().astype(np.complex128)
            if not np.all(np.isfinite(sparse_h.data)):
                raise ValueError("Hamiltonian must contain only finite values.")
            arrays.update(
                {
                    "hamiltonian_csr_data": sparse_h.data,
                    "hamiltonian_csr_indices": sparse_h.indices,
                    "hamiltonian_csr_indptr": sparse_h.indptr,
                    "hamiltonian_shape": np.asarray(sparse_h.shape, dtype=np.int64),
                }
            )
            storage = "csr"
        else:
            dense_h = np.asarray(hamiltonian, dtype=np.complex128)
            if not np.all(np.isfinite(dense_h)):
                raise ValueError("Hamiltonian must contain only finite values.")
            arrays["hamiltonian_dense"] = dense_h
            storage = "dense"
        prepared_energy = energy_expectation(result.final_prepared_state, hamiltonian)
        error_per_site = energy_error_per_site(
            result.final_prepared_state, hamiltonian, ground_energy
        )
        energy_evaluation = {
            "storage": storage,
            "ground_energy": ground_energy,
            "prepared_energy": prepared_energy,
            "energy_error_per_site": error_per_site,
        }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(arrays_path, **arrays)
    depth_rows = []
    for record in result.depth_records:
        depth_rows.append(
            {
                "depth": record.depth,
                "candidate_amplitude_fidelity": record.candidate_metrics.amplitude_fidelity,
                "optimized_amplitude_fidelity": record.optimized_metrics.amplitude_fidelity,
                "optimized_squared_fidelity": record.optimized_metrics.squared_state_fidelity,
                "paper_infidelity": record.optimized_metrics.paper_amplitude_infidelity,
                "gate_updates_this_depth": record.gate_updates_this_depth,
                "cumulative_gate_updates": record.cumulative_gate_updates,
                "analytic_overlap_identity_error": record.analytic.extracted_overlap_error,
                "candidate_overlap_identity_error": record.candidate_overlap_identity_error,
                "residual_overlap_identity_error": record.residual_overlap_identity_error,
                "max_sweep_environment_error": max(
                    (
                        sweep.max_environment_identity_error
                        for sweep in record.optimization.sweeps
                    ),
                    default=0.0,
                ),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "Iter[D_i O_all]",
        "target_name": target_name,
        "target_metadata": target_metadata or {},
        "num_qubits": result.final_circuit.num_qubits,
        "depth": result.max_depth,
        "sweeps_per_depth": result.sweeps_per_depth,
        "learning_rate": result.learning_rate,
        "gate_layout": {
            "shape": list(gates.shape),
            "order": "physical-execution layer-major; each layer left-site 0..N-2",
            "qubit_order": QUBIT_ORDER,
            "two_qubit_basis": list(TWO_QUBIT_BASIS),
            "terminal_u2_folded": True,
        },
        "final_metrics": {
            "overlap": _complex_json(result.final_metrics.overlap),
            "amplitude_fidelity": result.final_metrics.amplitude_fidelity,
            "paper_amplitude_infidelity": result.final_metrics.paper_amplitude_infidelity,
            "squared_state_fidelity": result.final_metrics.squared_state_fidelity,
            "squared_infidelity": result.final_metrics.squared_infidelity,
        },
        "energy_evaluation": energy_evaluation,
        "total_gate_updates": result.total_gate_updates,
        "depth_results": depth_rows,
        "semantic_sha256": {
            name: semantic_array_sha256(array) for name, array in arrays.items()
        },
        "source_provenance": _source_tree_provenance(),
        "execution": {
            "argv": list(sys.argv),
            "working_directory": str(Path.cwd().resolve()),
        },
        "reproduction_assumptions": {
            "random_mps_seed": "explicit CLI/API seed; paper does not publish sample seeds",
            "unitary_completion": "deterministic computational-anchor double MGS",
            "fractional_power": "principal complex-Schur branch; eigenphase -pi mapped to +pi",
            "heisenberg_grid_site_order": "row-major open boundaries",
        },
        "numerical_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    checksums = {
        "manifest.json": file_sha256(manifest_path),
        "result_arrays.npz": file_sha256(arrays_path),
    }
    checksum_path = output / "SHA256SUMS.json"
    checksum_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def verify_reproduction_artifact(output_directory: str | Path) -> dict[str, Any]:
    """Verify internal integrity, source identity, metrics, layout, and replay."""

    output = Path(output_directory).resolve()
    checksums = json.loads((output / "SHA256SUMS.json").read_text(encoding="utf-8"))
    if set(checksums) != {"manifest.json", "result_arrays.npz"}:
        raise ValueError("Checksum index has an unexpected file set.")
    for relative, expected in checksums.items():
        observed = file_sha256(output / relative)
        if observed != expected:
            raise ValueError(f"File checksum mismatch for {relative}.")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported artifact schema.")
    if manifest.get("protocol") != "Iter[D_i O_all]":
        raise ValueError("Artifact protocol identifier is incorrect.")
    with np.load(output / "result_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if set(arrays) != set(manifest.get("semantic_sha256", {})):
        raise ValueError("Array archive and semantic-hash index have different keys.")
    for name, array in arrays.items():
        expected = manifest["semantic_sha256"][name]
        if semantic_array_sha256(array) != expected:
            raise ValueError(f"Semantic checksum mismatch for {name}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Array {name} contains non-finite values.")

    required = {"ordered_u4", "target_state", "prepared_state"}
    if not required.issubset(arrays):
        raise ValueError("Artifact is missing required gate/state arrays.")
    gates = arrays["ordered_u4"]
    target = validate_normalized_statevector(arrays["target_state"])
    prepared = validate_normalized_statevector(arrays["prepared_state"])
    num_qubits = validate_integer(
        manifest.get("num_qubits"), name="manifest.num_qubits", minimum=2
    )
    depth = validate_integer(manifest.get("depth"), name="manifest.depth", minimum=0)
    sweeps = validate_integer(
        manifest.get("sweeps_per_depth"),
        name="manifest.sweeps_per_depth",
        minimum=0,
    )
    validate_real(
        manifest.get("learning_rate"),
        name="manifest.learning_rate",
        minimum=0.0,
        maximum=1.0,
    )
    expected_shape = (depth, num_qubits - 1, 4, 4)
    if gates.shape != expected_shape:
        raise ValueError(
            f"Gate array shape {gates.shape} does not match manifest {expected_shape}."
        )
    layout = manifest.get("gate_layout", {})
    if layout.get("shape") != list(expected_shape):
        raise ValueError("gate_layout.shape does not match the stored gates.")
    if layout.get("order") != (
        "physical-execution layer-major; each layer left-site 0..N-2"
    ):
        raise ValueError("Unsupported gate execution order in gate layout.")
    if layout.get("qubit_order") != QUBIT_ORDER:
        raise ValueError("Unsupported qubit order in gate layout.")
    if tuple(layout.get("two_qubit_basis", ())) != TWO_QUBIT_BASIS:
        raise ValueError("Unsupported two-qubit basis in gate layout.")
    if layout.get("terminal_u2_folded") is not True:
        raise ValueError("Artifact does not declare the required terminal-U(2) fold.")
    circuit = StaircaseCircuit.from_gate_array(gates)
    replay = circuit.prepare()
    replay_error = float(np.max(np.abs(replay - prepared)))
    if replay_error > 1e-11:
        raise ValueError(f"Ordered-gate replay mismatch: {replay_error:.3e}.")
    metrics = fidelity_metrics(target, replay)
    overlap = metrics.overlap
    expected_overlap = manifest["final_metrics"]["overlap"]
    serialized = complex(expected_overlap["real"], expected_overlap["imag"])
    if not math.isfinite(serialized.real) or not math.isfinite(serialized.imag):
        raise ValueError("Manifest overlap is non-finite.")
    if abs(overlap - serialized) > 1e-11:
        raise ValueError("Manifest overlap does not match replay.")
    for name, observed in (
        ("amplitude_fidelity", metrics.amplitude_fidelity),
        ("paper_amplitude_infidelity", metrics.paper_amplitude_infidelity),
        ("squared_state_fidelity", metrics.squared_state_fidelity),
        ("squared_infidelity", metrics.squared_infidelity),
    ):
        _require_close(
            f"final_metrics.{name}",
            observed,
            float(manifest["final_metrics"][name]),
            atol=1e-12,
        )

    expected_updates = expected_cumulative_gate_updates(num_qubits, depth, sweeps)
    if manifest.get("total_gate_updates") != expected_updates:
        raise ValueError("Manifest total_gate_updates violates the protocol identity.")
    depth_rows = manifest.get("depth_results")
    if not isinstance(depth_rows, list) or len(depth_rows) != depth:
        raise ValueError("depth_results length does not match circuit depth.")
    for row_depth, row in enumerate(depth_rows, start=1):
        expected_this = sweeps * row_depth * (num_qubits - 1)
        expected_cumulative = expected_cumulative_gate_updates(
            num_qubits, row_depth, sweeps
        )
        if (
            row.get("depth") != row_depth
            or row.get("gate_updates_this_depth") != expected_this
            or row.get("cumulative_gate_updates") != expected_cumulative
        ):
            raise ValueError(f"Depth-row update identity failed at depth {row_depth}.")
        for metric_name in (
            "candidate_amplitude_fidelity",
            "optimized_amplitude_fidelity",
            "optimized_squared_fidelity",
            "paper_infidelity",
            "analytic_overlap_identity_error",
            "candidate_overlap_identity_error",
            "residual_overlap_identity_error",
            "max_sweep_environment_error",
        ):
            value = row.get(metric_name)
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValueError(f"Depth metric {metric_name} is non-finite or missing.")

    energy_record = manifest.get("energy_evaluation")
    if energy_record is not None:
        storage = energy_record.get("storage")
        if storage == "csr":
            csr_keys = {
                "hamiltonian_csr_data",
                "hamiltonian_csr_indices",
                "hamiltonian_csr_indptr",
                "hamiltonian_shape",
            }
            if not csr_keys.issubset(arrays):
                raise ValueError("CSR Hamiltonian arrays are incomplete.")
            shape_values = tuple(int(value) for value in arrays["hamiltonian_shape"])
            if len(shape_values) != 2:
                raise ValueError("Hamiltonian shape must have two entries.")
            hamiltonian = csr_matrix(
                (
                    arrays["hamiltonian_csr_data"],
                    arrays["hamiltonian_csr_indices"],
                    arrays["hamiltonian_csr_indptr"],
                ),
                shape=shape_values,
            )
        elif storage == "dense":
            if "hamiltonian_dense" not in arrays:
                raise ValueError("Dense Hamiltonian array is missing.")
            hamiltonian = arrays["hamiltonian_dense"]
        else:
            raise ValueError("Unknown Hamiltonian storage mode.")
        ground_energy = float(energy_record["ground_energy"])
        observed_energy = energy_expectation(replay, hamiltonian)
        observed_error = energy_error_per_site(replay, hamiltonian, ground_energy)
        _require_close(
            "energy_evaluation.prepared_energy",
            observed_energy,
            float(energy_record["prepared_energy"]),
            atol=1e-10,
        )
        _require_close(
            "energy_evaluation.energy_error_per_site",
            observed_error,
            float(energy_record["energy_error_per_site"]),
            atol=1e-11,
        )
    elif any(name.startswith("hamiltonian_") for name in arrays):
        raise ValueError("Hamiltonian arrays exist without an energy-evaluation record.")

    source = manifest.get("source_provenance")
    if source != _source_tree_provenance():
        raise ValueError("Current rudolph2023 source tree differs from artifact provenance.")
    return {
        "status": "verified",
        "schema_version": SCHEMA_VERSION,
        "replay_error": replay_error,
        "amplitude_fidelity": metrics.amplitude_fidelity,
        "source_tree_sha256": source["aggregate_sha256"],
        "energy_verified": energy_record is not None,
    }


__all__ = [
    "SCHEMA_VERSION",
    "file_sha256",
    "save_reproduction_artifact",
    "semantic_array_sha256",
    "verify_reproduction_artifact",
]
