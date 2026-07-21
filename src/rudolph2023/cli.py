"""Command-line entry points for smoke and formal main-method runs."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.sparse import csr_matrix

from .export import save_reproduction_artifact, verify_reproduction_artifact
from .metrics import energy_error_per_site, energy_expectation
from .native_noise_protocol import run_artifact_native_noise_protocol
from .protocol import run_iter_di_o_all
from .targets import (
    bars_and_stripes_state,
    exact_ground_state,
    heisenberg_grid_hamiltonian,
    random_real_mps,
    tfim_hamiltonian,
    xxz_hamiltonian,
)


def doctor_snapshot() -> dict[str, Any]:
    """Return a read-only numerical/convention snapshot."""

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "protocol": "Iter[D_i O_all]",
        "learning_rate": 0.6,
        "complex_dtype": "complex128",
        "qubit_order": "q0-most-significant",
        "layer": "N-1 ordered adjacent U(4), no terminal U(2)",
        "fidelity": "paper infidelity = 1 - abs(<target|prepared>)",
        "fractional_power": "principal complex-Schur branch; -1 -> +pi",
        "target_global_phase": "first maximum-modulus amplitude positive real",
    }


def _build_target(
    args: argparse.Namespace,
) -> tuple[np.ndarray, str, dict[str, Any], csr_matrix | None]:
    if args.target == "random-mps":
        random = random_real_mps(
            args.num_qubits, max_bond=args.max_bond, seed=args.seed
        )
        return random.state, "random_real_mps", {
            "seed": args.seed,
            "max_bond": args.max_bond,
            "bond_dimensions": list(random.bond_dimensions),
        }, None
    if args.target == "bas":
        rows = 2 if args.rows is None else args.rows
        columns = 6 if args.columns is None else args.columns
        state = bars_and_stripes_state(rows, columns)
        return state, "bars_and_stripes", {
            "rows": rows,
            "columns": columns,
            "flattening": "row-major",
        }, None
    if args.target == "heisenberg-grid":
        rows = 3 if args.rows is None else args.rows
        columns = 4 if args.columns is None else args.columns
        hamiltonian = heisenberg_grid_hamiltonian(
            rows=rows, columns=columns, coupling=args.coupling
        )
        ground = exact_ground_state(hamiltonian, tolerance=args.eigensolver_tolerance)
        return ground.state, "heisenberg_grid_ground_state", {
            "rows": rows,
            "columns": columns,
            "coupling": args.coupling,
            "site_order": "row-major",
            "energy": ground.energy,
            "gap": ground.gap,
            "eigen_residual": ground.residual,
            "solver": ground.solver,
            "first_excited_energy": ground.first_excited_energy,
        }, hamiltonian
    if args.target == "tfim":
        hamiltonian = tfim_hamiltonian(
            args.num_qubits,
            coupling_zz=args.coupling,
            field_x=args.field_x,
            field_z=args.field_z,
        )
        ground = exact_ground_state(hamiltonian, tolerance=args.eigensolver_tolerance)
        return ground.state, "tfim_ground_state", {
            "num_qubits": args.num_qubits,
            "coupling_zz": args.coupling,
            "field_x": args.field_x,
            "field_z": args.field_z,
            "energy": ground.energy,
            "first_excited_energy": ground.first_excited_energy,
            "gap": ground.gap,
            "eigen_residual": ground.residual,
            "solver": ground.solver,
        }, hamiltonian
    if args.target == "xxz":
        hamiltonian = xxz_hamiltonian(
            args.num_qubits,
            delta=args.delta,
            coupling=args.coupling,
            field_z=args.field_z,
        )
        ground = exact_ground_state(hamiltonian, tolerance=args.eigensolver_tolerance)
        return ground.state, "xxz_ground_state", {
            "num_qubits": args.num_qubits,
            "delta": args.delta,
            "coupling": args.coupling,
            "field_z": args.field_z,
            "energy": ground.energy,
            "first_excited_energy": ground.first_excited_energy,
            "gap": ground.gap,
            "eigen_residual": ground.residual,
            "solver": ground.solver,
        }, hamiltonian
    raise AssertionError(f"Unhandled target {args.target!r}.")


def _run(args: argparse.Namespace) -> int:
    target, name, metadata, hamiltonian = _build_target(args)
    result = run_iter_di_o_all(
        target,
        max_depth=args.depth,
        sweeps_per_depth=args.sweeps,
        learning_rate=args.learning_rate,
    )
    ground_energy = metadata.get("energy") if hamiltonian is not None else None
    manifest = save_reproduction_artifact(
        result,
        args.output,
        target_name=name,
        target_metadata=metadata,
        hamiltonian=hamiltonian,
        ground_energy=ground_energy,
    )
    verification = verify_reproduction_artifact(manifest.parent)
    summary = {
        "status": "completed",
        "manifest": str(manifest),
        "target": name,
        "depth": result.max_depth,
        "sweeps_per_depth": result.sweeps_per_depth,
        "total_gate_updates": result.total_gate_updates,
        "amplitude_fidelity": result.final_metrics.amplitude_fidelity,
        "paper_infidelity": result.final_metrics.paper_amplitude_infidelity,
        "squared_state_fidelity": result.final_metrics.squared_state_fidelity,
        "verification": verification,
    }
    if hamiltonian is not None and ground_energy is not None:
        summary["prepared_energy"] = energy_expectation(
            result.final_prepared_state, hamiltonian
        )
        summary["energy_error_per_site"] = energy_error_per_site(
            result.final_prepared_state, hamiltonian, ground_energy
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _smoke(args: argparse.Namespace) -> int:
    random = random_real_mps(4, max_bond=4, seed=args.seed)
    result = run_iter_di_o_all(
        random.state,
        max_depth=2,
        sweeps_per_depth=2,
        learning_rate=0.6,
    )
    manifest = save_reproduction_artifact(
        result,
        args.output,
        target_name="random_real_mps_smoke",
        target_metadata={"seed": args.seed, "num_qubits": 4, "max_bond": 4},
    )
    verification = verify_reproduction_artifact(manifest.parent)
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest": str(manifest),
                "total_gate_updates": result.total_gate_updates,
                "amplitude_fidelity": result.final_metrics.amplitude_fidelity,
                "verification": verification,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _native_noise(args: argparse.Namespace) -> int:
    exported = run_artifact_native_noise_protocol(
        args.artifact,
        args.output,
        physical_path=args.physical_path,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
        seed_simulator=args.seed_simulator,
        max_density_matrix_qubits=args.max_density_matrix_qubits,
        compute_minimum_eigenvalue=args.compute_minimum_eigenvalue,
        compilation_fidelity_atol=args.compilation_fidelity_atol,
        gate_error=not args.disable_gate_error,
        thermal_relaxation=not args.disable_thermal_relaxation,
        temperature_millikelvin=args.temperature_millikelvin,
        provenance={"command": "native-noise", "doctor": doctor_snapshot()},
    )
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
    noisy = manifest["noise_evaluation"]
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest": str(exported.manifest_path),
                "manifest_sha256": exported.manifest_sha256,
                "backend_snapshot_sha256": manifest["noise_protocol"][
                    "backend_snapshot_sha256"
                ],
                "physical_path": manifest["noise_protocol"]["physical_path"],
                "logical_target_fidelity": noisy["fidelity"]["logical_target"],
                "compilation_fidelity": noisy["fidelity"]["compilation"],
                "noisy_target_fidelity": noisy["fidelity"]["noisy_target"],
                "noisy_native_retention_fidelity": noisy["fidelity"][
                    "noisy_native_retention"
                ],
                "energy_error_per_site": noisy["energy"]["error_per_site"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI parser."""

    parser = argparse.ArgumentParser(prog="rudolph2023")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="print numerical conventions")
    doctor.set_defaults(handler=lambda _args: print(json.dumps(doctor_snapshot(), indent=2)) or 0)

    smoke = subparsers.add_parser("smoke", help="run an audited N=4 end-to-end smoke")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.set_defaults(handler=_smoke)

    run = subparsers.add_parser("run", help="run Iter[D_i O_all] on one target")
    run.add_argument(
        "--target",
        choices=("random-mps", "bas", "heisenberg-grid", "tfim", "xxz"),
        required=True,
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--depth", type=int, default=4)
    run.add_argument("--sweeps", type=int, default=10)
    run.add_argument("--learning-rate", type=float, default=0.6)
    run.add_argument("--num-qubits", type=int, default=12)
    run.add_argument("--max-bond", type=int, default=64)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--rows",
        type=int,
        default=None,
        help="target-specific default: BAS=2, Heisenberg grid=3",
    )
    run.add_argument(
        "--columns",
        type=int,
        default=None,
        help="target-specific default: BAS=6, Heisenberg grid=4",
    )
    run.add_argument("--coupling", type=float, default=1.0)
    run.add_argument("--field-x", type=float, default=1.0)
    run.add_argument("--field-z", type=float, default=0.0)
    run.add_argument("--delta", type=float, default=1.0)
    run.add_argument("--eigensolver-tolerance", type=float, default=1e-12)
    run.set_defaults(handler=_run)

    verify = subparsers.add_parser("verify-artifact", help="verify hashes and gate replay")
    verify.add_argument("artifact", type=Path)
    verify.set_defaults(
        handler=lambda args: print(
            json.dumps(verify_reproduction_artifact(args.artifact), indent=2)
        )
        or 0
    )

    noise = subparsers.add_parser(
        "native-noise",
        help="post-process one existing artifact with compact FakeWashingtonV2 noise",
    )
    noise.add_argument("--artifact", type=Path, required=True)
    noise.add_argument("--output", type=Path, required=True)
    noise.add_argument("--physical-path", type=int, nargs="+", default=None)
    noise.add_argument("--optimization-level", type=int, choices=(0, 1, 2, 3), default=0)
    noise.add_argument("--seed-transpiler", type=int, default=2026)
    noise.add_argument("--seed-simulator", type=int, default=2026)
    noise.add_argument("--max-density-matrix-qubits", type=int, default=12)
    noise.add_argument("--compilation-fidelity-atol", type=float, default=2e-10)
    noise.add_argument("--temperature-millikelvin", type=float, default=0.0)
    noise.add_argument("--disable-gate-error", action="store_true")
    noise.add_argument("--disable-thermal-relaxation", action="store_true")
    eigenvalue = noise.add_mutually_exclusive_group()
    eigenvalue.add_argument(
        "--compute-minimum-eigenvalue",
        dest="compute_minimum_eigenvalue",
        action="store_true",
    )
    eigenvalue.add_argument(
        "--skip-minimum-eigenvalue",
        dest="compute_minimum_eigenvalue",
        action="store_false",
    )
    noise.set_defaults(
        compute_minimum_eigenvalue=None,
        handler=_native_noise,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process status."""

    args = build_parser().parse_args(argv)
    return int(args.handler(args))


__all__ = ["build_parser", "doctor_snapshot", "main"]
