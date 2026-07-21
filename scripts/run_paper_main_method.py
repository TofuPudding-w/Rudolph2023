"""Run the paper's three N=12 targets for Iter[D_i O_all] only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rudolph2023.export import (
    save_reproduction_artifact,
    verify_reproduction_artifact,
)
from rudolph2023.metrics import energy_error_per_site, energy_expectation
from rudolph2023.protocol import run_iter_di_o_all
from rudolph2023.targets import (
    bars_and_stripes_state,
    exact_ground_state,
    heisenberg_grid_hamiltonian,
    random_real_mps,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--sweeps", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--random-seed", type=int, default=2023)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)

    heisenberg_hamiltonian = heisenberg_grid_hamiltonian(rows=3, columns=4)
    heisenberg_ground = exact_ground_state(
        heisenberg_hamiltonian, tolerance=1e-11
    )
    random_target = random_real_mps(12, max_bond=64, seed=args.random_seed)
    targets = (
        (
            "heisenberg_3x4",
            heisenberg_ground.state,
            {
                "rows": 3,
                "columns": 4,
                "site_order": "row-major",
                "energy": heisenberg_ground.energy,
                "gap": heisenberg_ground.gap,
                "eigen_residual": heisenberg_ground.residual,
            },
            heisenberg_hamiltonian,
            heisenberg_ground.energy,
        ),
        (
            "bas_2x6",
            bars_and_stripes_state(2, 6),
            {"rows": 2, "columns": 6, "flattening": "row-major", "support_size": 66},
            None,
            None,
        ),
        (
            "random_real_mps",
            random_target.state,
            {
                "seed": args.random_seed,
                "max_bond": 64,
                "bond_dimensions": list(random_target.bond_dimensions),
            },
            None,
            None,
        ),
    )

    rows = []
    for sweeps in args.sweeps:
        for target_name, target_state, metadata, hamiltonian, ground_energy in targets:
            started = time.perf_counter()
            result = run_iter_di_o_all(
                target_state,
                max_depth=args.depth,
                sweeps_per_depth=sweeps,
                learning_rate=0.6,
            )
            run_directory = args.output_root / f"{target_name}_K{args.depth}_T{sweeps}"
            manifest = save_reproduction_artifact(
                result,
                run_directory,
                target_name=target_name,
                target_metadata=metadata,
                hamiltonian=hamiltonian,
                ground_energy=ground_energy,
            )
            verification = verify_reproduction_artifact(manifest.parent)
            row = {
                "target": target_name,
                "depth": args.depth,
                "sweeps_per_depth": sweeps,
                "seconds": time.perf_counter() - started,
                "total_gate_updates": result.total_gate_updates,
                "amplitude_fidelity": result.final_metrics.amplitude_fidelity,
                "paper_infidelity": result.final_metrics.paper_amplitude_infidelity,
                "squared_state_fidelity": result.final_metrics.squared_state_fidelity,
                "manifest": str(manifest),
                "verification": verification,
            }
            if hamiltonian is not None and ground_energy is not None:
                row["prepared_energy"] = energy_expectation(
                    result.final_prepared_state, hamiltonian
                )
                row["energy_error_per_site"] = energy_error_per_site(
                    result.final_prepared_state, hamiltonian, ground_energy
                )
            rows.append(row)
            print(json.dumps(rows[-1], ensure_ascii=False))

    summary = args.output_root / "figure4_main_method_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "rudolph2023-figure4-main-method-v1",
                "protocol": "Iter[D_i O_all]",
                "runs": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
