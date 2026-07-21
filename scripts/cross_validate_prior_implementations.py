"""Cross-check D_i against Ran_2020 and environment/polar against Ben-Dov."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
import types
from pathlib import Path

import numpy as np

from rudolph2023.analytic import analytic_step
from rudolph2023.conventions import normalized_statevector
from rudolph2023.linalg import polar_optimal_unitary
from rudolph2023.optimizer import gate_environment


def _package_stub(name: str, source_directory: Path) -> None:
    package = types.ModuleType(name)
    package.__path__ = [str(source_directory)]  # type: ignore[attr-defined]
    sys.modules[name] = package


def _python_tree_sha256(source_directory: Path) -> str:
    records = {}
    for path in sorted(source_directory.rglob("*.py")):
        relative = path.relative_to(source_directory).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records[relative] = digest
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ran-root", type=Path, required=True)
    parser.add_argument("--bendov-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _package_stub("ran2020", args.ran_root / "src" / "ran2020")
    ran_mps = importlib.import_module("ran2020.mps")
    ran_truncation = importlib.import_module("ran2020.truncation")
    ran_mpd = importlib.import_module("ran2020.mpd")

    analytic_worst = {"compressed": 0.0, "residual": 0.0, "prepared": 0.0}
    for seed in range(20):
        rng = np.random.default_rng(seed)
        state = normalized_statevector(
            rng.normal(size=64) + 1j * rng.normal(size=64)
        )
        ours = analytic_step(state)
        ran_compressed = ran_truncation.truncate_mps(
            ran_mps.dense_to_mps(state), max_bond=2, normalize=True
        )
        ran_layer = ran_mpd.build_single_layer_mpd(ran_compressed.mps)
        analytic_worst["compressed"] = max(
            analytic_worst["compressed"],
            float(np.max(np.abs(ours.compressed_state - ran_compressed.mps.to_dense()))),
        )
        analytic_worst["residual"] = max(
            analytic_worst["residual"],
            float(np.max(np.abs(ours.next_residual - ran_layer.disentangle_dense(state)))),
        )
        analytic_worst["prepared"] = max(
            analytic_worst["prepared"],
            float(np.max(np.abs(ours.layer.prepare() - ran_layer.prepare_dense()))),
        )

    # Avoid executing Ben-Dov's package __init__, which imports optional noise components.
    _package_stub("bendov2024", args.bendov_root / "src" / "bendov2024")
    bendov_optimizer = importlib.import_module("bendov2024.optimizer")
    bendov_polar = importlib.import_module("bendov2024.polar")
    environment_worst = 0.0
    polar_worst = 0.0
    for seed in range(50):
        rng = np.random.default_rng(seed)
        prefix = normalized_statevector(
            rng.normal(size=16) + 1j * rng.normal(size=16)
        )
        suffix = normalized_statevector(
            rng.normal(size=16) + 1j * rng.normal(size=16)
        )
        left_site = seed % 3
        ours_environment = gate_environment(
            prefix, suffix, left_site=left_site, num_qubits=4
        )
        theirs_environment = bendov_optimizer.gate_environment(
            prefix, suffix, sites=(left_site, left_site + 1), num_qubits=4
        )
        ours_polar = polar_optimal_unitary(ours_environment).unitary
        theirs_polar = bendov_polar.polar_decomposition(theirs_environment).unitary
        environment_worst = max(
            environment_worst,
            float(np.max(np.abs(ours_environment - theirs_environment))),
        )
        polar_worst = max(
            polar_worst,
            float(np.max(np.abs(ours_polar - theirs_polar))),
        )

    report = {
        "schema_version": "rudolph2023-cross-project-validation-v1",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source_provenance": {
            "rudolph2023": _python_tree_sha256(
                Path(__file__).parents[1] / "src" / "rudolph2023"
            ),
            "ran2020": _python_tree_sha256(args.ran_root / "src" / "ran2020"),
            "bendov2024": _python_tree_sha256(
                args.bendov_root / "src" / "bendov2024"
            ),
        },
        "ran2020": {"seeds": 20, "worst_errors": analytic_worst},
        "bendov2024": {
            "seeds": 50,
            "worst_environment_error": environment_worst,
            "worst_polar_error": polar_worst,
        },
        "status": "passed"
        if max(*analytic_worst.values(), environment_worst, polar_worst) < 1e-11
        else "failed",
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
