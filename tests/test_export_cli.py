from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rudolph2023.cli import doctor_snapshot, main
from rudolph2023.export import (
    file_sha256,
    save_reproduction_artifact,
    semantic_array_sha256,
    verify_reproduction_artifact,
)
from rudolph2023.protocol import run_iter_di_o_all
from rudolph2023.targets import exact_ground_state, random_real_mps, tfim_hamiltonian


def test_semantic_hash_includes_shape_dtype_and_values() -> None:
    base = np.arange(8, dtype=np.float64)
    assert semantic_array_sha256(base) != semantic_array_sha256(base.astype(np.complex128))
    assert semantic_array_sha256(base) != semantic_array_sha256(base.reshape(2, 4))
    changed = base.copy()
    changed[-1] += 1
    assert semantic_array_sha256(base) != semantic_array_sha256(changed)


def test_artifact_save_verify_and_gate_replay(tmp_path: Path) -> None:
    target = random_real_mps(4, max_bond=4, seed=5).state
    result = run_iter_di_o_all(target, max_depth=2, sweeps_per_depth=1)
    output = tmp_path / "artifact"
    manifest_path = save_reproduction_artifact(
        result,
        output,
        target_name="random_real_mps",
        target_metadata={"seed": 5},
    )
    assert manifest_path.exists()
    verification = verify_reproduction_artifact(output)
    assert verification["status"] == "verified"
    assert verification["replay_error"] < 1e-12
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["gate_layout"]["shape"] == [2, 3, 4, 4]
    assert manifest["gate_layout"]["terminal_u2_folded"] is True
    assert manifest["total_gate_updates"] == 9
    with pytest.raises(FileExistsError):
        save_reproduction_artifact(result, output, target_name="duplicate")


def test_artifact_tampering_is_detected(tmp_path: Path) -> None:
    target = random_real_mps(3, max_bond=2, seed=6).state
    result = run_iter_di_o_all(target, max_depth=1, sweeps_per_depth=0)
    output = tmp_path / "artifact"
    save_reproduction_artifact(result, output, target_name="tamper_test")
    manifest = output / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_reproduction_artifact(output)


def test_artifact_semantic_metric_tampering_is_detected_after_rehash(
    tmp_path: Path,
) -> None:
    target = random_real_mps(3, max_bond=2, seed=61).state
    result = run_iter_di_o_all(target, max_depth=1, sweeps_per_depth=1)
    output = tmp_path / "semantic_tamper"
    save_reproduction_artifact(result, output, target_name="tamper_test")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_metrics"]["amplitude_fidelity"] = 0.123
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checksum_path = output / "SHA256SUMS.json"
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    checksums["manifest.json"] = file_sha256(manifest_path)
    checksum_path.write_text(json.dumps(checksums), encoding="utf-8")
    with pytest.raises(ValueError, match="amplitude_fidelity mismatch"):
        verify_reproduction_artifact(output)


def test_ground_state_artifact_stores_and_recomputes_energy(tmp_path: Path) -> None:
    hamiltonian = tfim_hamiltonian(3, coupling_zz=1.0, field_x=0.7)
    ground = exact_ground_state(hamiltonian)
    result = run_iter_di_o_all(ground.state, max_depth=1, sweeps_per_depth=1)
    output = tmp_path / "energy_artifact"
    manifest_path = save_reproduction_artifact(
        result,
        output,
        target_name="tfim_ground_state",
        hamiltonian=hamiltonian,
        ground_energy=ground.energy,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["energy_evaluation"]["storage"] == "csr"
    assert manifest["energy_evaluation"]["energy_error_per_site"] >= 0.0
    verification = verify_reproduction_artifact(output)
    assert verification["energy_verified"] is True


def test_doctor_and_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    snapshot = doctor_snapshot()
    assert snapshot["protocol"] == "Iter[D_i O_all]"
    assert snapshot["learning_rate"] == 0.6
    output = tmp_path / "cli_smoke"
    assert main(["smoke", "--seed", "7", "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "completed"
    assert printed["verification"]["status"] == "verified"
    assert printed["total_gate_updates"] == 18


def test_cli_bas_uses_paper_two_by_six_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "bas_default"
    assert main(
        [
            "run",
            "--target",
            "bas",
            "--output",
            str(output),
            "--depth",
            "0",
            "--sweeps",
            "0",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["verification"]["status"] == "verified"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_metadata"]["rows"] == 2
    assert manifest["target_metadata"]["columns"] == 6
    assert manifest["num_qubits"] == 12


@pytest.mark.parametrize(
    ("target", "extra"),
    [
        ("random-mps", ["--num-qubits", "3", "--max-bond", "2", "--seed", "1"]),
        ("bas", ["--rows", "1", "--columns", "3"]),
        ("heisenberg-grid", ["--rows", "1", "--columns", "2"]),
        ("tfim", ["--num-qubits", "2"]),
        ("xxz", ["--num-qubits", "2", "--delta", "0.5"]),
    ],
)
def test_cli_run_builds_every_target_branch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
    extra: list[str],
) -> None:
    output = tmp_path / target
    argv = [
        "run",
        "--target",
        target,
        "--output",
        str(output),
        "--depth",
        "0",
        "--sweeps",
        "0",
        *extra,
    ]
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "completed"
    assert verify_reproduction_artifact(output)["status"] == "verified"
    if target in {"heisenberg-grid", "tfim", "xxz"}:
        assert summary["energy_error_per_site"] >= 0.0
        assert summary["verification"]["energy_verified"] is True


def test_cli_doctor_and_verify_subcommands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor"]) == 0
    assert json.loads(capsys.readouterr().out)["protocol"] == "Iter[D_i O_all]"
    output = tmp_path / "verify_cli"
    assert main(["smoke", "--output", str(output)]) == 0
    capsys.readouterr()
    assert main(["verify-artifact", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
