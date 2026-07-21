from __future__ import annotations

import numpy as np
import pytest

from rudolph2023.protocol import run_iter_di_o_all
from rudolph2023.targets import (
    bars_and_stripes_state,
    exact_ground_state,
    heisenberg_grid_hamiltonian,
    random_real_mps,
)


def paper_heisenberg_target() -> np.ndarray:
    ground = exact_ground_state(
        heisenberg_grid_hamiltonian(rows=3, columns=4), tolerance=1e-11
    )
    # Independent high-precision regression value for the 17-edge OBC S=sigma/2 model.
    assert ground.energy == pytest.approx(-6.691680193514956, abs=2e-10)
    assert ground.residual < 2e-10
    return ground.state


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize(
    ("name", "expected_final_amplitude", "target_factory"),
    [
        (
            "heisenberg_3x4",
            0.971655798233501,
            paper_heisenberg_target,
        ),
        ("bas_2x6", 0.996314482709296, lambda: bars_and_stripes_state(2, 6)),
        (
            "random_mps_seed2023",
            0.8724798693348269,
            lambda: random_real_mps(12, max_bond=64, seed=2023).state,
        ),
    ],
)
def test_figure4_main_method_n12_k8_t10_regression(
    name: str,
    expected_final_amplitude: float,
    target_factory,
) -> None:
    target = target_factory()
    result = run_iter_di_o_all(target, max_depth=8, sweeps_per_depth=10)
    assert result.total_gate_updates == 3960, name
    assert result.final_circuit.gate_array().shape == (8, 11, 4, 4)
    assert result.final_metrics.amplitude_fidelity == pytest.approx(
        expected_final_amplitude,
        abs=2e-3,  # allows degenerate-SVD/BLAS gauge differences across platforms
    )
    amplitudes = [record.optimized_metrics.amplitude_fidelity for record in result.depth_records]
    # This is a fixed-target regression observation, not a general theorem for r=0.6.
    assert np.all(np.diff(amplitudes) >= -1e-10)
    assert max(
        sweep.max_environment_identity_error
        for depth in result.depth_records
        for sweep in depth.optimization.sweeps
    ) < 1e-10
    assert max(
        depth.residual_overlap_identity_error for depth in result.depth_records
    ) < 1e-10
