"""Paper-native and cross-paper state/energy metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .conventions import infer_num_qubits, validate_normalized_statevector


@dataclass(frozen=True, slots=True)
class FidelityMetrics:
    """All fidelity conventions needed to avoid cross-paper ambiguity."""

    overlap: complex
    amplitude_fidelity: float
    paper_amplitude_infidelity: float
    squared_state_fidelity: float
    squared_infidelity: float


def fidelity_metrics(target: ArrayLike, candidate: ArrayLike) -> FidelityMetrics:
    """Return amplitude and squared fidelities for two normalized pure states."""

    target_state = validate_normalized_statevector(target)
    candidate_state = validate_normalized_statevector(candidate)
    if target_state.shape != candidate_state.shape:
        raise ValueError("Target and candidate statevector shapes differ.")
    # The validator deliberately permits machine-scale norm drift.  Remove it
    # before taking the overlap so a pair of physically identical states can
    # never produce a fidelity above one or a negative infidelity.
    target_state = target_state / np.linalg.norm(target_state)
    candidate_state = candidate_state / np.linalg.norm(candidate_state)
    overlap = complex(np.vdot(target_state, candidate_state))
    amplitude = float(np.clip(abs(overlap), 0.0, 1.0))
    squared = float(np.clip(amplitude**2, 0.0, 1.0))
    return FidelityMetrics(
        overlap=overlap,
        amplitude_fidelity=amplitude,
        paper_amplitude_infidelity=1.0 - amplitude,
        squared_state_fidelity=squared,
        squared_infidelity=1.0 - squared,
    )


def energy_expectation(state: ArrayLike, hamiltonian: ArrayLike | Any) -> float:
    """Return a real normalized energy expectation for dense or sparse H."""

    vector = validate_normalized_statevector(state)
    dimension = vector.size
    if getattr(hamiltonian, "shape", None) != (dimension, dimension):
        raise ValueError("Hamiltonian shape does not match state dimension.")
    evolved = hamiltonian @ vector
    if not np.all(np.isfinite(evolved)):
        raise FloatingPointError("Hamiltonian action produced non-finite values.")
    value = complex(np.vdot(vector, evolved))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise FloatingPointError("Energy expectation is non-finite.")
    scale = max(1.0, abs(value.real))
    if abs(value.imag) > 1e-11 * scale:
        raise FloatingPointError(f"Energy has a nonphysical imaginary part: {value.imag:.3e}.")
    return float(value.real)


def energy_error_per_site(
    state: ArrayLike,
    hamiltonian: ArrayLike | Any,
    ground_energy: float,
) -> float:
    """Return ``(E(state)-E0)/N`` with tiny negative roundoff clipped to zero."""

    if not isinstance(ground_energy, int | float | np.floating):
        raise TypeError("ground_energy must be real.")
    ground_energy = float(ground_energy)
    if not math.isfinite(ground_energy):
        raise ValueError("ground_energy must be finite.")
    num_qubits = infer_num_qubits(state)
    error = (energy_expectation(state, hamiltonian) - ground_energy) / num_qubits
    if error < 0.0 and error >= -1e-11:
        return 0.0
    if error < -1e-11:
        raise ArithmeticError(
            "Energy lies below the supplied ground energy; conventions or E0 are inconsistent."
        )
    return float(error)


def entanglement_entropy(state: ArrayLike, cut: int) -> float:
    """Return von Neumann entropy from squared Schmidt coefficients."""

    vector = validate_normalized_statevector(state)
    num_qubits = infer_num_qubits(vector)
    if isinstance(cut, bool) or not isinstance(cut, int | np.integer):
        raise TypeError("cut must be an integer.")
    cut = int(cut)
    if not 1 <= cut < num_qubits:
        raise ValueError("cut must lie strictly inside the chain.")
    singular_values = np.linalg.svd(
        vector.reshape(1 << cut, 1 << (num_qubits - cut)),
        compute_uv=False,
    )
    probabilities = np.square(singular_values)
    probabilities = probabilities[probabilities > 0.0]
    return float(-np.sum(probabilities * np.log(probabilities)))


__all__ = [
    "FidelityMetrics",
    "energy_error_per_site",
    "energy_expectation",
    "entanglement_entropy",
    "fidelity_metrics",
]
