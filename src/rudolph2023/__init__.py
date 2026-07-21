"""Rudolph et al. main-method reproduction."""

from .analytic import analytic_step, chi2_mps_to_layer
from .circuit import StaircaseCircuit, StaircaseLayer
from .metrics import fidelity_metrics
from .optimizer import forward_sweep, optimize_all
from .protocol import run_iter_di_o_all

__version__ = "0.1.0"

__all__ = [
    "StaircaseCircuit",
    "StaircaseLayer",
    "analytic_step",
    "chi2_mps_to_layer",
    "fidelity_metrics",
    "forward_sweep",
    "optimize_all",
    "run_iter_di_o_all",
]
