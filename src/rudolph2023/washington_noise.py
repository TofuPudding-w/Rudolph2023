"""Auditable native-gate noise evaluation on a FakeWashingtonV2 sub-chain.

The Washington backend has 127 physical qubits.  A density-matrix simulation
of a circuit retaining that full register is impossible for the problem sizes
used by this reproduction.  This module therefore performs a *calibration
preserving compaction*:

1. select (or validate) a simple physical path in the backend ``Target``;
2. copy only the selected qubit and native-instruction properties;
3. re-index the path to a compact ``0, ..., N-1`` target;
4. derive an Aer ``NoiseModel`` from that compact BackendV2 view; and
5. compile and simulate only the compact native circuit.

All Qiskit imports are lazy.  Importing :mod:`rudolph2023.washington_noise`
therefore does not make Qiskit a mandatory dependency of the noiseless paper
reproduction.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import numbers
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]

COMPACT_DEVICE_SCHEMA = "rudolph2023-compact-backend-noise-v1"
NOISY_RESULT_SCHEMA = "rudolph2023-native-density-matrix-v1"
DEFAULT_NATIVE_OPERATIONS = ("rz", "sx", "x", "cx")


class QiskitNoiseDependencyError(ImportError):
    """Raised when an explicitly requested Qiskit noise operation is unavailable."""


def _qiskit_dependencies() -> dict[str, Any]:
    """Import the Qiskit objects used by this module, with one clear error."""

    try:
        from qiskit.quantum_info import Statevector
        from qiskit.transpiler import InstructionProperties, Target
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
    except (ImportError, ModuleNotFoundError) as exc:
        raise QiskitNoiseDependencyError(
            "Qiskit noise evaluation requires compatible installations of qiskit and qiskit-aer."
        ) from exc
    return {
        "AerSimulator": AerSimulator,
        "InstructionProperties": InstructionProperties,
        "NoiseModel": NoiseModel,
        "Statevector": Statevector,
        "Target": Target,
        "generate_preset_pass_manager": generate_preset_pass_manager,
    }


def load_fake_washington_v2() -> Any:
    """Return ``FakeWashingtonV2`` from a supported public fake-provider path.

    Qiskit moved fake backends between packages over its release history.  The
    provider used is recorded later in the device provenance; no silent generic
    backend is substituted when Washington is unavailable.
    """

    errors: list[BaseException] = []
    try:
        from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2

        return FakeWashingtonV2()
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        errors.append(exc)
    try:
        from qiskit.providers.fake_provider import FakeWashingtonV2

        return FakeWashingtonV2()
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        errors.append(exc)
    raise QiskitNoiseDependencyError(
        "FakeWashingtonV2 is not available. Install a qiskit-ibm-runtime "
        "version that supplies qiskit_ibm_runtime.fake_provider.FakeWashingtonV2."
    ) from errors[-1]


def _backend_name(backend: Any) -> str:
    value = getattr(backend, "name", type(backend).__name__)
    if callable(value):
        value = value()
    return str(value)


def _backend_num_qubits(backend: Any) -> int:
    value = getattr(backend, "num_qubits", None)
    if value is None:
        value = getattr(getattr(backend, "target", None), "num_qubits", None)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("backend must expose an integer num_qubits through BackendV2.")
    result = int(value)
    if result < 1:
        raise ValueError("backend.num_qubits must be positive.")
    return result


def _target_two_qubit_edges(target: Any, operation: str) -> set[tuple[int, int]]:
    if operation not in tuple(target.operation_names):
        raise ValueError(f"Backend Target does not support {operation!r}.")
    edges: set[tuple[int, int]] = set()
    for qargs in target[operation]:
        if qargs is not None and len(qargs) == 2:
            edges.add((int(qargs[0]), int(qargs[1])))
    if not edges:
        raise ValueError(f"Backend Target contains no two-qubit {operation!r} edges.")
    return edges


def validate_physical_path(
    backend: Any,
    physical_path: tuple[int, ...] | list[int],
    *,
    two_qubit_gate: str = "cx",
) -> tuple[int, ...]:
    """Validate a unique, contiguous physical path under an undirected edge test.

    Directional native support is retained in the compact Target.  The path
    itself only requires at least one supported direction for each adjacent
    pair; Qiskit's direction-correction pass may synthesize the opposite CX
    direction without routing or SWAP insertion.
    """

    if not isinstance(two_qubit_gate, str) or not two_qubit_gate:
        raise TypeError("two_qubit_gate must be a nonempty instruction name.")
    try:
        raw_path = tuple(physical_path)
    except TypeError as exc:
        raise TypeError("physical_path must be an iterable of integer qubits.") from exc
    if any(
        isinstance(qubit, bool) or not isinstance(qubit, numbers.Integral)
        for qubit in raw_path
    ):
        raise TypeError("physical_path must contain integer qubit indices.")
    path = tuple(int(qubit) for qubit in raw_path)
    if not path:
        raise ValueError("physical_path must contain at least one physical qubit.")
    if len(set(path)) != len(path):
        raise ValueError("physical_path must not repeat a physical qubit.")
    num_backend_qubits = _backend_num_qubits(backend)
    if any(qubit < 0 or qubit >= num_backend_qubits for qubit in path):
        raise ValueError(
            f"physical_path={path} leaves the backend range [0, {num_backend_qubits - 1}]."
        )
    if len(path) == 1:
        return path
    directed_edges = _target_two_qubit_edges(backend.target, two_qubit_gate)
    undirected_edges = {(min(left, right), max(left, right)) for left, right in directed_edges}
    for left, right in zip(path[:-1], path[1:], strict=True):
        edge = (min(left, right), max(left, right))
        if edge not in undirected_edges:
            raise ValueError(
                f"physical_path is not contiguous: {two_qubit_gate}{(left, right)} "
                "is unsupported in either direction."
            )
    return path


def select_physical_path(
    backend: Any,
    num_qubits: int,
    *,
    two_qubit_gate: str = "cx",
) -> tuple[int, ...]:
    """Return the lexicographically first simple native-gate path of length ``N``.

    This is deliberately deterministic rather than calibration-optimizing.  A
    formal comparison should normally pass an explicit path after inspecting
    the returned calibration audit.  Auto-selection remains reproducible for a
    frozen backend Target and its hash is included in provenance.
    """

    if isinstance(num_qubits, bool) or not isinstance(num_qubits, numbers.Integral):
        raise TypeError("num_qubits must be an integer.")
    num_qubits = int(num_qubits)
    available = _backend_num_qubits(backend)
    if not 1 <= num_qubits <= available:
        raise ValueError(f"num_qubits must lie in [1, {available}], got {num_qubits}.")
    if num_qubits == 1:
        return (0,)

    directed_edges = _target_two_qubit_edges(backend.target, two_qubit_gate)
    adjacency: dict[int, set[int]] = {qubit: set() for qubit in range(available)}
    for left, right in directed_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    def extend(path: tuple[int, ...]) -> tuple[int, ...] | None:
        if len(path) == num_qubits:
            return path
        for neighbour in sorted(adjacency[path[-1]]):
            if neighbour not in path:
                result = extend((*path, neighbour))
                if result is not None:
                    return result
        return None

    for start in range(available):
        result = extend((start,))
        if result is not None:
            return result
    raise ValueError(
        f"No simple {two_qubit_gate!r} path of length {num_qubits} exists "
        f"on backend {_backend_name(backend)!r}."
    )


def _finite_float_or_none(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}={result!r} in backend Target.")
    return result


@dataclass(frozen=True, slots=True)
class QubitCalibrationAudit:
    compact_qubit: int
    physical_qubit: int
    t1_seconds: float | None
    t2_seconds: float | None
    frequency_hz: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "compact_qubit": self.compact_qubit,
            "physical_qubit": self.physical_qubit,
            "t1_seconds": self.t1_seconds,
            "t2_seconds": self.t2_seconds,
            "frequency_hz": self.frequency_hz,
        }


@dataclass(frozen=True, slots=True)
class InstructionCalibrationAudit:
    name: str
    compact_qubits: tuple[int, ...]
    physical_qubits: tuple[int, ...]
    duration_seconds: float | None
    error: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "compact_qubits": list(self.compact_qubits),
            "physical_qubits": list(self.physical_qubits),
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


class _CompactBackendV2View:
    """The minimal public BackendV2 surface consumed by NoiseModel.from_backend."""

    version = 2

    def __init__(self, target: Any, name: str) -> None:
        self.target = target
        self.name = name

    @property
    def operation_names(self) -> list[str]:
        return list(self.target.operation_names)

    @property
    def dt(self) -> float | None:
        return self.target.dt

    @property
    def num_qubits(self) -> int:
        return int(self.target.num_qubits)


@dataclass(frozen=True, slots=True)
class CompactWashingtonDevice:
    """Compact calibration snapshot and its derived Aer noise model."""

    source_backend_name: str
    source_backend_class: str
    source_num_qubits: int
    physical_path: tuple[int, ...]
    two_qubit_gate: str
    target: Any = field(repr=False, compare=False)
    noise_model: Any = field(repr=False, compare=False)
    qubit_calibrations: tuple[QubitCalibrationAudit, ...]
    instruction_calibrations: tuple[InstructionCalibrationAudit, ...]
    include_gate_error: bool
    include_thermal_relaxation: bool
    include_readout_error: bool
    temperature_millikelvin: float
    snapshot_sha256: str

    @property
    def num_qubits(self) -> int:
        return len(self.physical_path)

    @property
    def compact_to_physical(self) -> tuple[int, ...]:
        return self.physical_path

    @property
    def physical_to_compact(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted((physical, compact) for compact, physical in enumerate(self.physical_path))
        )

    @property
    def project_site_to_physical_q0_msb(self) -> tuple[int, ...]:
        """Map each project q0-MSB site to its Washington physical qubit.

        The convention-safe Qiskit adapter maps project site ``i`` to compact
        Qiskit qubit ``N - 1 - i``.  Since compact qubit ``j`` maps to
        ``physical_path[j]``, this audit mapping is the reversed path.
        """

        return tuple(reversed(self.physical_path))

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPACT_DEVICE_SCHEMA,
            "source_backend_name": self.source_backend_name,
            "source_backend_class": self.source_backend_class,
            "source_num_qubits": self.source_num_qubits,
            "physical_path": list(self.physical_path),
            "compact_to_physical": list(self.compact_to_physical),
            "physical_to_compact": [list(pair) for pair in self.physical_to_compact],
            "project_site_to_physical_q0_msb": list(self.project_site_to_physical_q0_msb),
            "project_to_qiskit_convention": "site_i_to_compact_qubit_N_minus_1_minus_i",
            "two_qubit_gate": self.two_qubit_gate,
            "target_operation_names": sorted(self.target.operation_names),
            "noise_options": {
                "gate_error": self.include_gate_error,
                "thermal_relaxation": self.include_thermal_relaxation,
                "readout_error": self.include_readout_error,
                "temperature_millikelvin": self.temperature_millikelvin,
            },
            "qubit_calibrations": [item.as_dict() for item in self.qubit_calibrations],
            "instruction_calibrations": [item.as_dict() for item in self.instruction_calibrations],
            "snapshot_sha256": self.snapshot_sha256,
        }


def _compact_target_and_audit(
    backend: Any,
    path: tuple[int, ...],
) -> tuple[Any, tuple[QubitCalibrationAudit, ...], tuple[InstructionCalibrationAudit, ...]]:
    dependencies = _qiskit_dependencies()
    InstructionProperties = dependencies["InstructionProperties"]
    Target = dependencies["Target"]
    source = backend.target
    source_qubit_properties = source.qubit_properties
    if source_qubit_properties is None:
        compact_qubit_properties = None
        qubit_audit = tuple(
            QubitCalibrationAudit(index, physical, None, None, None)
            for index, physical in enumerate(path)
        )
    else:
        compact_qubit_properties = [
            copy.deepcopy(source_qubit_properties[physical]) for physical in path
        ]
        qubit_audit = tuple(
            QubitCalibrationAudit(
                compact_qubit=index,
                physical_qubit=physical,
                t1_seconds=_finite_float_or_none(
                    getattr(source_qubit_properties[physical], "t1", None), label="T1"
                ),
                t2_seconds=_finite_float_or_none(
                    getattr(source_qubit_properties[physical], "t2", None), label="T2"
                ),
                frequency_hz=_finite_float_or_none(
                    getattr(source_qubit_properties[physical], "frequency", None),
                    label="frequency",
                ),
            )
            for index, physical in enumerate(path)
        )

    compact = Target(
        description=f"Compact path extracted from {_backend_name(backend)}",
        num_qubits=len(path),
        dt=source.dt,
        granularity=getattr(source, "granularity", 1),
        min_length=getattr(source, "min_length", 1),
        pulse_alignment=getattr(source, "pulse_alignment", 1),
        acquire_alignment=getattr(source, "acquire_alignment", 1),
        qubit_properties=compact_qubit_properties,
    )
    physical_to_compact = {physical: compact for compact, physical in enumerate(path)}
    path_edges = {frozenset((path[index], path[index + 1])) for index in range(len(path) - 1)}
    instruction_audit: list[InstructionCalibrationAudit] = []

    for name in source.operation_names:
        copied_properties: dict[tuple[int, ...], Any] = {}
        for physical_qargs, source_properties in source[name].items():
            if physical_qargs is None:
                continue
            physical_qargs = tuple(int(qubit) for qubit in physical_qargs)
            if len(physical_qargs) == 1:
                if physical_qargs[0] not in physical_to_compact:
                    continue
            elif len(physical_qargs) == 2:
                if any(qubit not in physical_to_compact for qubit in physical_qargs):
                    continue
                if frozenset(physical_qargs) not in path_edges:
                    continue
            else:
                # The Rudolph native-noise protocol uses only one- and two-qubit gates.
                continue
            compact_qargs = tuple(physical_to_compact[qubit] for qubit in physical_qargs)
            duration = _finite_float_or_none(
                getattr(source_properties, "duration", None), label=f"{name} duration"
            )
            error = _finite_float_or_none(
                getattr(source_properties, "error", None), label=f"{name} error"
            )
            # Pulse calibrations are intentionally not copied: the compact target
            # is a simulator/audit target, while duration and error are precisely
            # the public fields used to derive the Aer device noise model.
            copied_properties[compact_qargs] = InstructionProperties(
                duration=duration,
                error=error,
            )
            instruction_audit.append(
                InstructionCalibrationAudit(
                    name=str(name),
                    compact_qubits=compact_qargs,
                    physical_qubits=physical_qargs,
                    duration_seconds=duration,
                    error=error,
                )
            )
        if copied_properties:
            compact.add_instruction(
                copy.deepcopy(source.operation_from_name(name)),
                copied_properties,
                name=name,
            )

    instruction_audit.sort(key=lambda item: (item.name, item.compact_qubits))
    return compact, qubit_audit, tuple(instruction_audit)


def _canonical_snapshot_payload(
    *,
    backend: Any,
    path: tuple[int, ...],
    two_qubit_gate: str,
    target: Any,
    qubits: tuple[QubitCalibrationAudit, ...],
    instructions: tuple[InstructionCalibrationAudit, ...],
    gate_error: bool,
    thermal_relaxation: bool,
    readout_error: bool,
    temperature: float,
) -> dict[str, Any]:
    return {
        "schema_version": COMPACT_DEVICE_SCHEMA,
        "source_backend_name": _backend_name(backend),
        "source_backend_class": (f"{type(backend).__module__}.{type(backend).__qualname__}"),
        "source_num_qubits": _backend_num_qubits(backend),
        "physical_path": list(path),
        "two_qubit_gate": two_qubit_gate,
        "target_operation_names": sorted(target.operation_names),
        "noise_options": {
            "gate_error": gate_error,
            "thermal_relaxation": thermal_relaxation,
            "readout_error": readout_error,
            "temperature_millikelvin": temperature,
        },
        "qubit_calibrations": [item.as_dict() for item in qubits],
        "instruction_calibrations": [item.as_dict() for item in instructions],
    }


def build_compact_washington_device(
    num_qubits: int | None = None,
    *,
    backend: Any | None = None,
    physical_path: tuple[int, ...] | list[int] | None = None,
    two_qubit_gate: str = "cx",
    gate_error: bool = True,
    thermal_relaxation: bool = True,
    readout_error: bool = False,
    temperature_millikelvin: float = 0.0,
) -> CompactWashingtonDevice:
    """Extract an auditable compact Washington path and build its Aer noise.

    ``backend`` is injectable to permit independent tests with a small
    BackendV2.  If omitted, the function loads ``FakeWashingtonV2`` and never
    silently falls back to random calibration data.
    """

    dependencies = _qiskit_dependencies()
    if backend is None:
        backend = load_fake_washington_v2()
    if not hasattr(backend, "target"):
        raise TypeError("backend must implement the BackendV2 Target interface.")
    if not isinstance(two_qubit_gate, str) or not two_qubit_gate:
        raise TypeError("two_qubit_gate must be a nonempty instruction name.")
    for name, value in {
        "gate_error": gate_error,
        "thermal_relaxation": thermal_relaxation,
        "readout_error": readout_error,
    }.items():
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool.")
    if (
        isinstance(temperature_millikelvin, bool)
        or not isinstance(temperature_millikelvin, numbers.Real)
    ):
        raise TypeError("temperature_millikelvin must be a real scalar.")
    if not math.isfinite(float(temperature_millikelvin)):
        raise ValueError("temperature_millikelvin must be finite.")
    temperature_millikelvin = float(temperature_millikelvin)
    if temperature_millikelvin < 0.0:
        raise ValueError("temperature_millikelvin must be nonnegative.")

    if physical_path is None:
        if num_qubits is None:
            raise ValueError("num_qubits is required when physical_path is omitted.")
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, numbers.Integral):
            raise TypeError("num_qubits must be an integer.")
        path = select_physical_path(backend, num_qubits, two_qubit_gate=two_qubit_gate)
    else:
        path = validate_physical_path(backend, physical_path, two_qubit_gate=two_qubit_gate)
        if num_qubits is not None and (
            isinstance(num_qubits, bool) or not isinstance(num_qubits, numbers.Integral)
        ):
            raise TypeError("num_qubits must be an integer when it is provided.")
        if num_qubits is not None and int(num_qubits) != len(path):
            raise ValueError(
                f"num_qubits={num_qubits} disagrees with physical_path length {len(path)}."
            )

    compact_target, qubit_audit, instruction_audit = _compact_target_and_audit(backend, path)
    if two_qubit_gate not in compact_target.operation_names and len(path) > 1:
        raise ValueError(f"Compact target unexpectedly contains no {two_qubit_gate!r} instruction.")
    view = _CompactBackendV2View(
        compact_target,
        f"compact-{_backend_name(backend)}-{','.join(map(str, path))}",
    )
    noise_model = dependencies["NoiseModel"].from_backend(
        view,
        gate_error=bool(gate_error),
        thermal_relaxation=bool(thermal_relaxation),
        readout_error=bool(readout_error),
        temperature=temperature_millikelvin,
    )
    payload = _canonical_snapshot_payload(
        backend=backend,
        path=path,
        two_qubit_gate=two_qubit_gate,
        target=compact_target,
        qubits=qubit_audit,
        instructions=instruction_audit,
        gate_error=bool(gate_error),
        thermal_relaxation=bool(thermal_relaxation),
        readout_error=bool(readout_error),
        temperature=temperature_millikelvin,
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    return CompactWashingtonDevice(
        source_backend_name=_backend_name(backend),
        source_backend_class=f"{type(backend).__module__}.{type(backend).__qualname__}",
        source_num_qubits=_backend_num_qubits(backend),
        physical_path=path,
        two_qubit_gate=two_qubit_gate,
        target=compact_target,
        noise_model=noise_model,
        qubit_calibrations=qubit_audit,
        instruction_calibrations=instruction_audit,
        include_gate_error=bool(gate_error),
        include_thermal_relaxation=bool(thermal_relaxation),
        include_readout_error=bool(readout_error),
        temperature_millikelvin=temperature_millikelvin,
        snapshot_sha256=digest,
    )


@dataclass(frozen=True, slots=True)
class NativeCompilation:
    """A native compact circuit plus deterministic compilation diagnostics."""

    circuit: Any = field(repr=False, compare=False)
    gate_counts: tuple[tuple[str, int], ...]
    depth: int
    size: int
    swap_count: int
    unitary_count: int
    initial_index_layout: tuple[int, ...]
    final_index_layout: tuple[int, ...]
    optimization_level: int
    seed_transpiler: int

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "gate_counts": dict(self.gate_counts),
            "depth": self.depth,
            "size": self.size,
            "swap_count": self.swap_count,
            "unitary_count": self.unitary_count,
            "initial_index_layout": list(self.initial_index_layout),
            "final_index_layout": list(self.final_index_layout),
            "optimization_level": self.optimization_level,
            "seed_transpiler": self.seed_transpiler,
        }


def compile_to_native_target(
    logical_circuit: Any,
    device: CompactWashingtonDevice,
    *,
    optimization_level: int = 0,
    seed_transpiler: int = 2026,
    enforce_no_swaps: bool = True,
) -> NativeCompilation:
    """Exactly transpile a compact logical circuit to the selected native Target."""

    dependencies = _qiskit_dependencies()
    from qiskit import QuantumCircuit

    if not isinstance(logical_circuit, QuantumCircuit):
        raise TypeError("logical_circuit must be a qiskit.QuantumCircuit.")
    if not isinstance(device, CompactWashingtonDevice):
        raise TypeError("device must be a CompactWashingtonDevice.")
    if logical_circuit.num_qubits != device.num_qubits:
        raise ValueError(
            f"logical circuit has {logical_circuit.num_qubits} qubits but the "
            f"compact device has {device.num_qubits}."
        )
    if isinstance(optimization_level, bool) or not isinstance(
        optimization_level, numbers.Integral
    ):
        raise TypeError("optimization_level must be an integer.")
    if optimization_level not in (0, 1, 2, 3):
        raise ValueError("optimization_level must be 0, 1, 2, or 3.")
    if isinstance(seed_transpiler, bool) or not isinstance(seed_transpiler, numbers.Integral):
        raise TypeError("seed_transpiler must be an integer.")
    if int(seed_transpiler) < 0:
        raise ValueError("seed_transpiler must be nonnegative.")
    if not isinstance(enforce_no_swaps, bool):
        raise TypeError("enforce_no_swaps must be bool.")
    pass_manager = dependencies["generate_preset_pass_manager"](
        optimization_level=optimization_level,
        target=device.target,
        initial_layout=list(range(device.num_qubits)),
        routing_method="none",
        approximation_degree=1.0,
        seed_transpiler=int(seed_transpiler),
    )
    compiled = pass_manager.run(logical_circuit)
    counts = {str(name): int(count) for name, count in compiled.count_ops().items()}
    swaps = counts.get("swap", 0)
    unitaries = counts.get("unitary", 0)
    if enforce_no_swaps and swaps:
        raise ArithmeticError(f"Native compilation inserted {swaps} SWAP gates.")
    if unitaries:
        raise ArithmeticError(f"Native compilation left {unitaries} opaque unitary instructions.")
    allowed = set(device.target.operation_names) | {"barrier"}
    unsupported = sorted(set(counts) - allowed)
    if unsupported:
        raise ArithmeticError(
            f"Native circuit contains operations absent from the compact Target: {unsupported}."
        )
    if compiled.num_qubits != device.num_qubits:
        raise ArithmeticError(
            "Native compilation changed compact register width; refusing a possible "
            "full-backend density-matrix simulation."
        )

    layout = compiled.layout
    if layout is None:
        initial_layout = tuple(range(device.num_qubits))
        final_layout = initial_layout
    else:
        initial_layout = tuple(
            int(index) for index in layout.initial_index_layout(filter_ancillas=True)
        )
        final_layout = tuple(
            int(index) for index in layout.final_index_layout(filter_ancillas=True)
        )
    expected = tuple(range(device.num_qubits))
    if final_layout != expected:
        raise ArithmeticError(
            "routing_method='none' nevertheless changed the compact logical ordering: "
            f"{final_layout!r}."
        )
    return NativeCompilation(
        circuit=compiled,
        gate_counts=tuple(sorted(counts.items())),
        depth=int(compiled.depth()),
        size=int(compiled.size()),
        swap_count=swaps,
        unitary_count=unitaries,
        initial_index_layout=initial_layout,
        final_index_layout=final_layout,
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed_transpiler),
    )


def _normalized_vector(state: ArrayLike, *, num_qubits: int, name: str) -> ComplexArray:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    expected = 1 << num_qubits
    if vector.size != expected:
        raise ValueError(
            f"{name} has length {vector.size}, expected {expected} for {num_qubits} qubits."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite entries.")
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=2e-11):
        raise ValueError(f"{name} must be normalized; norm={norm:.17g}.")
    return np.asarray(vector, dtype=np.complex128)


def _pure_fidelity(left: ComplexArray, right: ComplexArray) -> float:
    overlap = np.vdot(left, right)
    value = float(overlap.real * overlap.real + overlap.imag * overlap.imag)
    if value < -2e-12 or value > 1.0 + 2e-10:
        raise ArithmeticError(f"Invalid pure-state fidelity {value:.17g}.")
    return value


def _mixed_fidelity(state: ComplexArray, density_matrix: ComplexArray) -> float:
    value = complex(np.vdot(state, density_matrix @ state))
    if abs(value.imag) > 2e-9:
        raise ArithmeticError(
            f"Mixed-state fidelity has a significant imaginary part {value.imag:.3e}."
        )
    result = float(value.real)
    if result < -2e-9 or result > 1.0 + 2e-9:
        raise ArithmeticError(f"Invalid mixed-state fidelity {result:.17g}.")
    return result


def _hermiticity_residual_chunked(matrix: ComplexArray, block_rows: int = 128) -> float:
    residual = 0.0
    dimension = matrix.shape[0]
    for start in range(0, dimension, block_rows):
        stop = min(dimension, start + block_rows)
        block = matrix[start:stop, :] - matrix[:, start:stop].conj().T
        residual = max(residual, float(np.max(np.abs(block), initial=0.0)))
    return residual


def _energy_expectation(hamiltonian: Any, density_matrix: ComplexArray) -> float:
    dimension = density_matrix.shape[0]
    try:
        from scipy import sparse

        if sparse.issparse(hamiltonian):
            matrix = hamiltonian.tocoo(copy=False)
            if matrix.shape != (dimension, dimension):
                raise ValueError(
                    f"hamiltonian has shape {matrix.shape}, expected {(dimension, dimension)}."
                )
            value = np.sum(matrix.data * density_matrix[matrix.col, matrix.row])
        else:
            matrix = np.asarray(hamiltonian, dtype=np.complex128)
            if matrix.shape != (dimension, dimension):
                raise ValueError(
                    f"hamiltonian has shape {matrix.shape}, expected {(dimension, dimension)}."
                )
            value = np.einsum("ij,ji->", matrix, density_matrix, optimize=True)
    except ImportError:
        matrix = np.asarray(hamiltonian, dtype=np.complex128)
        if matrix.shape != (dimension, dimension):
            raise ValueError(
                f"hamiltonian has shape {matrix.shape}, expected {(dimension, dimension)}."
            ) from None
        value = np.einsum("ij,ji->", matrix, density_matrix, optimize=True)
    if abs(complex(value).imag) > 2e-8:
        raise ArithmeticError("Noisy energy has a significant imaginary component.")
    return float(complex(value).real)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True, slots=True)
class NoisyEncodingResult:
    """Ground-state encoding metrics from one exact density-matrix simulation."""

    num_qubits: int
    logical_target_fidelity: float
    compilation_fidelity: float
    native_target_fidelity: float
    noisy_target_fidelity: float
    noisy_native_retention_fidelity: float
    purity: float
    density_trace_real: float
    density_trace_imag: float
    density_trace_error: float
    hermiticity_residual: float
    minimum_eigenvalue: float | None
    eigenvalue_check_performed: bool
    noisy_energy: float | None
    energy_error_per_site: float | None
    gate_counts: tuple[tuple[str, int], ...]
    native_depth: int
    native_size: int
    initial_index_layout: tuple[int, ...]
    final_index_layout: tuple[int, ...]
    physical_path: tuple[int, ...] | None
    backend_snapshot_sha256: str | None
    simulator_method: str
    seed_simulator: int
    runtime_seconds: float
    qiskit_version: str | None
    qiskit_aer_version: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NOISY_RESULT_SCHEMA,
            "num_qubits": self.num_qubits,
            "fidelity": {
                "logical_target": self.logical_target_fidelity,
                "compilation": self.compilation_fidelity,
                "native_target": self.native_target_fidelity,
                "noisy_target": self.noisy_target_fidelity,
                "noisy_native_retention": self.noisy_native_retention_fidelity,
            },
            "density_matrix_diagnostics": {
                "purity": self.purity,
                "trace_real": self.density_trace_real,
                "trace_imag": self.density_trace_imag,
                "trace_error": self.density_trace_error,
                "hermiticity_residual": self.hermiticity_residual,
                "minimum_eigenvalue": self.minimum_eigenvalue,
                "eigenvalue_check_performed": self.eigenvalue_check_performed,
            },
            "energy": {
                "noisy": self.noisy_energy,
                "error_per_site": self.energy_error_per_site,
            },
            "native_circuit": {
                "gate_counts": dict(self.gate_counts),
                "depth": self.native_depth,
                "size": self.native_size,
                "initial_index_layout": list(self.initial_index_layout),
                "final_index_layout": list(self.final_index_layout),
            },
            "hardware": {
                "physical_path": (None if self.physical_path is None else list(self.physical_path)),
                "backend_snapshot_sha256": self.backend_snapshot_sha256,
            },
            "simulation": {
                "method": self.simulator_method,
                "seed_simulator": self.seed_simulator,
                "runtime_seconds": self.runtime_seconds,
                "qiskit_version": self.qiskit_version,
                "qiskit_aer_version": self.qiskit_aer_version,
            },
        }


def evaluate_native_density_matrix(
    target_state: ArrayLike,
    logical_state: ArrayLike,
    native: NativeCompilation | Any,
    *,
    noise_model: Any | None = None,
    device: CompactWashingtonDevice | None = None,
    hamiltonian: Any | None = None,
    ground_energy: float | None = None,
    num_sites: int | None = None,
    seed_simulator: int = 2026,
    max_density_matrix_qubits: int = 16,
    compute_minimum_eigenvalue: bool | None = None,
) -> NoisyEncodingResult:
    """Evaluate an already native circuit against a pure ground-state target.

    No measurements are added, so readout noise cannot affect this metric.  The
    result is the deterministic channel output, not a finite-shot tomography
    estimate.  Full-register circuits wider than ``max_density_matrix_qubits``
    are rejected *before* state allocation; this explicitly protects against
    accidental 127-wire FakeWashington simulation.
    """

    dependencies = _qiskit_dependencies()
    from qiskit import QuantumCircuit

    if device is not None and not isinstance(device, CompactWashingtonDevice):
        raise TypeError("device must be a CompactWashingtonDevice or None.")
    if device is not None and noise_model is not None:
        raise ValueError(
            "Provide either device or noise_model, not both: their provenance would be ambiguous."
        )
    if isinstance(max_density_matrix_qubits, bool) or not isinstance(
        max_density_matrix_qubits, numbers.Integral
    ):
        raise TypeError("max_density_matrix_qubits must be an integer.")
    max_density_matrix_qubits = int(max_density_matrix_qubits)
    if max_density_matrix_qubits < 1:
        raise ValueError("max_density_matrix_qubits must be positive.")
    if isinstance(seed_simulator, bool) or not isinstance(seed_simulator, numbers.Integral):
        raise TypeError("seed_simulator must be an integer.")
    seed_simulator = int(seed_simulator)
    if seed_simulator < 0:
        raise ValueError("seed_simulator must be nonnegative.")
    if compute_minimum_eigenvalue is not None and not isinstance(
        compute_minimum_eigenvalue, bool
    ):
        raise TypeError("compute_minimum_eigenvalue must be bool or None.")
    if isinstance(native, NativeCompilation):
        compilation = native
        circuit = native.circuit
    else:
        if not isinstance(native, QuantumCircuit):
            raise TypeError("native must be a NativeCompilation or qiskit.QuantumCircuit.")
        circuit = native
        counts = tuple(sorted((str(k), int(v)) for k, v in circuit.count_ops().items()))
        layout = getattr(circuit, "layout", None)
        if layout is None:
            initial_layout = tuple(range(circuit.num_qubits))
            final_layout = initial_layout
        else:
            initial_layout = tuple(
                int(index) for index in layout.initial_index_layout(filter_ancillas=True)
            )
            final_layout = tuple(
                int(index) for index in layout.final_index_layout(filter_ancillas=True)
            )
        compilation = NativeCompilation(
            circuit=circuit,
            gate_counts=counts,
            depth=int(circuit.depth()),
            size=int(circuit.size()),
            swap_count=dict(counts).get("swap", 0),
            unitary_count=dict(counts).get("unitary", 0),
            initial_index_layout=initial_layout,
            final_index_layout=final_layout,
            optimization_level=-1,
            seed_transpiler=-1,
        )
    num_qubits = int(circuit.num_qubits)
    if num_qubits > max_density_matrix_qubits:
        raise ValueError(
            f"Refusing a {num_qubits}-qubit full density matrix; "
            f"max_density_matrix_qubits={max_density_matrix_qubits}. Compact the "
            "selected physical sub-chain before simulation."
        )
    if num_qubits < 1:
        raise ValueError("native circuit must contain at least one qubit.")
    if device is not None and device.num_qubits != num_qubits:
        raise ValueError(
            f"device has {device.num_qubits} compact qubits but circuit has {num_qubits}."
        )
    target = _normalized_vector(target_state, num_qubits=num_qubits, name="target_state")
    logical = _normalized_vector(logical_state, num_qubits=num_qubits, name="logical_state")

    nonunitary = {
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name in {"measure", "reset", "initialize"}
        or instruction.operation.name.startswith("save_")
    }
    if nonunitary:
        raise ValueError(
            "native circuit must contain only preparation unitaries before evaluation; "
            f"found {sorted(nonunitary)}."
        )
    native_state = np.asarray(
        dependencies["Statevector"].from_instruction(circuit).data,
        dtype=np.complex128,
    )
    native_state = _normalized_vector(native_state, num_qubits=num_qubits, name="native_state")
    if noise_model is None and device is not None:
        noise_model = device.noise_model

    simulation_circuit = circuit.copy()
    simulation_circuit.save_density_matrix(label="rho_noisy")
    simulator = dependencies["AerSimulator"](
        method="density_matrix",
        noise_model=noise_model,
    )
    started = time.perf_counter()
    run_result = simulator.run(
        simulation_circuit,
        seed_simulator=int(seed_simulator),
        shots=1,
    ).result()
    runtime_seconds = time.perf_counter() - started
    density_matrix = np.asarray(run_result.data(0)["rho_noisy"], dtype=np.complex128)
    expected_shape = (1 << num_qubits, 1 << num_qubits)
    if density_matrix.shape != expected_shape:
        raise ArithmeticError(
            f"Aer returned density matrix shape {density_matrix.shape}, expected {expected_shape}."
        )

    density_trace = complex(np.trace(density_matrix))
    trace_error = float(abs(density_trace - 1.0))
    hermiticity = _hermiticity_residual_chunked(density_matrix)
    purity_value = complex(np.einsum("ij,ji->", density_matrix, density_matrix, optimize=True))
    if abs(purity_value.imag) > 2e-8:
        raise ArithmeticError("Density-matrix purity has a significant imaginary part.")
    purity = float(purity_value.real)
    if trace_error > 2e-8 or hermiticity > 2e-8:
        raise ArithmeticError(
            "Aer density matrix failed trace/Hermiticity diagnostics: "
            f"trace_error={trace_error:.3e}, hermiticity={hermiticity:.3e}."
        )

    if compute_minimum_eigenvalue is None:
        compute_minimum_eigenvalue = density_matrix.shape[0] <= 256
    if compute_minimum_eigenvalue:
        minimum_eigenvalue = float(np.linalg.eigvalsh(density_matrix)[0])
        if minimum_eigenvalue < -2e-8:
            raise ArithmeticError(
                f"Density matrix is not positive semidefinite: min={minimum_eigenvalue:.3e}."
            )
    else:
        minimum_eigenvalue = None

    noisy_energy: float | None = None
    energy_error: float | None = None
    if hamiltonian is not None:
        noisy_energy = _energy_expectation(hamiltonian, density_matrix)
        if ground_energy is not None:
            if isinstance(ground_energy, bool) or not isinstance(ground_energy, numbers.Real):
                raise TypeError("ground_energy must be a real scalar.")
            if not math.isfinite(float(ground_energy)):
                raise ValueError("ground_energy must be finite.")
            if num_sites is not None and (
                isinstance(num_sites, bool) or not isinstance(num_sites, numbers.Integral)
            ):
                raise TypeError("num_sites must be an integer when it is provided.")
            sites = num_qubits if num_sites is None else int(num_sites)
            if sites < 1:
                raise ValueError("num_sites must be positive.")
            energy_error = (noisy_energy - float(ground_energy)) / sites
    elif ground_energy is not None:
        raise ValueError("ground_energy requires hamiltonian.")

    return NoisyEncodingResult(
        num_qubits=num_qubits,
        logical_target_fidelity=_pure_fidelity(target, logical),
        compilation_fidelity=_pure_fidelity(logical, native_state),
        native_target_fidelity=_pure_fidelity(target, native_state),
        noisy_target_fidelity=_mixed_fidelity(target, density_matrix),
        noisy_native_retention_fidelity=_mixed_fidelity(native_state, density_matrix),
        purity=purity,
        density_trace_real=float(density_trace.real),
        density_trace_imag=float(density_trace.imag),
        density_trace_error=trace_error,
        hermiticity_residual=hermiticity,
        minimum_eigenvalue=minimum_eigenvalue,
        eigenvalue_check_performed=bool(compute_minimum_eigenvalue),
        noisy_energy=noisy_energy,
        energy_error_per_site=energy_error,
        gate_counts=compilation.gate_counts,
        native_depth=compilation.depth,
        native_size=compilation.size,
        initial_index_layout=compilation.initial_index_layout,
        final_index_layout=compilation.final_index_layout,
        physical_path=None if device is None else device.physical_path,
        backend_snapshot_sha256=(None if device is None else device.snapshot_sha256),
        simulator_method="density_matrix",
        seed_simulator=int(seed_simulator),
        runtime_seconds=float(runtime_seconds),
        qiskit_version=_package_version("qiskit"),
        qiskit_aer_version=_package_version("qiskit-aer"),
    )


__all__ = [
    "COMPACT_DEVICE_SCHEMA",
    "NOISY_RESULT_SCHEMA",
    "CompactWashingtonDevice",
    "InstructionCalibrationAudit",
    "NativeCompilation",
    "NoisyEncodingResult",
    "QiskitNoiseDependencyError",
    "QubitCalibrationAudit",
    "build_compact_washington_device",
    "compile_to_native_target",
    "evaluate_native_density_matrix",
    "load_fake_washington_v2",
    "select_physical_path",
    "validate_physical_path",
]
