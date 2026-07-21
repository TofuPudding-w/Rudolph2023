"""Paper targets and additional many-body ground states for encoding tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, isspmatrix, kron
from scipy.sparse.linalg import eigsh

from .conventions import (
    COMPLEX_DTYPE,
    phase_fix_state,
    validate_integer,
    validate_real,
)
from .mps import (
    MPS,
    canonicalize_to_ran,
    mps_to_dense,
    normalize_mps,
    open_boundary_bond_dimensions,
)

ComplexArray = NDArray[np.complex128]

PAULI_I = np.eye(2, dtype=COMPLEX_DTYPE)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=COMPLEX_DTYPE)
PAULI_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=COMPLEX_DTYPE)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=COMPLEX_DTYPE)
SPIN_X = PAULI_X / 2.0
SPIN_Y = PAULI_Y / 2.0
SPIN_Z = PAULI_Z / 2.0


def grid_edges(rows: int, columns: int) -> tuple[tuple[int, int], ...]:
    """Return row-major open-boundary horizontal then vertical grid edges."""

    rows = validate_integer(rows, name="rows", minimum=1)
    columns = validate_integer(columns, name="columns", minimum=1)
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(columns - 1):
            site = row * columns + column
            edges.append((site, site + 1))
    for row in range(rows - 1):
        for column in range(columns):
            site = row * columns + column
            edges.append((site, site + columns))
    if len(set(edges)) != len(edges) or any(left == right for left, right in edges):
        raise AssertionError("Grid edge construction produced duplicates or self loops.")
    return tuple(edges)


def single_site_operator(
    operator: ArrayLike,
    site: int,
    num_qubits: int,
) -> csr_matrix:
    """Embed a 2x2 operator at one q0-MSB site as a sparse matrix."""

    num_qubits = validate_integer(num_qubits, name="num_qubits", minimum=1)
    site = validate_integer(site, name="site", minimum=0)
    if site >= num_qubits:
        raise IndexError("site is outside the register.")
    local = np.asarray(operator, dtype=COMPLEX_DTYPE)
    if local.shape != (2, 2) or not np.all(np.isfinite(local)):
        raise ValueError("operator must be a finite 2x2 matrix.")
    result = csr_matrix([[1.0 + 0.0j]])
    for index in range(num_qubits):
        factor = csr_matrix(local if index == site else PAULI_I)
        result = kron(result, factor, format="csr")
    return result


def two_site_product_operator(
    first_operator: ArrayLike,
    second_operator: ArrayLike,
    first_site: int,
    second_site: int,
    num_qubits: int,
) -> csr_matrix:
    """Embed (A_i B_j) at any two distinct q0-MSB sites."""

    num_qubits = validate_integer(num_qubits, name="num_qubits", minimum=2)
    first_site = validate_integer(first_site, name="first_site", minimum=0)
    second_site = validate_integer(second_site, name="second_site", minimum=0)
    if first_site >= num_qubits or second_site >= num_qubits:
        raise IndexError("Operator site is outside the register.")
    if first_site == second_site:
        raise ValueError("Two-site product requires distinct sites.")
    first = np.asarray(first_operator, dtype=COMPLEX_DTYPE)
    second = np.asarray(second_operator, dtype=COMPLEX_DTYPE)
    if (
        first.shape != (2, 2)
        or second.shape != (2, 2)
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("Local operators must be finite 2x2 matrices.")
    result = csr_matrix([[1.0 + 0.0j]])
    for site in range(num_qubits):
        if site == first_site:
            local = first
        elif site == second_site:
            local = second
        else:
            local = PAULI_I
        result = kron(result, csr_matrix(local), format="csr")
    return result


def heisenberg_grid_hamiltonian(
    *,
    rows: int = 3,
    columns: int = 4,
    coupling: float = 1.0,
) -> csr_matrix:
    """Return the paper's OBC AFM grid H using spin operators Pauli/2."""

    rows = validate_integer(rows, name="rows", minimum=1)
    columns = validate_integer(columns, name="columns", minimum=1)
    if rows * columns < 2:
        raise ValueError("Heisenberg grid needs at least two sites.")
    coupling = validate_real(coupling, name="coupling")
    num_qubits = rows * columns
    dimension = 1 << num_qubits
    hamiltonian = csr_matrix((dimension, dimension), dtype=COMPLEX_DTYPE)
    for first, second in grid_edges(rows, columns):
        for spin in (SPIN_X, SPIN_Y, SPIN_Z):
            hamiltonian += coupling * two_site_product_operator(
                spin, spin, first, second, num_qubits
            )
    hamiltonian = hamiltonian.tocsr()
    if not np.all(np.isfinite(hamiltonian.data)):
        raise FloatingPointError("Heisenberg Hamiltonian contains non-finite values.")
    return hamiltonian


def tfim_hamiltonian(
    num_qubits: int,
    *,
    coupling_zz: float = 1.0,
    field_x: float = 1.0,
    field_z: float = 0.0,
) -> csr_matrix:
    """Return the OBC transverse-field Ising Hamiltonian."""

    num_qubits = validate_integer(num_qubits, name="num_qubits", minimum=2)
    coupling_zz = validate_real(coupling_zz, name="coupling_zz")
    field_x = validate_real(field_x, name="field_x")
    field_z = validate_real(field_z, name="field_z")
    dimension = 1 << num_qubits
    hamiltonian = csr_matrix((dimension, dimension), dtype=COMPLEX_DTYPE)
    for site in range(num_qubits - 1):
        hamiltonian -= coupling_zz * two_site_product_operator(
            PAULI_Z, PAULI_Z, site, site + 1, num_qubits
        )
    for site in range(num_qubits):
        hamiltonian -= field_x * single_site_operator(PAULI_X, site, num_qubits)
        if field_z != 0.0:
            hamiltonian -= field_z * single_site_operator(PAULI_Z, site, num_qubits)
    hamiltonian = hamiltonian.tocsr()
    if not np.all(np.isfinite(hamiltonian.data)):
        raise FloatingPointError("TFIM Hamiltonian contains non-finite values.")
    return hamiltonian


def xxz_hamiltonian(
    num_qubits: int,
    *,
    delta: float = 1.0,
    coupling: float = 1.0,
    field_z: float = 0.0,
) -> csr_matrix:
    """Return the OBC spin-one-half XXZ Hamiltonian with optional z field."""

    num_qubits = validate_integer(num_qubits, name="num_qubits", minimum=2)
    delta = validate_real(delta, name="delta")
    coupling = validate_real(coupling, name="coupling")
    field_z = validate_real(field_z, name="field_z")
    dimension = 1 << num_qubits
    hamiltonian = csr_matrix((dimension, dimension), dtype=COMPLEX_DTYPE)
    for site in range(num_qubits - 1):
        hamiltonian += coupling * two_site_product_operator(
            SPIN_X, SPIN_X, site, site + 1, num_qubits
        )
        hamiltonian += coupling * two_site_product_operator(
            SPIN_Y, SPIN_Y, site, site + 1, num_qubits
        )
        hamiltonian += coupling * delta * two_site_product_operator(
            SPIN_Z, SPIN_Z, site, site + 1, num_qubits
        )
    if field_z != 0.0:
        for site in range(num_qubits):
            hamiltonian += field_z * single_site_operator(SPIN_Z, site, num_qubits)
    hamiltonian = hamiltonian.tocsr()
    if not np.all(np.isfinite(hamiltonian.data)):
        raise FloatingPointError("XXZ Hamiltonian contains non-finite values.")
    return hamiltonian


@dataclass(frozen=True, slots=True)
class GroundStateResult:
    """Exact/sparse-ED ground-state diagnostics."""

    state: ComplexArray
    energy: float
    first_excited_energy: float
    gap: float
    residual: float
    solver: str


def exact_ground_state(
    hamiltonian: ArrayLike | csr_matrix,
    *,
    tolerance: float = 1e-12,
    dense_dimension_cutoff: int = 512,
) -> GroundStateResult:
    """Return a deterministic phase-fixed ground state with residual check."""

    tolerance = validate_real(tolerance, name="tolerance", minimum=0.0)
    dense_dimension_cutoff = validate_integer(
        dense_dimension_cutoff, name="dense_dimension_cutoff", minimum=2
    )
    shape = getattr(hamiltonian, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Hamiltonian must be square.")
    dimension = int(shape[0])
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("Hamiltonian dimension must be a power of two >=2.")

    if isspmatrix(hamiltonian):
        sparse_h = hamiltonian.tocsr().astype(COMPLEX_DTYPE)
        if not np.all(np.isfinite(sparse_h.data)):
            raise ValueError("Hamiltonian must contain only finite values.")
        hermitian_error = sparse_h - sparse_h.getH()
        residual_h = (
            float(np.max(np.abs(hermitian_error.data)))
            if hermitian_error.nnz
            else 0.0
        )
    else:
        dense_h = np.asarray(hamiltonian, dtype=COMPLEX_DTYPE)
        if not np.all(np.isfinite(dense_h)):
            raise ValueError("Hamiltonian must contain only finite values.")
        residual_h = float(np.linalg.norm(dense_h - dense_h.conj().T, ord=np.inf))
        sparse_h = csr_matrix(dense_h)
    if residual_h > max(1e-12, tolerance * 10.0):
        raise ValueError(f"Hamiltonian is not Hermitian: residual={residual_h:.3e}.")

    if dimension <= dense_dimension_cutoff:
        values, vectors = eigh(sparse_h.toarray(), check_finite=True)
        energy = float(values[0])
        excited = float(values[1])
        state = vectors[:, 0]
        solver = "scipy.linalg.eigh"
    else:
        v0 = np.linspace(1.0, 2.0, dimension, dtype=np.float64)
        v0 /= np.linalg.norm(v0)
        values, vectors = eigsh(
            sparse_h,
            k=2,
            which="SA",
            tol=tolerance,
            v0=v0.astype(COMPLEX_DTYPE),
            maxiter=max(1000, 50 * dimension),
        )
        order = np.argsort(values)
        energy = float(values[order[0]])
        excited = float(values[order[1]])
        state = vectors[:, order[0]]
        solver = "scipy.sparse.linalg.eigsh"
    gap = excited - energy
    degeneracy_tolerance = max(1e-10, 100.0 * tolerance)
    if gap <= degeneracy_tolerance:
        raise ValueError(
            "Ground state is degenerate or numerically unresolved; a unique target "
            "requires an explicit symmetry-sector/tie-break convention."
        )
    state = phase_fix_state(state)
    residual = float(np.linalg.norm(sparse_h @ state - energy * state))
    if residual > max(1e-10, 100.0 * tolerance):
        raise ArithmeticError(f"Ground-state eigen residual is too large: {residual:.3e}.")
    return GroundStateResult(
        state=state,
        energy=energy,
        first_excited_energy=excited,
        gap=gap,
        residual=residual,
        solver=solver,
    )


def bars_and_stripes_bitstrings(rows: int = 2, columns: int = 6) -> tuple[str, ...]:
    """Enumerate unique row-major Bars-and-Stripes bitstrings."""

    rows = validate_integer(rows, name="rows", minimum=1)
    columns = validate_integer(columns, name="columns", minimum=1)
    patterns: set[str] = set()
    # Bars: each column is constant over rows.
    for column_mask in range(1 << columns):
        row = "".join(str((column_mask >> (columns - 1 - c)) & 1) for c in range(columns))
        patterns.add(row * rows)
    # Stripes: each row is constant over columns.
    for row_mask in range(1 << rows):
        bits = [str((row_mask >> (rows - 1 - r)) & 1) for r in range(rows)]
        patterns.add("".join(bit * columns for bit in bits))
    return tuple(sorted(patterns))


def bars_and_stripes_state(rows: int = 2, columns: int = 6) -> ComplexArray:
    """Return the uniform quantum superposition over unique BAS images."""

    patterns = bars_and_stripes_bitstrings(rows, columns)
    num_qubits = rows * columns
    state = np.zeros(1 << num_qubits, dtype=COMPLEX_DTYPE)
    amplitude = 1.0 / math.sqrt(len(patterns))
    for pattern in patterns:
        state[int(pattern, 2)] = amplitude
    return state


@dataclass(frozen=True, slots=True)
class RandomRealMPSResult:
    """Real-normal raw tensors and their normalized Ran-canonical target."""

    seed: int
    max_bond: int
    bond_dimensions: tuple[int, ...]
    raw_tensors: tuple[ComplexArray, ...]
    canonical_mps: MPS
    state: ComplexArray


def random_real_mps(
    num_qubits: int,
    *,
    max_bond: int,
    seed: int,
) -> RandomRealMPSResult:
    """Generate the paper's real N(0,1) random OBC MPS target."""

    num_qubits = validate_integer(num_qubits, name="num_qubits", minimum=2)
    max_bond = validate_integer(max_bond, name="max_bond", minimum=1)
    seed = validate_integer(seed, name="seed", minimum=0)
    bonds = open_boundary_bond_dimensions(num_qubits, max_bond)
    dimensions = (1, *bonds, 1)
    rng = np.random.default_rng(seed)
    raw_tensors = tuple(
        np.asarray(
            rng.normal(0.0, 1.0, size=(dimensions[site], 2, dimensions[site + 1])),
            dtype=COMPLEX_DTYPE,
        )
        for site in range(num_qubits)
    )
    raw_mps = MPS(raw_tensors)
    normalized = normalize_mps(raw_mps)
    canonical = canonicalize_to_ran(normalized, normalize=True)
    state = phase_fix_state(mps_to_dense(canonical))
    # Put the canonical MPS in the same phase gauge as the exported target.
    canonical_tensors = [tensor.copy() for tensor in canonical.tensors]
    overlap = np.vdot(mps_to_dense(canonical), state)
    canonical_tensors[0] *= overlap / abs(overlap)
    canonical = MPS(canonical_tensors)
    return RandomRealMPSResult(
        seed=seed,
        max_bond=max_bond,
        bond_dimensions=bonds,
        raw_tensors=raw_tensors,
        canonical_mps=canonical,
        state=state,
    )


__all__ = [
    "GroundStateResult",
    "PAULI_I",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "RandomRealMPSResult",
    "SPIN_X",
    "SPIN_Y",
    "SPIN_Z",
    "bars_and_stripes_bitstrings",
    "bars_and_stripes_state",
    "exact_ground_state",
    "grid_edges",
    "heisenberg_grid_hamiltonian",
    "random_real_mps",
    "single_site_operator",
    "tfim_hamiltonian",
    "two_site_product_operator",
    "xxz_hamiltonian",
]
