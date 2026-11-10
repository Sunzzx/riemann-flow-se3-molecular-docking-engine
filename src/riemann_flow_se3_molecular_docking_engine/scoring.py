"""Physics-inspired intermolecular scoring for rigid-body docking.

Combines:
- Soft-sphere steric repulsion (Lennard-Jones-like, 12-6 truncated)
- Coulomb electrostatics with distance-dependent dielectric
- Hydrophobic contact reward (simple carbon-carbon proximity)
- Shape complementarity via Gaussian overlap (efficient)

All functions are pure NumPy and differentiable in the finite-difference sense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

EPS = 1e-8


# Minimal atom-type radii (Å) and partial charges for toy chemotypes
VDW_RADIUS = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "X": 1.70,  # generic heavy atom
}

DEFAULT_CHARGE = {
    "H": 0.05,
    "C": 0.0,
    "N": -0.40,
    "O": -0.50,
    "S": -0.15,
    "P": 0.40,
    "F": -0.25,
    "Cl": -0.10,
    "Br": -0.08,
    "I": -0.05,
    "X": 0.0,
}


def pairwise_distances(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
) -> NDArray[np.floating]:
    """(Na, Nb) Euclidean distance matrix."""
    a = np.asarray(a, dtype=float).reshape(-1, 3)
    b = np.asarray(b, dtype=float).reshape(-1, 3)
    # (a-b)^2 = a^2 + b^2 - 2 a·b
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(d2 + EPS)


def lennard_jones(
    d: NDArray[np.floating],
    sigma: NDArray[np.floating],
    epsilon: float = 0.2,
    soft_cap: float = 50.0,
) -> NDArray[np.floating]:
    """Soft 12-6 LJ potential, capped to avoid explosions at d→0."""
    r = np.maximum(d, 0.2 * sigma)
    x = sigma / r
    x2 = x * x
    x6 = x2 * x2 * x2
    x12 = x6 * x6
    e = 4.0 * epsilon * (x12 - x6)
    return np.clip(e, -soft_cap, soft_cap)


def coulomb(
    d: NDArray[np.floating],
    q_i: NDArray[np.floating],
    q_j: NDArray[np.floating],
    dielectric: float = 4.0,
    cutoff: float = 12.0,
) -> NDArray[np.floating]:
    """Distance-dependent dielectric Coulomb: E = qi qj / (ε d)."""
    # 332.0636 converts e²/Å → kcal/mol
    KE = 332.0636 / dielectric
    qi = q_i.reshape(-1, 1)
    qj = q_j.reshape(1, -1)
    mask = d < cutoff
    e = np.zeros_like(d)
    e[mask] = KE * (qi * qj)[mask] / np.maximum(d[mask], 0.5)
    return e


def hydrophobic_contact(
    d: NDArray[np.floating],
    is_hydrophobic_i: NDArray[np.bool_],
    is_hydrophobic_j: NDArray[np.bool_],
    well_min: float = 3.5,
    well_max: float = 5.5,
    strength: float = -0.15,
) -> NDArray[np.floating]:
    """Square-well hydrophobic reward for C–C (and generic X–X) pairs."""
    hi = is_hydrophobic_i.reshape(-1, 1)
    hj = is_hydrophobic_j.reshape(1, -1)
    pair = hi & hj
    in_well = (d >= well_min) & (d <= well_max) & pair
    e = np.zeros_like(d)
    e[in_well] = strength
    return e


def gaussian_overlap(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    radii_a: NDArray[np.floating],
    radii_b: NDArray[np.floating],
    alpha_scale: float = 0.8,
) -> float:
    """Gaussian shape overlap volume proxy (Grant & Pickup style).

    O = Σᵢⱼ exp(−αᵢⱼ ‖xᵢ − yⱼ‖²) with αᵢⱼ = α / (rᵢ² + rⱼ²)
    Higher overlap → better shape complementarity (we return −O as energy).
    """
    d = pairwise_distances(a, b)
    ra = radii_a.reshape(-1, 1)
    rb = radii_b.reshape(1, -1)
    alpha = alpha_scale / (ra ** 2 + rb ** 2 + EPS)
    return float(np.sum(np.exp(-alpha * d ** 2)))


@dataclass
class AtomTypedCloud:
    """Lightweight atom-typed point cloud for scoring."""

    coords: NDArray[np.floating]
    elements: NDArray[np.str_]

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float).reshape(-1, 3)
        self.elements = np.asarray(self.elements).astype(str).reshape(-1)
        if len(self.coords) != len(self.elements):
            raise ValueError("coords / elements length mismatch")

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])

    def radii(self) -> NDArray[np.floating]:
        return np.array([VDW_RADIUS.get(e, 1.70) for e in self.elements], dtype=float)

    def charges(self) -> NDArray[np.floating]:
        return np.array([DEFAULT_CHARGE.get(e, 0.0) for e in self.elements], dtype=float)

    def hydrophobic_mask(self) -> NDArray[np.bool_]:
        return np.array([e in {"C", "X", "S"} for e in self.elements], dtype=bool)

    def centroid(self) -> NDArray[np.floating]:
        return self.coords.mean(axis=0)

    def centered(self) -> "AtomTypedCloud":
        return AtomTypedCloud(self.coords - self.centroid(), self.elements)

    def transformed(self, R: NDArray[np.floating], t: NDArray[np.floating]) -> "AtomTypedCloud":
        R = np.asarray(R, dtype=float).reshape(3, 3)
        t = np.asarray(t, dtype=float).reshape(3)
        return AtomTypedCloud(self.coords @ R.T + t, self.elements)


@dataclass
class DockingScoreWeights:
    steric: float = 1.0
    electro: float = 1.0
    hydrophobic: float = 1.0
    shape: float = 0.5
    clash_threshold: float = 0.7  # fraction of sum of VdW radii


@dataclass
class ScoreBreakdown:
    total: float
    steric: float
    electro: float
    hydrophobic: float
    shape: float
    n_clashes: int


def score_pose(
    receptor: AtomTypedCloud,
    ligand: AtomTypedCloud,
    weights: Optional[DockingScoreWeights] = None,
) -> ScoreBreakdown:
    """Full intermolecular score (lower is better)."""
    if weights is None:
        weights = DockingScoreWeights()

    d = pairwise_distances(receptor.coords, ligand.coords)
    rr = receptor.radii()
    rl = ligand.radii()
    sigma = 0.5 * (rr.reshape(-1, 1) + rl.reshape(1, -1))  # Lorentz mixing ~ arithmetic

    steric = float(np.sum(lennard_jones(d, sigma)))
    electro = float(np.sum(coulomb(d, receptor.charges(), ligand.charges())))
    hydro = float(
        np.sum(
            hydrophobic_contact(
                d, receptor.hydrophobic_mask(), ligand.hydrophobic_mask()
            )
        )
    )
    shape = -gaussian_overlap(receptor.coords, ligand.coords, rr, rl)

    clash_limit = weights.clash_threshold * (rr.reshape(-1, 1) + rl.reshape(1, -1))
    n_clashes = int(np.sum(d < clash_limit))

    total = (
        weights.steric * steric
        + weights.electro * electro
        + weights.hydrophobic * hydro
        + weights.shape * shape
    )
    return ScoreBreakdown(
        total=float(total),
        steric=steric,
        electro=electro,
        hydrophobic=hydro,
        shape=shape,
        n_clashes=n_clashes,
    )


def make_toy_receptor(seed: int = 0) -> AtomTypedCloud:
    """Synthetic pocket: ring of carbons + polar walls (N/O)."""
    rng = np.random.default_rng(seed)
    # Circular hydrophobic floor
    n_ring = 12
    angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    ring = np.stack([2.5 * np.cos(angles), 2.5 * np.sin(angles), np.zeros(n_ring)], axis=1)
    # Polar lid atoms
    lid = np.array(
        [
            [0.0, 0.0, 3.0],
            [1.5, 0.0, 2.5],
            [-1.5, 0.0, 2.5],
            [0.0, 1.5, 2.5],
            [0.0, -1.5, 2.5],
        ]
    )
    coords = np.vstack([ring, lid])
    # mild noise
    coords = coords + rng.normal(0, 0.05, size=coords.shape)
    elements = np.array(["C"] * n_ring + ["O", "N", "N", "O", "O"])
    return AtomTypedCloud(coords, elements)


def make_toy_ligand(seed: int = 1) -> AtomTypedCloud:
    """Synthetic ligand: small aromatic + H-bond donor/acceptor."""
    rng = np.random.default_rng(seed)
    # Benzene-like hexagon in xy plane, centered at origin
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring = np.stack([1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)], axis=1)
    extras = np.array(
        [
            [2.6, 0.0, 0.0],   # O acceptor
            [-2.6, 0.0, 0.0],  # N donor-ish
            [0.0, 0.0, 1.2],   # axial C
        ]
    )
    coords = np.vstack([ring, extras])
    coords = coords + rng.normal(0, 0.02, size=coords.shape)
    elements = np.array(["C"] * 6 + ["O", "N", "C"])
    return AtomTypedCloud(coords, elements)
