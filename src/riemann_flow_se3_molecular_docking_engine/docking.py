"""SE(3) flow-matching molecular docking engine.

Pipeline
--------
1. Sample ligand poses from a noise distribution on SE(3)
2. Flow them toward the binding-mode manifold via Riemannian CFM
3. Re-score with physics-based intermolecular energy
4. Optionally refine by geodesic gradient descent on the score landscape
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from riemann_flow_se3_molecular_docking_engine.flow_matching import (
    ConditionalFlowMatcher,
    OracleGeodesicFlow,
    VelocityFieldMLP,
)
from riemann_flow_se3_molecular_docking_engine.se3 import (
    SE3,
    exp_map,
    geodesic,
    log_map,
    random_se3,
    se3_distance,
)
from riemann_flow_se3_molecular_docking_engine.scoring import (
    AtomTypedCloud,
    DockingScoreWeights,
    ScoreBreakdown,
    make_toy_ligand,
    make_toy_receptor,
    score_pose,
)
from riemann_flow_se3_molecular_docking_engine import so3


@dataclass
class DockedPose:
    """Single docking hypothesis."""

    pose: SE3
    score: ScoreBreakdown
    rank: int = 0
    method: str = "flow"

    @property
    def energy(self) -> float:
        return self.score.total


@dataclass
class DockingResult:
    """Ranked ensemble of docked poses."""

    poses: List[DockedPose]
    receptor: AtomTypedCloud
    ligand_ref: AtomTypedCloud  # ligand in canonical frame
    runtime_steps: int = 0

    @property
    def top(self) -> DockedPose:
        return self.poses[0]

    def energies(self) -> NDArray[np.floating]:
        return np.array([p.energy for p in self.poses], dtype=float)


def apply_pose(ligand: AtomTypedCloud, pose: SE3) -> AtomTypedCloud:
    """Place ligand atoms under rigid SE(3) pose."""
    return ligand.transformed(pose.R, pose.t)


def score_se3_pose(
    receptor: AtomTypedCloud,
    ligand: AtomTypedCloud,
    pose: SE3,
    weights: Optional[DockingScoreWeights] = None,
) -> ScoreBreakdown:
    return score_pose(receptor, apply_pose(ligand, pose), weights=weights)


def finite_diff_score_twist(
    receptor: AtomTypedCloud,
    ligand: AtomTypedCloud,
    pose: SE3,
    eps: float = 1e-3,
    weights: Optional[DockingScoreWeights] = None,
) -> NDArray[np.floating]:
    """Body-frame gradient of score w.r.t. se(3) twist via central differences."""
    grad = np.zeros(6)
    base = score_se3_pose(receptor, ligand, pose, weights).total
    _ = base
    for i in range(6):
        e = np.zeros(6)
        e[i] = eps
        sp = score_se3_pose(receptor, ligand, pose.compose(exp_map(e)), weights).total
        sm = score_se3_pose(receptor, ligand, pose.compose(exp_map(-e)), weights).total
        grad[i] = (sp - sm) / (2.0 * eps)
    return grad


def geodesic_score_descent(
    receptor: AtomTypedCloud,
    ligand: AtomTypedCloud,
    pose0: SE3,
    n_steps: int = 25,
    lr: float = 0.05,
    weights: Optional[DockingScoreWeights] = None,
) -> Tuple[SE3, List[float]]:
    """Manifold gradient descent on docking score (lower = better)."""
    pose = pose0
    energies = [score_se3_pose(receptor, ligand, pose, weights).total]
    for _ in range(n_steps):
        g = finite_diff_score_twist(receptor, ligand, pose, weights=weights)
        # Descend
        pose = pose.compose(exp_map(-lr * g))
        energies.append(score_se3_pose(receptor, ligand, pose, weights).total)
    return pose, energies


def random_pose_near_pocket(
    pocket_center: NDArray[np.floating],
    rng: np.random.Generator,
    translation_std: float = 1.5,
) -> SE3:
    """Noise pose concentrated near a pocket center."""
    R = so3.random_rotation(rng)
    t = pocket_center + rng.standard_normal(3) * translation_std
    return SE3(R, t)


@dataclass
class FlowDockingEngine:
    """End-to-end SE(3) flow-matching docking engine."""

    cfm: ConditionalFlowMatcher = field(
        default_factory=lambda: ConditionalFlowMatcher(
            model=VelocityFieldMLP(hidden=48, seed=7)
        )
    )
    weights: DockingScoreWeights = field(default_factory=DockingScoreWeights)
    n_flow_steps: int = 40
    n_refine_steps: int = 15
    refine_lr: float = 0.04

    def generate_targets_from_crystal(
        self,
        crystal_pose: SE3,
        n: int = 16,
        jitter_rot: float = 0.15,
        jitter_trans: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ) -> List[SE3]:
        """Augment a crystal pose into a mini binding-mode dataset."""
        if rng is None:
            rng = np.random.default_rng()
        out = [crystal_pose]
        for _ in range(n - 1):
            dxi = np.concatenate(
                [
                    rng.normal(0, jitter_rot, size=3),
                    rng.normal(0, jitter_trans, size=3),
                ]
            )
            out.append(crystal_pose.compose(exp_map(dxi)))
        return out

    def train_on_poses(
        self,
        target_poses: Sequence[SE3],
        n_epochs: int = 8,
        batch_size: int = 16,
        lr: float = 8e-3,
        noise_scale: float = 2.0,
        seed: int = 0,
    ) -> List[float]:
        """Fit CFM so noise → binding poses (demo-scale finite-diff training)."""
        rng = np.random.default_rng(seed)
        losses: List[float] = []
        targets = list(target_poses)
        for _ in range(n_epochs):
            idx = rng.integers(0, len(targets), size=batch_size)
            data = [targets[i] for i in idx]
            noise = [random_se3(rng, translation_scale=noise_scale) for _ in range(batch_size)]
            loss = self.cfm.train_step_fd(noise, data, lr=lr, rng=rng)
            losses.append(loss)
        return losses

    def dock(
        self,
        receptor: AtomTypedCloud,
        ligand: AtomTypedCloud,
        n_poses: int = 12,
        pocket_center: Optional[NDArray[np.floating]] = None,
        seed: int = 0,
        use_oracle_target: Optional[SE3] = None,
        refine: bool = True,
    ) -> DockingResult:
        """Generate, score, and rank ligand poses.

        If ``use_oracle_target`` is set, integrate the closed-form geodesic
        flow toward that pose (exact recovery test / privileged docking).
        Otherwise sample from the learned CFM velocity field.
        """
        rng = np.random.default_rng(seed)
        if pocket_center is None:
            pocket_center = receptor.centroid()

        docked: List[DockedPose] = []
        for i in range(n_poses):
            g0 = random_pose_near_pocket(pocket_center, rng)

            if use_oracle_target is not None:
                oracle = OracleGeodesicFlow(use_oracle_target)
                pose = oracle.integrate(g0, n_steps=self.n_flow_steps)
                method = "oracle-flow"
            else:
                pose = self.cfm.sample(g0=g0, n_steps=self.n_flow_steps, rng=rng)
                method = "cfm-flow"

            if refine:
                pose, _ = geodesic_score_descent(
                    receptor,
                    ligand,
                    pose,
                    n_steps=self.n_refine_steps,
                    lr=self.refine_lr,
                    weights=self.weights,
                )
                method = method + "+refine"

            sc = score_se3_pose(receptor, ligand, pose, self.weights)
            docked.append(DockedPose(pose=pose, score=sc, method=method))

        docked.sort(key=lambda p: p.energy)
        for rank, p in enumerate(docked):
            p.rank = rank

        return DockingResult(
            poses=docked,
            receptor=receptor,
            ligand_ref=ligand,
            runtime_steps=n_poses * (self.n_flow_steps + (self.n_refine_steps if refine else 0)),
        )

    def rmsd_to_reference(
        self,
        ligand: AtomTypedCloud,
        pose: SE3,
        ref_pose: SE3,
    ) -> float:
        """Heavy-atom RMSD between pose and reference pose."""
        a = apply_pose(ligand, pose).coords
        b = apply_pose(ligand, ref_pose).coords
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def synthetic_crystal_pose(
    receptor: AtomTypedCloud,
    ligand: AtomTypedCloud,
    seed: int = 3,
) -> SE3:
    """Construct a plausible 'crystal' pose: ligand centroid → pocket, mild tilt."""
    rng = np.random.default_rng(seed)
    # Align ligand centroid to pocket center with small rotation
    R = so3.exp_map(rng.normal(0, 0.2, size=3))
    t = receptor.centroid() - R @ ligand.centroid()
    # Drop slightly into the pocket (−z if pocket opens +z)
    t = t + np.array([0.0, 0.0, -0.4])
    return SE3(R, t)


def run_demo_docking(
    n_poses: int = 8,
    seed: int = 42,
    train_epochs: int = 4,
) -> DockingResult:
    """Full synthetic docking demo used by CLI and tests."""
    rng = np.random.default_rng(seed)
    receptor = make_toy_receptor(seed=seed)
    ligand = make_toy_ligand(seed=seed + 1).centered()
    crystal = synthetic_crystal_pose(receptor, ligand, seed=seed + 2)

    engine = FlowDockingEngine(
        cfm=ConditionalFlowMatcher(model=VelocityFieldMLP(hidden=32, seed=seed)),
        n_flow_steps=24,
        n_refine_steps=10,
    )

    # Train CFM toward crystal-augmented binding modes
    targets = engine.generate_targets_from_crystal(crystal, n=24, rng=rng)
    engine.train_on_poses(targets, n_epochs=train_epochs, batch_size=12, seed=seed)

    # Dock with learned flow + physics refine
    result = engine.dock(
        receptor,
        ligand,
        n_poses=n_poses,
        pocket_center=receptor.centroid(),
        seed=seed + 7,
        refine=True,
    )
    return result
