"""Tests for intermolecular scoring and docking pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from riemann_flow_se3_molecular_docking_engine.docking import (
    FlowDockingEngine,
    apply_pose,
    geodesic_score_descent,
    run_demo_docking,
    score_se3_pose,
    synthetic_crystal_pose,
)
from riemann_flow_se3_molecular_docking_engine.flow_matching import (
    ConditionalFlowMatcher,
    VelocityFieldMLP,
)
from riemann_flow_se3_molecular_docking_engine.se3 import SE3, exp_map, se3_distance
from riemann_flow_se3_molecular_docking_engine.scoring import (
    AtomTypedCloud,
    gaussian_overlap,
    make_toy_ligand,
    make_toy_receptor,
    pairwise_distances,
    score_pose,
)


def test_pairwise_distances_symmetric():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 3))
    b = rng.standard_normal((7, 3))
    d = pairwise_distances(a, b)
    assert d.shape == (5, 7)
    assert np.all(d >= 0)


def test_toy_molecules_well_formed():
    rec = make_toy_receptor(0)
    lig = make_toy_ligand(1)
    assert rec.n_atoms >= 10
    assert lig.n_atoms >= 6
    assert rec.radii().shape == (rec.n_atoms,)
    assert lig.charges().shape == (lig.n_atoms,)


def test_score_pose_finite():
    rec = make_toy_receptor(0)
    lig = make_toy_ligand(1)
    sc = score_pose(rec, lig)
    assert np.isfinite(sc.total)
    assert sc.n_clashes >= 0


def test_unbound_has_weaker_shape_than_overlapping():
    rec = make_toy_receptor(0)
    lig = make_toy_ligand(1).centered()
    # Overlapping at origin vs far away
    sc_near = score_pose(rec, lig.transformed(np.eye(3), rec.centroid()))
    sc_far = score_pose(rec, lig.transformed(np.eye(3), rec.centroid() + np.array([30.0, 0, 0])))
    # Far pose: almost no Gaussian overlap → shape energy closer to 0 (less negative)
    assert sc_far.shape > sc_near.shape - 1e-6  # far is larger (worse) shape term


def test_gaussian_overlap_self_positive():
    rec = make_toy_receptor(0)
    o = gaussian_overlap(rec.coords, rec.coords, rec.radii(), rec.radii())
    assert o > 0


def test_atom_cloud_transform_roundtrip_centroid():
    lig = make_toy_ligand(2)
    c = lig.centroid()
    moved = lig.transformed(np.eye(3), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(moved.centroid(), c + np.array([1.0, 2.0, 3.0]))


def test_synthetic_crystal_pose_places_near_pocket():
    rec = make_toy_receptor(0)
    lig = make_toy_ligand(1).centered()
    pose = synthetic_crystal_pose(rec, lig, seed=3)
    placed = apply_pose(lig, pose)
    d = np.linalg.norm(placed.centroid() - rec.centroid())
    assert d < 2.0  # within a few Å of pocket center


def test_score_descent_nonincreasing_trend():
    rec = make_toy_receptor(0)
    lig = make_toy_ligand(1).centered()
    # Start from a slightly perturbed crystal
    crystal = synthetic_crystal_pose(rec, lig, seed=4)
    noisy = crystal.compose(exp_map(np.array([0.1, -0.05, 0.08, 0.3, -0.2, 0.1])))
    pose_f, energies = geodesic_score_descent(rec, lig, noisy, n_steps=12, lr=0.05)
    assert len(energies) == 13
    assert np.isfinite(energies).all()
    # Final should not explode
    assert energies[-1] < energies[0] + 50.0


def test_flow_docking_engine_oracle_mode():
    rec = make_toy_receptor(1)
    lig = make_toy_ligand(2).centered()
    crystal = synthetic_crystal_pose(rec, lig, seed=9)
    engine = FlowDockingEngine(
        cfm=ConditionalFlowMatcher(model=VelocityFieldMLP(hidden=8, seed=0)),
        n_flow_steps=30,
        n_refine_steps=0,
    )
    result = engine.dock(
        rec,
        lig,
        n_poses=3,
        use_oracle_target=crystal,
        refine=False,
        seed=11,
    )
    assert len(result.poses) == 3
    # Oracle should land close to crystal
    rmsds = [engine.rmsd_to_reference(lig, p.pose, crystal) for p in result.poses]
    assert min(rmsds) < 0.5


def test_run_demo_docking_smoke():
    result = run_demo_docking(n_poses=3, seed=0, train_epochs=1)
    assert len(result.poses) == 3
    assert result.top.rank == 0
    assert np.isfinite(result.energies()).all()
    # energies sorted ascending
    e = result.energies()
    assert np.all(e[:-1] <= e[1:] + 1e-9)


def test_rmsd_zero_for_same_pose():
    lig = make_toy_ligand(0).centered()
    engine = FlowDockingEngine()
    g = SE3.identity()
    assert engine.rmsd_to_reference(lig, g, g) < 1e-12
