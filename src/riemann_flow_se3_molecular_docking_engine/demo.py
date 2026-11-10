"""Interactive terminal demo for SE(3) Riemannian flow-matching docking."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from riemann_flow_se3_molecular_docking_engine.docking import (
    FlowDockingEngine,
    apply_pose,
    run_demo_docking,
    synthetic_crystal_pose,
)
from riemann_flow_se3_molecular_docking_engine.flow_matching import (
    OracleGeodesicFlow,
    geodesic_cfm_consistency_error,
)
from riemann_flow_se3_molecular_docking_engine.se3 import SE3, random_se3, se3_distance
from riemann_flow_se3_molecular_docking_engine.scoring import (
    make_toy_ligand,
    make_toy_receptor,
    score_pose,
)
from riemann_flow_se3_molecular_docking_engine import so3


def _banner() -> None:
    print(
        r"""
╔══════════════════════════════════════════════════════════════════╗
║   RIEMANN-FLOW SE(3) MOLECULAR DOCKING ENGINE                    ║
║   Equivariant Riemannian Flow Matching on SE(3)                  ║
║   Rigid-body receptor–ligand pose generation                     ║
╚══════════════════════════════════════════════════════════════════╝
"""
    )


def _ascii_pose_panel(points: np.ndarray, width: int = 48, height: int = 18) -> str:
    """Project XY of a point cloud into a crude ASCII density panel."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return "(empty)"
    xy = pts[:, :2]
    mn = xy.min(axis=0) - 0.5
    mx = xy.max(axis=0) + 0.5
    span = np.maximum(mx - mn, 1e-6)
    grid = np.zeros((height, width), dtype=int)
    for x, y in xy:
        c = int((x - mn[0]) / span[0] * (width - 1))
        r = int((1.0 - (y - mn[1]) / span[1]) * (height - 1))
        r = int(np.clip(r, 0, height - 1))
        c = int(np.clip(c, 0, width - 1))
        grid[r, c] += 1
    chars = " ··:+*#"
    lines = []
    for row in grid:
        line = "".join(chars[min(v, len(chars) - 1)] for v in row)
        lines.append("│" + line + "│")
    border = "┌" + "─" * width + "┐"
    border2 = "└" + "─" * width + "┘"
    return "\n".join([border] + lines + [border2])


def run_geometry_selftest() -> None:
    print("▸ SO(3) / SE(3) geometric self-test")
    rng = np.random.default_rng(0)
    R = so3.random_rotation(rng)
    assert so3.rotation_matrix_check(R), "Haar sample not in SO(3)"
    omega = so3.log_map(R)
    R2 = so3.exp_map(omega)
    err = so3.rotation_distance(R, R2)
    print(f"  exp∘log SO(3) error: {err:.3e}")

    g0 = random_se3(rng, 1.0)
    g1 = random_se3(rng, 1.0)
    cfm_err = geodesic_cfm_consistency_error(g0, g1)
    print(f"  geodesic CFM path consistency: {cfm_err:.3e}")

    oracle = OracleGeodesicFlow(g1)
    g_rec = oracle.integrate(g0, n_steps=60)
    rec_err = se3_distance(g_rec, g1)
    print(f"  oracle flow recovery SE(3) distance: {rec_err:.3e}")
    print("  ✓ geometry OK\n")


def run_scoring_panel() -> None:
    print("▸ Intermolecular scoring on synthetic chemotypes")
    receptor = make_toy_receptor(0)
    ligand = make_toy_ligand(1).centered()
    crystal = synthetic_crystal_pose(receptor, ligand, seed=5)
    placed = apply_pose(ligand, crystal)
    sc = score_pose(receptor, placed)
    print(f"  receptor atoms: {receptor.n_atoms} | ligand atoms: {ligand.n_atoms}")
    print(
        f"  crystal score  total={sc.total:8.3f}  "
        f"steric={sc.steric:7.3f}  electro={sc.electro:7.3f}  "
        f"hydro={sc.hydrophobic:7.3f}  shape={sc.shape:7.3f}  clashes={sc.n_clashes}"
    )
    # Far-away pose should be worse (less shape, near-zero contacts) or clash-free high
    far = SE3(np.eye(3), np.array([20.0, 0.0, 0.0]))
    sc_far = score_pose(receptor, apply_pose(ligand, far))
    print(f"  unbound score  total={sc_far.total:8.3f}  (shape-only baseline)")
    print("  pocket (receptor XY):")
    print(_ascii_pose_panel(receptor.coords))
    print("  ligand at crystal pose (XY):")
    print(_ascii_pose_panel(placed.coords))
    print()


def run_docking_demo(n_poses: int = 8, epochs: int = 3) -> None:
    print("▸ SE(3) flow-matching docking demo")
    t0 = time.perf_counter()
    result = run_demo_docking(n_poses=n_poses, seed=42, train_epochs=epochs)
    dt = time.perf_counter() - t0

    print(f"  generated {len(result.poses)} poses in {dt:.2f}s "
          f"({result.runtime_steps} integrator steps)")
    print("  rank | energy    | steric   | electro  | hydro    | shape    | clashes | method")
    print("  " + "-" * 78)
    for p in result.poses[: min(8, len(result.poses))]:
        s = p.score
        print(
            f"  {p.rank:4d} | {s.total:8.3f} | {s.steric:7.3f} | {s.electro:7.3f} | "
            f"{s.hydrophobic:7.3f} | {s.shape:7.3f} | {s.n_clashes:7d} | {p.method}"
        )

    top = result.top
    print("\n  top-1 ligand pose (XY projection):")
    print(_ascii_pose_panel(apply_pose(result.ligand_ref, top.pose).coords))
    print(
        f"\n  top pose R det={np.linalg.det(top.pose.R):.6f}  "
        f"t={np.array2string(top.pose.t, precision=3)}"
    )
    print("  ✓ docking demo complete\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="riemann-flow",
        description="SE(3) Riemannian Flow Matching Molecular Docking Engine",
    )
    parser.add_argument("--poses", type=int, default=8, help="number of docking poses")
    parser.add_argument("--epochs", type=int, default=3, help="CFM training epochs")
    parser.add_argument("--skip-dock", action="store_true", help="geometry + score only")
    args = parser.parse_args(argv)

    _banner()
    run_geometry_selftest()
    run_scoring_panel()
    if not args.skip_dock:
        run_docking_demo(n_poses=args.poses, epochs=args.epochs)

    print("Humanity impact: accelerates structure-based drug design by generating")
    print("geometrically valid ligand poses on SE(3) without expensive MD sampling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
