"""Unit tests for SO(3) Lie group operations."""

from __future__ import annotations

import numpy as np
import pytest

from riemann_flow_se3_molecular_docking_engine import so3


def test_hat_vee_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.standard_normal(3)
        assert np.allclose(so3.vee(so3.hat(w)), w)


def test_hat_is_skew_symmetric():
    w = np.array([1.0, -2.0, 0.5])
    W = so3.hat(w)
    assert so3.skew_symmetric_check(W)
    assert np.allclose(W @ w, 0.0)  # ω × ω = 0


def test_exp_log_roundtrip_small():
    rng = np.random.default_rng(1)
    for _ in range(30):
        omega = rng.standard_normal(3) * 0.5
        R = so3.exp_map(omega)
        assert so3.rotation_matrix_check(R)
        omega2 = so3.log_map(R)
        R2 = so3.exp_map(omega2)
        assert so3.rotation_distance(R, R2) < 1e-6


def test_exp_log_roundtrip_haar():
    rng = np.random.default_rng(2)
    for _ in range(40):
        R = so3.random_rotation(rng)
        assert so3.rotation_matrix_check(R)
        R2 = so3.exp_map(so3.log_map(R))
        assert so3.rotation_distance(R, R2) < 1e-5


def test_identity_log_zero():
    assert np.allclose(so3.log_map(np.eye(3)), 0.0, atol=1e-12)


def test_geodesic_endpoints():
    rng = np.random.default_rng(3)
    R0 = so3.random_rotation(rng)
    R1 = so3.random_rotation(rng)
    assert so3.rotation_distance(so3.geodesic(R0, R1, 0.0), R0) < 1e-9
    assert so3.rotation_distance(so3.geodesic(R0, R1, 1.0), R1) < 1e-9


def test_geodesic_is_on_manifold():
    rng = np.random.default_rng(4)
    R0, R1 = so3.random_rotation(rng), so3.random_rotation(rng)
    for t in np.linspace(0, 1, 9):
        R = so3.geodesic(R0, R1, float(t))
        assert so3.rotation_matrix_check(R)


def test_distance_triangle_inequality():
    rng = np.random.default_rng(5)
    A, B, C = [so3.random_rotation(rng) for _ in range(3)]
    dAB = so3.rotation_distance(A, B)
    dBC = so3.rotation_distance(B, C)
    dAC = so3.rotation_distance(A, C)
    assert dAC <= dAB + dBC + 1e-9


def test_apply_rotation_preserves_norms():
    rng = np.random.default_rng(6)
    R = so3.random_rotation(rng)
    pts = rng.standard_normal((20, 3))
    out = so3.apply_rotation(R, pts)
    assert np.allclose(np.linalg.norm(pts, axis=1), np.linalg.norm(out, axis=1))


def test_rodrigues_known_90deg_z():
    omega = np.array([0.0, 0.0, np.pi / 2])
    R = so3.exp_map(omega)
    # 90° about z: e1 → e2
    e1 = np.array([1.0, 0.0, 0.0])
    assert np.allclose(R @ e1, np.array([0.0, 1.0, 0.0]), atol=1e-10)
