"""Unit tests for SE(3) rigid motions."""

from __future__ import annotations

import numpy as np
import pytest

from riemann_flow_se3_molecular_docking_engine.se3 import (
    SE3,
    align_point_clouds_kabsch,
    exp_map,
    geodesic,
    hat,
    log_map,
    random_se3,
    se3_distance,
    vee,
)
from riemann_flow_se3_molecular_docking_engine import so3


def test_hat_vee_se3():
    rng = np.random.default_rng(0)
    for _ in range(15):
        xi = rng.standard_normal(6)
        assert np.allclose(vee(hat(xi)), xi)


def test_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(30):
        # moderate twists avoid π singularities
        xi = rng.standard_normal(6) * 0.4
        g = exp_map(xi)
        assert so3.rotation_matrix_check(g.R)
        xi2 = log_map(g)
        g2 = exp_map(xi2)
        assert se3_distance(g, g2) < 1e-5


def test_identity():
    g = SE3.identity()
    assert np.allclose(log_map(g), 0.0)
    assert se3_distance(g, exp_map(np.zeros(6))) < 1e-12


def test_inverse_compose():
    rng = np.random.default_rng(2)
    g = random_se3(rng)
    h = g.compose(g.inverse())
    assert se3_distance(h, SE3.identity()) < 1e-8
    h2 = g.inverse().compose(g)
    assert se3_distance(h2, SE3.identity()) < 1e-8


def test_matrix_roundtrip():
    rng = np.random.default_rng(3)
    g = random_se3(rng)
    g2 = SE3.from_matrix(g.as_matrix())
    assert se3_distance(g, g2) < 1e-12


def test_geodesic_endpoints():
    rng = np.random.default_rng(4)
    g0, g1 = random_se3(rng), random_se3(rng)
    assert se3_distance(geodesic(g0, g1, 0.0), g0) < 1e-9
    assert se3_distance(geodesic(g0, g1, 1.0), g1) < 1e-8


def test_apply_composition():
    rng = np.random.default_rng(5)
    g = random_se3(rng)
    h = random_se3(rng)
    p = rng.standard_normal(3)
    # (g∘h)·p = g·(h·p)
    assert np.allclose((g @ h).apply(p), g.apply(h.apply(p)), atol=1e-10)


def test_kabsch_recovers_pose():
    rng = np.random.default_rng(6)
    src = rng.standard_normal((30, 3))
    true = random_se3(rng, translation_scale=2.0)
    dst = true.apply(src)
    est = align_point_clouds_kabsch(src, dst)
    assert se3_distance(est, true) < 1e-6
    # RMSD after alignment
    err = np.sqrt(np.mean(np.sum((est.apply(src) - dst) ** 2, axis=1)))
    assert err < 1e-8


def test_adjoint_action():
    rng = np.random.default_rng(7)
    g = random_se3(rng)
    Ad = g.adjoint()
    assert Ad.shape == (6, 6)
    # Ad should be invertible
    assert abs(np.linalg.det(Ad)) > 0.1
