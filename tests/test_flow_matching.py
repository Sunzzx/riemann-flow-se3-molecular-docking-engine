"""Tests for Riemannian conditional flow matching on SE(3)."""

from __future__ import annotations

import numpy as np
import pytest

from riemann_flow_se3_molecular_docking_engine.flow_matching import (
    ConditionalFlowMatcher,
    OracleGeodesicFlow,
    VelocityFieldMLP,
    geodesic_cfm_consistency_error,
)
from riemann_flow_se3_molecular_docking_engine.se3 import SE3, random_se3, se3_distance


def test_geodesic_cfm_consistency():
    rng = np.random.default_rng(0)
    for _ in range(10):
        g0 = random_se3(rng, 1.0)
        g1 = random_se3(rng, 1.0)
        err = geodesic_cfm_consistency_error(g0, g1, n_times=21)
        assert err < 1e-5, f"CFM path error {err}"


def test_oracle_flow_recovers_target():
    rng = np.random.default_rng(1)
    g0 = random_se3(rng, 1.5)
    g1 = random_se3(rng, 1.5)
    oracle = OracleGeodesicFlow(g1)
    g_rec = oracle.integrate(g0, n_steps=80)
    assert se3_distance(g_rec, g1) < 5e-3


def test_conditional_path_endpoints():
    rng = np.random.default_rng(2)
    cfm = ConditionalFlowMatcher()
    g0, g1 = random_se3(rng), random_se3(rng)
    x0, u0 = cfm.sample_conditional_path(g0, g1, 0.0)
    x1, u1 = cfm.sample_conditional_path(g0, g1, 1.0)
    assert se3_distance(x0, g0) < 1e-9
    assert se3_distance(x1, g1) < 1e-8
    # body velocity constant along geodesic
    assert np.allclose(u0, u1)


def test_velocity_mlp_output_shape():
    model = VelocityFieldMLP(hidden=16, seed=0)
    g = SE3.identity()
    v = model.forward(g, 0.3)
    assert v.shape == (6,)
    assert np.isfinite(v).all()


def test_loss_batch_finite():
    rng = np.random.default_rng(3)
    cfm = ConditionalFlowMatcher(model=VelocityFieldMLP(hidden=16, seed=3))
    noise = [random_se3(rng, 0.8) for _ in range(8)]
    data = [SE3.identity() for _ in range(8)]
    loss = cfm.loss_batch(noise, data, rng=rng)
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_train_step_reduces_or_finite():
    rng = np.random.default_rng(4)
    cfm = ConditionalFlowMatcher(model=VelocityFieldMLP(hidden=16, seed=4))
    noise = [random_se3(rng, 0.5) for _ in range(6)]
    data = [SE3.identity() for _ in range(6)]
    l0 = cfm.loss_batch(noise, data, rng=rng)
    l1 = cfm.train_step_fd(noise, data, lr=1e-2, rng=rng)
    assert np.isfinite(l0) and np.isfinite(l1)


def test_euler_sample_returns_se3():
    rng = np.random.default_rng(5)
    cfm = ConditionalFlowMatcher(model=VelocityFieldMLP(hidden=8, seed=5))
    g = cfm.sample(n_steps=5, rng=rng)
    assert isinstance(g, SE3)
    from riemann_flow_se3_molecular_docking_engine import so3

    assert so3.rotation_matrix_check(g.R)


def test_heun_sample_returns_se3():
    rng = np.random.default_rng(6)
    cfm = ConditionalFlowMatcher(model=VelocityFieldMLP(hidden=8, seed=6))
    g = cfm.sample_heun(n_steps=4, rng=rng)
    assert isinstance(g, SE3)
