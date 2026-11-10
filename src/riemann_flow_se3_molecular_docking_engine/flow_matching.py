"""Riemannian Conditional Flow Matching on SE(3).

Implements the continuous normalizing flow objective of Chen & Lipman (2023)
on the SE(3) manifold with closed-form geodesic conditional paths and a
lightweight velocity field approximator for pose generation.

Conditional path (x₀ ~ p₀ noise, x₁ ~ p₁ data):
    x_t = Exp_{x₀}( t · Log_{x₀}(x₁) )     # geodesic CFM
Target velocity (body frame):
    u_t(x_t | x₁) = Log_{x₀}(x₁)           # constant along geodesic
Training loss:
    L = E_{t,x₀,x₁} [ ‖ v_θ(x_t, t) − u_t ‖² ]
Sampling (Euler / Heun on the manifold):
    x_{t+Δt} = Exp_{x_t}( Δt · v_θ(x_t, t) )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from riemann_flow_se3_molecular_docking_engine.se3 import (
    SE3,
    exp_map,
    geodesic,
    geodesic_velocity,
    log_map,
    random_se3,
    se3_distance,
)


def _featurize_pose(g: SE3, t: float) -> NDArray[np.floating]:
    """Pose + time feature vector in R^{1+6+9} = (t, log(g), R.flatten())."""
    xi = log_map(g)
    return np.concatenate([[t], xi, g.R.reshape(-1)])


@dataclass
class VelocityFieldMLP:
    """Tiny pure-NumPy MLP approximating v_θ: SE(3) × [0,1] → se(3).

    Architecture: input(16) → hidden → hidden → output(6)
    Used as a trainable surrogate; also supports closed-form oracle mode.
    """

    hidden: int = 64
    seed: int = 0
    W1: NDArray[np.floating] = field(init=False, repr=False)
    b1: NDArray[np.floating] = field(init=False, repr=False)
    W2: NDArray[np.floating] = field(init=False, repr=False)
    b2: NDArray[np.floating] = field(init=False, repr=False)
    W3: NDArray[np.floating] = field(init=False, repr=False)
    b3: NDArray[np.floating] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        in_dim = 1 + 6 + 9  # t + xi + R
        h = self.hidden
        # Xavier / Glorot init
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), size=(in_dim, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, np.sqrt(2.0 / h), size=(h, h))
        self.b2 = np.zeros(h)
        self.W3 = rng.normal(0, np.sqrt(2.0 / h) * 0.1, size=(h, 6))
        self.b3 = np.zeros(6)

    @staticmethod
    def _relu(x: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.maximum(x, 0.0)

    def forward(self, g: SE3, t: float) -> NDArray[np.floating]:
        x = _featurize_pose(g, t)
        h = self._relu(x @ self.W1 + self.b1)
        h = self._relu(h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3

    def parameters(self) -> List[NDArray[np.floating]]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def clone_params(self) -> List[NDArray[np.floating]]:
        return [p.copy() for p in self.parameters()]

    def set_params(self, params: Sequence[NDArray[np.floating]]) -> None:
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = [
            np.asarray(p, dtype=float) for p in params
        ]


def _clamp_twist(xi: NDArray[np.floating], max_rot: float = np.pi, max_trans: float = 5.0) -> NDArray[np.floating]:
    """Bound body twist so a single integrator step stays numerically stable."""
    xi = np.asarray(xi, dtype=float).reshape(6).copy()
    w = xi[:3]
    nw = float(np.linalg.norm(w))
    if nw > max_rot:
        xi[:3] = w * (max_rot / nw)
    v = xi[3:]
    nv = float(np.linalg.norm(v))
    if nv > max_trans:
        xi[3:] = v * (max_trans / nv)
    return xi


@dataclass
class ConditionalFlowMatcher:
    """Riemannian Conditional Flow Matching trainer / sampler on SE(3)."""

    model: VelocityFieldMLP = field(default_factory=VelocityFieldMLP)
    rot_weight: float = 1.0
    trans_weight: float = 1.0
    max_step_rot: float = np.pi
    max_step_trans: float = 5.0

    def sample_conditional_path(
        self,
        g0: SE3,
        g1: SE3,
        t: float,
    ) -> Tuple[SE3, NDArray[np.floating]]:
        """Return (x_t, u_t) along the geodesic CFM path."""
        x_t = geodesic(g0, g1, t)
        u_t = geodesic_velocity(g0, g1, t)
        return x_t, u_t

    def loss_batch(
        self,
        noise_poses: Sequence[SE3],
        data_poses: Sequence[SE3],
        times: Optional[NDArray[np.floating]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Mean squared error between predicted and target body velocities."""
        n = len(data_poses)
        if len(noise_poses) != n:
            raise ValueError("noise_poses and data_poses length mismatch")
        if rng is None:
            rng = np.random.default_rng()
        if times is None:
            times = rng.uniform(0.0, 1.0, size=n)

        total = 0.0
        for g0, g1, t in zip(noise_poses, data_poses, times):
            x_t, u_t = self.sample_conditional_path(g0, g1, float(t))
            v_pred = self.model.forward(x_t, float(t))
            # Weighted MSE on se(3)
            diff = v_pred - u_t
            err = (
                self.rot_weight * np.sum(diff[:3] ** 2)
                + self.trans_weight * np.sum(diff[3:] ** 2)
            )
            total += float(err)
        return total / max(n, 1)

    def train_step_fd(
        self,
        noise_poses: Sequence[SE3],
        data_poses: Sequence[SE3],
        lr: float = 1e-2,
        eps: float = 1e-4,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """One finite-difference SGD step (pure NumPy, no autograd).

        Suitable for small demos / unit tests; production would use JAX/torch.
        """
        base_loss = self.loss_batch(noise_poses, data_poses, rng=rng)
        params = self.model.parameters()
        grads: List[NDArray[np.floating]] = []

        for p in params:
            g = np.zeros_like(p)
            flat_p = p.ravel()
            flat_g = g.ravel()
            # Coordinate-wise FD on a random subset for speed
            n_coords = flat_p.size
            # Full FD for small layers; subsample large ones
            max_coords = min(n_coords, 32)
            if rng is None:
                rng = np.random.default_rng()
            idxs = rng.choice(n_coords, size=max_coords, replace=False)
            for idx in idxs:
                flat_p[idx] += eps
                lp = self.loss_batch(noise_poses, data_poses, rng=rng)
                flat_p[idx] -= 2 * eps
                lm = self.loss_batch(noise_poses, data_poses, rng=rng)
                flat_p[idx] += eps
                flat_g[idx] = (lp - lm) / (2.0 * eps)
            grads.append(g)

        # SGD update
        for p, g in zip(params, grads):
            p -= lr * g

        return base_loss

    def euler_step(self, g: SE3, t: float, dt: float) -> SE3:
        """Single manifold Euler step: x ← Exp_x(dt · v_θ(x, t))."""
        v = _clamp_twist(
            self.model.forward(g, t),
            max_rot=self.max_step_rot,
            max_trans=self.max_step_trans,
        )
        return g.compose(exp_map(dt * v))

    def sample(
        self,
        g0: Optional[SE3] = None,
        n_steps: int = 50,
        t0: float = 0.0,
        t1: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        return_path: bool = False,
    ) -> SE3 | Tuple[SE3, List[SE3]]:
        """Integrate the learned ODE from noise (t=0) to data (t=1)."""
        if g0 is None:
            if rng is None:
                rng = np.random.default_rng()
            g0 = random_se3(rng, translation_scale=2.0)

        path = [g0]
        g = g0
        dt = (t1 - t0) / n_steps
        for i in range(n_steps):
            t = t0 + i * dt
            g = self.euler_step(g, t, dt)
            if return_path:
                path.append(g)

        if return_path:
            return g, path
        return g

    def heun_step(self, g: SE3, t: float, dt: float) -> SE3:
        """Heun (improved Euler) manifold integrator."""
        v1 = _clamp_twist(
            self.model.forward(g, t),
            max_rot=self.max_step_rot,
            max_trans=self.max_step_trans,
        )
        g_pred = g.compose(exp_map(dt * v1))
        v2 = _clamp_twist(
            self.model.forward(g_pred, t + dt),
            max_rot=self.max_step_rot,
            max_trans=self.max_step_trans,
        )
        v_avg = 0.5 * (v1 + v2)
        return g.compose(exp_map(dt * v_avg))

    def sample_heun(
        self,
        g0: Optional[SE3] = None,
        n_steps: int = 30,
        rng: Optional[np.random.Generator] = None,
    ) -> SE3:
        if g0 is None:
            if rng is None:
                rng = np.random.default_rng()
            g0 = random_se3(rng, translation_scale=2.0)
        g = g0
        dt = 1.0 / n_steps
        for i in range(n_steps):
            g = self.heun_step(g, i * dt, dt)
        return g


class OracleGeodesicFlow:
    """Closed-form oracle velocity field that knows the target pose g1.

    v(x_t, t | g1) = Log_{x_t}(g1) / (1 − t + ε)
    Used for verification that geodesic integration recovers g1 exactly.
    """

    def __init__(self, g1: SE3, eps: float = 1e-4):
        self.g1 = g1
        self.eps = eps

    def forward(self, g: SE3, t: float) -> NDArray[np.floating]:
        remaining = max(1.0 - t, self.eps)
        return log_map(g.inverse().compose(self.g1)) / remaining

    def integrate(self, g0: SE3, n_steps: int = 40) -> SE3:
        g = g0
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = i * dt
            v = self.forward(g, t)
            g = g.compose(exp_map(dt * v))
        return g


def fit_identity_flow(
    n_samples: int = 32,
    n_epochs: int = 5,
    lr: float = 5e-3,
    seed: int = 42,
) -> Tuple[ConditionalFlowMatcher, List[float]]:
    """Train CFM to map random noise → identity pose (smoke-fit for tests)."""
    rng = np.random.default_rng(seed)
    cfm = ConditionalFlowMatcher(model=VelocityFieldMLP(hidden=32, seed=seed))
    target = SE3.identity()
    losses: List[float] = []

    for _ in range(n_epochs):
        noise = [random_se3(rng, translation_scale=0.5) for _ in range(n_samples)]
        data = [target for _ in range(n_samples)]
        loss = cfm.train_step_fd(noise, data, lr=lr, rng=rng)
        losses.append(loss)
    return cfm, losses


def geodesic_cfm_consistency_error(
    g0: SE3,
    g1: SE3,
    n_times: int = 11,
) -> float:
    """Max reconstruction error of geodesic CFM path endpoints.

    Verifies γ(0)=g0, γ(1)=g1 and that integrating body velocity recovers path.
    """
    err0 = se3_distance(geodesic(g0, g1, 0.0), g0)
    err1 = se3_distance(geodesic(g0, g1, 1.0), g1)
    # Integrate body velocity from g0
    xi = geodesic_velocity(g0, g1, 0.0)
    g = g0
    dt = 1.0 / (n_times - 1)
    max_path_err = 0.0
    for i in range(n_times):
        t = i * dt
        g_true = geodesic(g0, g1, t)
        max_path_err = max(max_path_err, se3_distance(g, g_true))
        if i < n_times - 1:
            g = g.compose(exp_map(dt * xi))
    return float(max(err0, err1, max_path_err))
