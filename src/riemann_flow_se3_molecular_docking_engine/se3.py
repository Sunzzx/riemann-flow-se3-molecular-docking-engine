"""SE(3) special Euclidean group: rigid motions (rotation + translation).

Elements g = (R, t) ∈ SE(3) act on points x ∈ R³ via g·x = R x + t.
Lie algebra se(3) ≅ R⁶ = (ω, v) with exponential / logarithmic maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from riemann_flow_se3_molecular_docking_engine import so3

EPS = 1e-8


@dataclass(frozen=True)
class SE3:
    """Rigid transformation g = (R, t) ∈ SE(3)."""

    R: NDArray[np.floating]
    t: NDArray[np.floating]

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", np.asarray(self.R, dtype=float).reshape(3, 3))
        object.__setattr__(self, "t", np.asarray(self.t, dtype=float).reshape(3))

    @staticmethod
    def identity() -> "SE3":
        return SE3(np.eye(3), np.zeros(3))

    @staticmethod
    def from_matrix(T: NDArray[np.floating]) -> "SE3":
        T = np.asarray(T, dtype=float).reshape(4, 4)
        return SE3(T[:3, :3], T[:3, 3])

    def as_matrix(self) -> NDArray[np.floating]:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def inverse(self) -> "SE3":
        Rt = self.R.T
        return SE3(Rt, -Rt @ self.t)

    def compose(self, other: "SE3") -> "SE3":
        """self ∘ other: apply other first, then self."""
        return SE3(self.R @ other.R, self.R @ other.t + self.t)

    def __matmul__(self, other: "SE3") -> "SE3":
        return self.compose(other)

    def apply(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Transform point cloud (N, 3) or single point (3,)."""
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            return self.R @ points + self.t
        return points @ self.R.T + self.t

    def adjoint(self) -> NDArray[np.floating]:
        """Ad_g ∈ R^{6×6}: Ad_g [ω; v] = [R ω; [t]ˣ R ω + R v]."""
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = self.R
        Ad[3:, 3:] = self.R
        Ad[3:, :3] = so3.hat(self.t) @ self.R
        return Ad


def hat(xi: NDArray[np.floating]) -> NDArray[np.floating]:
    """se(3) hat: ξ = (ω, v) ∈ R⁶ → 4×4 twist matrix."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    omega, v = xi[:3], xi[3:]
    Xi = np.zeros((4, 4))
    Xi[:3, :3] = so3.hat(omega)
    Xi[:3, 3] = v
    return Xi


def vee(Xi: NDArray[np.floating]) -> NDArray[np.floating]:
    """Inverse of hat for se(3)."""
    Xi = np.asarray(Xi, dtype=float).reshape(4, 4)
    omega = so3.vee(Xi[:3, :3])
    v = Xi[:3, 3]
    return np.concatenate([omega, v])


def _V_matrix(omega: NDArray[np.floating]) -> NDArray[np.floating]:
    """Left Jacobian of SO(3) / SE(3) translation coupling matrix.

    V(ω) = I + ((1−cos θ)/θ²) [ω]ˣ + ((θ−sin θ)/θ³) [ω]ˣ²
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    if theta < EPS:
        return np.eye(3) + 0.5 * so3.hat(omega)

    # Stable coefficients; for large θ use asymptotic forms to avoid overflow
    K = so3.hat(omega)
    if theta > 1e8:
        # V ~ (1/θ²) terms vanish relatively; fall back to series-safe path
        # Reduce angle magnitude for coupling matrix via axis * (θ mod 2π) is wrong
        # for V; instead clamp using 1/θ expansions:
        a = (1.0 - np.cos(theta)) / (theta ** 2)
        b = 1.0 / (theta ** 2)  # (θ - sin θ)/θ³ → 1/θ²
    else:
        a = (1.0 - np.cos(theta)) / (theta ** 2)
        b = (theta - np.sin(theta)) / (theta ** 3)
    return np.eye(3) + a * K + b * (K @ K)


def _V_inv(omega: NDArray[np.floating]) -> NDArray[np.floating]:
    """Inverse of V(ω)."""
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    if theta < EPS:
        return np.eye(3) - 0.5 * so3.hat(omega)

    K = so3.hat(omega)
    half = 0.5
    # cot(θ/2) = cos(θ/2)/sin(θ/2)
    s = np.sin(theta)
    c = np.cos(theta)
    # Standard formula:
    # V^{-1} = I - ½[ω]ˣ + (1/θ² - (1+cos θ)/(2 θ sin θ)) [ω]ˣ²
    if abs(s) < EPS:
        beta = 1.0 / 12.0  # series limit
    else:
        beta = (1.0 / (theta ** 2)) - ((1.0 + c) / (2.0 * theta * s))
    return np.eye(3) - half * K + beta * (K @ K)


def exp_map(xi: NDArray[np.floating]) -> SE3:
    """Exponential map exp: se(3) → SE(3).

    R = exp([ω]ˣ)
    t = V(ω) v
    """
    xi = np.asarray(xi, dtype=float).reshape(6)
    omega, v = xi[:3], xi[3:]
    R = so3.exp_map(omega)
    t = _V_matrix(omega) @ v
    return SE3(R, t)


def log_map(g: SE3) -> NDArray[np.floating]:
    """Logarithmic map log: SE(3) → se(3) ≅ R⁶.

    ω = log(R)
    v = V(ω)^{-1} t
    """
    omega = so3.log_map(g.R)
    v = _V_inv(omega) @ g.t
    return np.concatenate([omega, v])


def geodesic(g0: SE3, g1: SE3, t: float) -> SE3:
    """Riemannian geodesic on SE(3) with left-invariant metric.

    γ(t) = g0 · exp(t · log(g0^{-1} ∘ g1))
    """
    t = float(np.clip(t, 0.0, 1.0))
    xi = log_map(g0.inverse().compose(g1))
    return g0.compose(exp_map(t * xi))


def geodesic_velocity(g0: SE3, g1: SE3, t: float) -> NDArray[np.floating]:
    """Body-frame twist velocity of the constant-speed geodesic (independent of t).

    ξ = log(g0^{-1} g1)  ∈ se(3)
    """
    _ = t
    return log_map(g0.inverse().compose(g1))


def se3_distance(g0: SE3, g1: SE3, rot_weight: float = 1.0, trans_weight: float = 1.0) -> float:
    """Weighted bi-invariant distance on SE(3).

    d² = w_r ‖ω‖² + w_t ‖v‖²
    """
    xi = log_map(g0.inverse().compose(g1))
    return float(
        np.sqrt(rot_weight * np.sum(xi[:3] ** 2) + trans_weight * np.sum(xi[3:] ** 2))
    )


def random_se3(
    rng: np.random.Generator | None = None,
    translation_scale: float = 1.0,
) -> SE3:
    """Random rigid motion: Haar rotation + Gaussian translation."""
    if rng is None:
        rng = np.random.default_rng()
    R = so3.random_rotation(rng)
    t = rng.standard_normal(3) * translation_scale
    return SE3(R, t)


def interpolate_path(
    g0: SE3,
    g1: SE3,
    n_steps: int = 20,
) -> List[SE3]:
    """Discretize geodesic from g0 to g1 into n_steps+1 poses."""
    return [geodesic(g0, g1, i / n_steps) for i in range(n_steps + 1)]


def relative_pose(g_ref: SE3, g: SE3) -> SE3:
    """Pose of g in the frame of g_ref: g_ref^{-1} ∘ g."""
    return g_ref.inverse().compose(g)


def align_point_clouds_kabsch(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
) -> SE3:
    """Kabsch / Umeyama rigid alignment: find g minimizing ‖g·src − dst‖_F.

    Requires corresponding point sets of shape (N, 3).
    """
    src = np.asarray(src, dtype=float).reshape(-1, 3)
    dst = np.asarray(dst, dtype=float).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError("src and dst must have the same shape")
    if src.shape[0] < 3:
        raise ValueError("need at least 3 corresponding points")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, np.sign(d) if d != 0 else 1.0])
    R = Vt.T @ S @ U.T
    t = mu_d - R @ mu_s
    return SE3(R, t)
