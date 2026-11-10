"""SO(3) special orthogonal group: rotations in 3D Euclidean space.

The Lie algebra so(3) is identified with R^3 via the hat / vee isomorphism.
Exponential and logarithmic maps use Rodrigues' formula.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

EPS = 1e-8
PI = np.pi


def hat(omega: NDArray[np.floating]) -> NDArray[np.floating]:
    """Map so(3) vector ω ∈ R³ to skew-symmetric matrix [ω]ˣ ∈ so(3).

    [ω]ˣ = [[  0, -ωz,  ωy],
            [ ωz,   0, -ωx],
            [-ωy,  ωx,   0]]
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    wx, wy, wz = omega
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=float,
    )


def vee(W: NDArray[np.floating]) -> NDArray[np.floating]:
    """Inverse of hat: skew-symmetric matrix → R³."""
    W = np.asarray(W, dtype=float).reshape(3, 3)
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)


def skew_symmetric_check(W: NDArray[np.floating], tol: float = 1e-6) -> bool:
    """Return True if W is approximately skew-symmetric."""
    W = np.asarray(W, dtype=float)
    return bool(np.allclose(W + W.T, 0.0, atol=tol))


def rotation_matrix_check(R: NDArray[np.floating], tol: float = 1e-5) -> bool:
    """Return True if R ∈ SO(3): RᵀR = I and det(R) = +1."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    ortho = np.allclose(R.T @ R, np.eye(3), atol=tol)
    det_ok = abs(float(np.linalg.det(R)) - 1.0) < tol
    return bool(ortho and det_ok)


def exp_map(omega: NDArray[np.floating]) -> NDArray[np.floating]:
    """Rodrigues formula: exp: so(3) → SO(3).

    R = I + (sin θ / θ) [ω]ˣ + ((1 − cos θ) / θ²) [ω]ˣ²
    where θ = ‖ω‖.
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    if theta < EPS:
        # First-order: R ≈ I + [ω]ˣ
        return np.eye(3) + hat(omega)

    K = hat(omega / theta)
    return (
        np.eye(3)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )


def log_map(R: NDArray[np.floating]) -> NDArray[np.floating]:
    """Logarithmic map: SO(3) → so(3) ≅ R³.

    θ = arccos((tr(R) − 1) / 2)
    ω = (θ / (2 sin θ)) vee(R − Rᵀ)
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    # Project to nearest SO(3) for numerical safety
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    cos_theta = float((np.trace(R) - 1.0) * 0.5)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < EPS:
        return vee(0.5 * (R - R.T))

    if abs(theta - PI) < 1e-4:
        # Near π singularity: use eigenvector of eigenvalue +1
        # R + I has rank 1; the nullspace of R - I is the axis
        M = R + np.eye(3)
        # Largest column of M is proportional to the axis
        col_norms = np.linalg.norm(M, axis=0)
        axis = M[:, int(np.argmax(col_norms))]
        axis = axis / (np.linalg.norm(axis) + EPS)
        return axis * PI

    coeff = theta / (2.0 * np.sin(theta))
    return coeff * vee(R - R.T)


def geodesic(
    R0: NDArray[np.floating],
    R1: NDArray[np.floating],
    t: float,
) -> NDArray[np.floating]:
    """Geodesic on SO(3): γ(t) = R0 · exp(t · log(R0ᵀ R1)), t ∈ [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    R0 = np.asarray(R0, dtype=float).reshape(3, 3)
    R1 = np.asarray(R1, dtype=float).reshape(3, 3)
    omega = log_map(R0.T @ R1)
    return R0 @ exp_map(t * omega)


def geodesic_velocity(
    R0: NDArray[np.floating],
    R1: NDArray[np.floating],
    t: float,
) -> NDArray[np.floating]:
    """Left-invariant body velocity of the SO(3) geodesic at time t.

    v_body = log(R0ᵀ R1)  (constant in body frame for constant-speed geodesic)
    Instantaneous spatial angular velocity ω_s = R(t) · v_body · R(t)ᵀ
    Returns body-frame omega ∈ R³ (constant along geodesic).
    """
    R0 = np.asarray(R0, dtype=float).reshape(3, 3)
    R1 = np.asarray(R1, dtype=float).reshape(3, 3)
    _ = t  # constant speed geodesic → body velocity independent of t
    return log_map(R0.T @ R1)


def random_rotation(rng: np.random.Generator | None = None) -> NDArray[np.floating]:
    """Uniform random rotation via QR of Gaussian matrix (Haar measure)."""
    if rng is None:
        rng = np.random.default_rng()
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Fix signs so det = +1 and diagonal of R positive
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rotation_distance(R0: NDArray[np.floating], R1: NDArray[np.floating]) -> float:
    """Geodesic (bi-invariant) distance d(R0, R1) = ‖log(R0ᵀ R1)‖ = θ ∈ [0, π]."""
    return float(np.linalg.norm(log_map(np.asarray(R0).T @ np.asarray(R1))))


def apply_rotation(
    R: NDArray[np.floating],
    points: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply rotation R to point cloud of shape (N, 3)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        return R @ points
    return points @ R.T


def slerp_axis_angle(
    axis: NDArray[np.floating],
    angle0: float,
    angle1: float,
    t: float,
) -> NDArray[np.floating]:
    """Interpolate rotation about a fixed axis."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / (np.linalg.norm(axis) + EPS)
    angle = (1.0 - t) * angle0 + t * angle1
    return exp_map(axis * angle)
