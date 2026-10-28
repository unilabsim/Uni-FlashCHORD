"""Pure NumPy quaternion operations using scalar-first ``wxyz`` order."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize_wxyz(quaternion: ArrayLike) -> NDArray[np.float64]:
    """Return a unit quaternion, preserving the input's double-cover sign."""
    q = _as_vector(quaternion, 4, "quaternion")
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("quaternion must be non-zero")
    return q / norm


def multiply_wxyz(left: ArrayLike, right: ArrayLike) -> NDArray[np.float64]:
    """Return the Hamilton product ``left * right`` in ``wxyz`` order."""
    a = _as_vector(left, 4, "left")
    b = _as_vector(right, 4, "right")
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def inverse_wxyz(quaternion: ArrayLike) -> NDArray[np.float64]:
    """Return the multiplicative inverse, including for non-unit quaternions."""
    q = _as_vector(quaternion, 4, "quaternion")
    norm_squared = np.dot(q, q)
    if norm_squared == 0.0:
        raise ValueError("quaternion must be non-zero")
    return q * np.array([1.0, -1.0, -1.0, -1.0]) / norm_squared


def rotate_wxyz(quaternion: ArrayLike, vector: ArrayLike) -> NDArray[np.float64]:
    """Rotate a three-vector by a quaternion; non-unit quaternions are normalized."""
    q = normalize_wxyz(quaternion)
    v = _as_vector(vector, 3, "vector")
    return np.asarray(
        v + 2.0 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v), dtype=np.float64
    )


def to_rotvec_wxyz(quaternion: ArrayLike) -> NDArray[np.float64]:
    """Return the shortest-path rotation vector ``axis * angle``."""
    q = normalize_wxyz(quaternion)
    if q[0] < 0.0:
        q = -q
    vector = q[1:]
    sine = np.linalg.norm(vector)
    if sine < 1.0e-8:
        return 2.0 * vector
    angle = 2.0 * np.arctan2(sine, q[0])
    return vector * (angle / sine)


def from_rotvec_wxyz(rotvec: ArrayLike) -> NDArray[np.float64]:
    """Convert a rotation vector to a unit quaternion; zero maps to identity."""
    rotation = _as_vector(rotvec, 3, "rotvec")
    angle = np.linalg.norm(rotation)
    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if angle < 1.0e-8:
        vector_scale = 0.5 - angle * angle / 48.0
    else:
        vector_scale = np.sin(0.5 * angle) / angle
    return np.concatenate(([np.cos(0.5 * angle)], vector_scale * rotation))


def _as_vector(value: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector
