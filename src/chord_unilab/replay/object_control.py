"""Backend-independent Sharpa wrist and object-root control targets."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chord_unilab.replay.quaternion import (
    inverse_wxyz,
    multiply_wxyz,
    normalize_wxyz,
    to_rotvec_wxyz,
)


def sharpa_ball_wrist_torque(
    current_quaternion_wxyz: ArrayLike,
    target_quaternion_wxyz: ArrayLike,
) -> NDArray[np.float64]:
    """Return the GT child-frame target torque for one Sharpa ball wrist."""
    current = normalize_wxyz(current_quaternion_wxyz)
    target = normalize_wxyz(target_quaternion_wxyz)
    if np.dot(current, target) < 0.0:
        target = -target
    error = multiply_wxyz(inverse_wxyz(current), target)
    return np.clip(25.0 * to_rotvec_wxyz(error), -45.0, 45.0)


def object_root_voc(
    current_position: ArrayLike,
    target_position: ArrayLike,
    current_quaternion_wxyz: ArrayLike,
    target_quaternion_wxyz: ArrayLike,
    linear_velocity: ArrayLike,
    angular_velocity: ArrayLike,
    total_mass: float,
    root_inertia: ArrayLike,
    gravity: float = 9.81,
    voc_scale: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return GT mass-normalized VOC force and torque for an object root.

    ``linear_velocity`` is measured at the link origin and is used directly.
    ``root_inertia`` is either three principal moments, shape ``(3,)``, or the
    full root inertia tensor, shape ``(3,3)``; only ``trace(I) / 3`` enters the
    GT angular control law. ``gravity`` is a non-negative magnitude compensated
    along world ``+z``. As in GT, ``voc_scale`` multiplies the wrench before
    component-wise clamping, so a zero scale produces exactly zero wrench.
    """
    position = _vector(current_position, "current_position")
    target = _vector(target_position, "target_position")
    linear = _vector(linear_velocity, "linear_velocity")
    angular = _vector(angular_velocity, "angular_velocity")
    mass = _nonnegative_scalar(total_mass, "total_mass")
    gravity_magnitude = _nonnegative_scalar(gravity, "gravity")
    scale = _finite_scalar(voc_scale, "voc_scale")
    inertia = np.asarray(root_inertia, dtype=np.float64)
    if inertia.shape == (3,):
        inertia_effective = np.sum(inertia) / 3.0
    elif inertia.shape == (3, 3):
        inertia_effective = np.trace(inertia) / 3.0
    else:
        raise ValueError(
            f"root_inertia must have shape (3,) or (3, 3), got {inertia.shape}"
        )
    if not np.all(np.isfinite(inertia)):
        raise ValueError("root_inertia must contain only finite values")

    force = mass * (400.0 * (target - position) - 40.0 * linear)
    force = force + np.array([0.0, 0.0, mass * gravity_magnitude])
    orientation_error = multiply_wxyz(
        target_quaternion_wxyz, inverse_wxyz(current_quaternion_wxyz)
    )
    torque = inertia_effective * (
        400.0 * to_rotvec_wxyz(orientation_error) - 40.0 * angular
    )
    return (
        np.clip(scale * force, -200.0, 200.0),
        np.clip(scale * torque, -50.0, 50.0),
    )


def _vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _finite_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _nonnegative_scalar(value: float, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar
