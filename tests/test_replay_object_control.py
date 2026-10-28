"""Independent numerical checks for the replay object's GT control laws."""

import numpy as np
import pytest

from chord_unilab.replay import (
    from_rotvec_wxyz,
    inverse_wxyz,
    multiply_wxyz,
    normalize_wxyz,
    object_root_voc,
    rotate_wxyz,
    sharpa_ball_wrist_torque,
    to_rotvec_wxyz,
)


def quaternion_from_rotation(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec)
    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate(([np.cos(angle / 2.0)], np.sin(angle / 2.0) * rotvec / angle))


def hamilton_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[0] * b[0] - np.dot(a[1:], b[1:]),
            *(a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])),
        ]
    )


def test_quaternion_normalize_inverse_and_multiply_are_independent_of_unit_input() -> (
    None
):
    np.testing.assert_array_equal(
        normalize_wxyz([-2.0, 0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0, 0.0]),
    )

    half_sqrt = np.sqrt(0.5)
    qx = np.array([half_sqrt, half_sqrt, 0.0, 0.0])
    qy = np.array([half_sqrt, 0.0, half_sqrt, 0.0])
    np.testing.assert_allclose(
        multiply_wxyz(qx, qy), np.full(4, 0.5), rtol=0.0, atol=1.0e-15
    )

    quaternion = np.array([2.0, 1.0, -2.0, 0.5])
    expected_inverse = np.array([2.0, -1.0, 2.0, -0.5]) / 9.25
    np.testing.assert_allclose(inverse_wxyz(quaternion), expected_inverse)
    np.testing.assert_allclose(
        hamilton_wxyz(quaternion, expected_inverse),
        np.array([1.0, 0.0, 0.0, 0.0]),
        atol=1.0e-15,
    )


def test_quaternion_rotation_and_rotvec_handle_nonunit_and_sign() -> None:
    np.testing.assert_allclose(
        rotate_wxyz([5.0, 0.0, 0.0, 5.0], [1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        atol=1.0e-15,
    )

    expected_rotvec = np.array([1.2, -0.6, 0.9])
    quaternion = -7.0 * quaternion_from_rotation(expected_rotvec)
    np.testing.assert_allclose(
        to_rotvec_wxyz(quaternion), expected_rotvec, rtol=0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        from_rotvec_wxyz(expected_rotvec),
        quaternion_from_rotation(expected_rotvec),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_array_equal(
        from_rotvec_wxyz([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_zero_quaternion_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        normalize_wxyz([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-zero"):
        inverse_wxyz([0.0, 0.0, 0.0, 0.0])


def test_sharpa_wrist_uses_child_frame_error_and_target_sign() -> None:
    current = quaternion_from_rotation(np.array([0.0, 0.0, np.pi / 2.0]))
    local_rotation = np.array([0.04, 0.0, 0.0])
    local_target = quaternion_from_rotation(local_rotation)
    target = hamilton_wxyz(current, local_target)

    expected = 25.0 * local_rotation
    np.testing.assert_allclose(
        sharpa_ball_wrist_torque(3.0 * current, target), expected, atol=1.0e-14
    )
    np.testing.assert_allclose(
        sharpa_ball_wrist_torque(current, -5.0 * target), expected, atol=1.0e-14
    )

    half_turn = quaternion_from_rotation(np.array([np.pi, 0.0, 0.0]))
    np.testing.assert_allclose(
        sharpa_ball_wrist_torque([1.0, 0.0, 0.0, 0.0], half_turn),
        np.array([45.0, 0.0, 0.0]),
        atol=1.0e-14,
    )


def test_object_root_voc_matches_independent_gt_formula() -> None:
    position = np.array([1.0, -2.0, 0.5])
    target_position = np.array([1.01, -2.03, 0.55])
    linear_velocity = np.array([0.2, -0.1, 0.05])
    angular_velocity = np.array([0.01, -0.02, 0.03])
    rotation = np.array([0.002, -0.003, 0.004])
    mass = 2.0
    principal_inertia = np.array([3.0, 6.0, 12.0])
    inertia_effective = 7.0
    gravity = 9.81

    force, torque = object_root_voc(
        position,
        target_position,
        [1.0, 0.0, 0.0, 0.0],
        quaternion_from_rotation(rotation),
        linear_velocity,
        angular_velocity,
        mass,
        principal_inertia,
        gravity=gravity,
    )

    expected_force = mass * (
        400.0 * (target_position - position) - 40.0 * linear_velocity
    ) + np.array([0.0, 0.0, mass * gravity])
    expected_torque = inertia_effective * (400.0 * rotation - 40.0 * angular_velocity)
    np.testing.assert_allclose(force, expected_force, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(torque, expected_torque, rtol=0.0, atol=1.0e-13)


def test_object_root_voc_full_tensor_matches_principal_trace() -> None:
    arguments = (
        np.array([0.1, 0.2, 0.3]),
        np.array([0.2, 0.1, 0.4]),
        quaternion_from_rotation(np.array([0.2, 0.0, 0.0])),
        quaternion_from_rotation(np.array([0.1, -0.1, 0.2])),
        np.array([0.3, -0.2, 0.1]),
        np.array([0.02, 0.03, -0.01]),
        1.7,
    )
    principal = np.array([2.0, 4.0, 6.0])
    full_tensor = np.array([[2.0, 8.0, 1.0], [-3.0, 4.0, 2.0], [5.0, 7.0, 6.0]])

    principal_wrench = object_root_voc(*arguments, principal)
    tensor_wrench = object_root_voc(*arguments, full_tensor)
    np.testing.assert_array_equal(principal_wrench[0], tensor_wrench[0])
    np.testing.assert_array_equal(principal_wrench[1], tensor_wrench[1])


def test_object_root_voc_uses_link_origin_velocity_without_com_correction() -> None:
    linear_velocity = np.array([1.0, -2.0, 0.5])
    angular_velocity = np.array([0.2, 0.0, 0.3])
    mass = 4.0
    inertia_effective = 2.0

    force, torque = object_root_voc(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        linear_velocity,
        angular_velocity,
        mass,
        np.full(3, inertia_effective),
        gravity=0.0,
    )

    np.testing.assert_allclose(
        force, np.clip(-40.0 * mass * linear_velocity, -200.0, 200.0)
    )
    np.testing.assert_allclose(torque, -40.0 * inertia_effective * angular_velocity)


def test_object_root_voc_scales_then_clamps_and_zero_scale_is_exact_zero() -> None:
    arguments = (
        np.zeros(3),
        np.array([1.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([1.0, -1.0, 0.0]),
        1.0,
        np.full(3, 3.0),
    )
    force, torque = object_root_voc(*arguments, gravity=0.0, voc_scale=0.25)
    np.testing.assert_array_equal(force, np.array([100.0, -100.0, 0.0]))
    np.testing.assert_array_equal(torque, np.array([-30.0, 30.0, 0.0]))

    zero_force, zero_torque = object_root_voc(*arguments, gravity=9.81, voc_scale=0.0)
    np.testing.assert_array_equal(zero_force, np.zeros(3))
    np.testing.assert_array_equal(zero_torque, np.zeros(3))
