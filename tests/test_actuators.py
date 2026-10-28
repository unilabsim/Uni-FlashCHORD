"""Owner-side mapping between per-joint targets and paired MJCF actuators."""

import numpy as np
import pytest

from chord_unilab.assets.actuators import position_velocity_control


def test_paired_control_maps_entity_qualified_actuators() -> None:
    ctrl = position_velocity_control(
        ["other/other_joint_pos", "left_hand/wrist_pos", "left_hand/wrist_vel"],
        ["wrist"],
        np.array([[0.25], [-0.5]], dtype=np.float32),
        np.array([[1.0], [-2.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        ctrl,
        np.array(
            [[0.0, 0.25, 1.0], [0.0, -0.5, -2.0]],
            dtype=np.float32,
        ),
    )


def test_paired_control_defaults_velocity_to_zero_and_rejects_ambiguity() -> None:
    ctrl = position_velocity_control(
        ["owner/a_pos", "owner/a_vel"], ["a"], np.array([[0.75]], dtype=np.float32)
    )
    np.testing.assert_array_equal(ctrl, np.array([[0.75, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="exactly one"):
        position_velocity_control(
            ["j_pos", "a/j_pos", "j_vel"], ["j"], np.zeros((1, 1), dtype=np.float32)
        )
