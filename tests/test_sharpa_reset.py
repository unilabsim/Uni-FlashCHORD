"""Per-env reference reset, VOC settling, and curriculum transitions."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from chord_unilab.assets.reference_state import reference_state_patch
from chord_unilab.tasks.sharpa_chord.commands import (
    VOC_THRESHOLDS,
    ReferenceCommand,
    voc_curriculum,
)


class _Reference:
    num_frames = 3

    def wrist_pos_w(self, side):
        return np.array(
            [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.2, 0.0, 0.5]], dtype=np.float32
        )

    def wrist_quat_w(self, side):
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1))

    def finger_joint_names(self, side):
        return ("finger",)

    def finger_joint_pos(self, side):
        return np.array([[0.2], [0.4], [0.8]], dtype=np.float32)

    def object_body_names(self):
        return ("root",)

    def object_body_pos_w(self):
        return np.array(
            [[[0.0, 0.0, 0.2]], [[0.1, 0.0, 0.2]], [[0.2, 0.0, 0.2]]], dtype=np.float32
        )

    def object_body_quat_w(self):
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1, 1))

    def object_articulation(self):
        return np.zeros((3, 0), dtype=np.float32)


def _assembly():
    hand = SimpleNamespace(
        side="right", entity_name="right_hand", joint_names=("finger",)
    )
    obj = SimpleNamespace(
        entity_name="object",
        joint_names=(),
        spec=SimpleNamespace(root_reference_name="root", articulations=()),
    )
    return SimpleNamespace(hand_bindings=(hand,), bindings=(obj,))


def test_reference_patch_supports_per_env_frames_and_finger_clamp() -> None:
    request = reference_state_patch(
        cast(Any, _Reference()),
        cast(Any, _assembly()),
        np.array([0, 2]),
        env_ids=(0, 1),
        finger_scale=np.array([0.5, 0.25]),
        finger_limits={"right": (np.array([0.0]), np.array([0.1]))},
    )
    hand, obj = cast(Any, request.patches)
    np.testing.assert_allclose(hand.root_pose[:, 0], [0.0, 0.2])
    np.testing.assert_allclose(hand.joint_positions, [[0.1], [0.1]])
    np.testing.assert_allclose(obj.root_pose[:, 0], [0.0, 0.2])
    np.testing.assert_array_equal(hand.root_velocity, 0.0)

    exact = reference_state_patch(cast(Any, _Reference()), cast(Any, _assembly()), 1)
    np.testing.assert_allclose(cast(Any, exact.patches[0]).joint_positions, [[0.4]])


def _settling_command():
    command = cast(Any, object.__new__(ReferenceCommand))
    command.cfg = SimpleNamespace(voc_decay_steps=20, curriculum_enabled=False)
    command._env = SimpleNamespace(
        reset_buf=torch.tensor([False, True]),
        common_step_counter=0,
        state=SimpleNamespace(info={"log": {}}),
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )
    command.curriculum_stage = 0
    command.first_frame_probability = 0.1
    command._last_published_stage = -1
    command.steps_since_reset = torch.zeros(2, dtype=torch.int64)
    command.timestep = torch.tensor([4, 7], dtype=torch.int64)
    command.applied_voc_scale = torch.ones(2)
    command.target_voc_scale = torch.tensor(0.1)
    command.command_counter = torch.zeros(2, dtype=torch.int64)
    command.update_curriculum = lambda: None
    command._current_reference = None
    return command


def test_voc_holds_reference_for_exactly_twenty_steps() -> None:
    command = _settling_command()
    for expected_age in range(1, 20):
        command._update_command()
        assert command.steps_since_reset.tolist() == [expected_age, 0]
        assert command.timestep.tolist() == [4, 7]
        torch.testing.assert_close(command.applied_voc_scale, torch.ones(2))
    command._update_command()
    assert command.steps_since_reset.tolist() == [20, 0]
    assert command.timestep.tolist() == [5, 7]
    torch.testing.assert_close(command.applied_voc_scale, torch.tensor([0.1, 1.0]))


def test_reference_cache_is_reused_until_command_update() -> None:
    command = _settling_command()
    command.num_frames = 3
    command.ref_wrist_pos = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    command.ref_wrist_quat = torch.ones(3, 2, 4)
    command.ref_finger = torch.ones(3, 2, 1)
    command.ref_object_pos = torch.ones(3, 1, 3)
    command.ref_object_quat = torch.ones(3, 1, 4)
    command.ref_object_articulation = torch.ones(3, 0)
    command.ref_hand_keypoints = torch.ones(3, 2, 6, 3)

    first = command.current_reference()
    assert command.current_reference() is first
    command._update_command()
    second = command.current_reference()
    assert second is not first
    assert command.current_reference() is second


def test_contact_state_is_cached_within_completed_step() -> None:
    command = cast(Any, object.__new__(ReferenceCommand))
    command.wrist_state = lambda: (
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, 4),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
    )
    command.object_state = lambda: (
        torch.zeros(1, 1, 3),
        torch.ones(1, 1, 4),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
    )
    reads: list[int] = []

    def read(*_args):
        reads.append(len(reads))
        return SimpleNamespace(calls=len(reads))

    command.contact = SimpleNamespace(read=read)
    command._env = SimpleNamespace(common_step_counter=7)
    command._contact_state = None
    command._contact_state_step = None
    command._contact_cache_allowed = True

    first = command.contact_state()
    assert command.contact_state() is first
    assert len(reads) == 1
    command._env.common_step_counter = 8
    second = command.contact_state()
    assert second is not first
    assert len(reads) == 2
    command._contact_cache_allowed = False
    assert command.contact_state() is not second
    assert len(reads) == 3


def test_curriculum_stage_updates_settled_rows_immediately() -> None:
    command = cast(Any, object.__new__(ReferenceCommand))
    command.cfg = SimpleNamespace(
        curriculum_enabled=True,
        voc_decay_steps=20,
        reset_to_first_frame_probability=0.1,
    )
    command._env = SimpleNamespace(
        common_step_counter=VOC_THRESHOLDS[0] // 2, num_envs=2
    )
    command.curriculum_stage = 0
    command.target_voc_scale = torch.tensor(1.0)
    command.applied_voc_scale = torch.ones(2)
    command.steps_since_reset = torch.tensor([20, 3])
    command.first_frame_probability = 0.1
    command.update_curriculum()
    assert command.curriculum_stage == 1
    assert command.target_voc_scale.item() == 0.75
    torch.testing.assert_close(command.applied_voc_scale, torch.tensor([0.75, 1.0]))

    command._env.common_step_counter = VOC_THRESHOLDS[-2] // 2
    command.update_curriculum()
    assert command.curriculum_stage == len(VOC_THRESHOLDS) - 1
    assert command.first_frame_probability == 0.5
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: command)
    )
    assert voc_curriculum(cast(Any, env)) == {
        "target_voc_scale": 0.0,
        "stage": float(len(VOC_THRESHOLDS) - 1),
        "first_frame_probability": 0.5,
    }

    command._env.common_step_counter = VOC_THRESHOLDS[-1]
    command.steps_since_reset[:] = 20
    command.update_curriculum()
    assert command.curriculum_stage == len(VOC_THRESHOLDS)
    assert command.first_frame_probability == 0.5
    torch.testing.assert_close(command.applied_voc_scale, torch.zeros(2))
