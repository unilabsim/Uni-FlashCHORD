"""Numerical FlashChord residual and per-substep control semantics."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from chord_unilab.tasks.sharpa_chord.actions import ResidualHandPoseAction
from chord_unilab.tasks.sharpa_chord.math import from_euler_xyz, multiply


def _reference():
    wrist_pos = torch.tensor([[[0.1, -0.2, 0.5], [0.2, 0.3, 0.6]]])
    z90 = torch.tensor([2.0**-0.5, 0.0, 0.0, 2.0**-0.5])
    wrist_quat = torch.stack((z90, torch.tensor([1.0, 0.0, 0.0, 0.0])))[None]
    fingers = torch.full((1, 2, 22), 0.2)
    objects = torch.zeros(1, 1, 3)
    object_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    return (
        wrist_pos,
        wrist_quat,
        fingers,
        objects,
        object_quat,
        torch.zeros(1, 0),
        torch.zeros(1, 2, 6, 3),
    )


def _processing_action():
    action = cast(Any, object.__new__(ResidualHandPoseAction))
    action.cfg = SimpleNamespace(ema=0.3)
    action._raw_action = torch.zeros(1, 56)
    action.filtered_action = torch.zeros_like(action._raw_action)
    action.processed_target = torch.zeros(1, 58)
    action.action_l2 = torch.zeros(1)
    action.action_rate_l2 = torch.zeros(1)
    action.scale = torch.tensor(([0.05] * 3 + [0.15] * 25) * 2)
    action.clip = torch.tensor(([0.2] * 3 + [1.0] * 25) * 2)
    action.input_scale = action.clip / ((1.0 - action.cfg.ema) * action.scale)
    action.command = SimpleNamespace(current_reference=_reference)
    action._control = torch.zeros(1, 50)
    action.wrist_pos_ctrl = torch.arange(6)
    action.finger_ctrl = torch.arange(6, 50)
    action.object_ctrl = torch.empty(0, dtype=torch.int64)
    action.object_vel_ctrl = torch.empty(0, dtype=torch.int64)
    action._action_finite_valid = torch.ones((), dtype=torch.bool)
    action._env = SimpleNamespace(num_envs=1)
    return action


def test_rational_ema_and_right_multiply_quaternion() -> None:
    action = _processing_action()
    policy = torch.ones(1, 56)
    action.process_actions(policy)
    torch.testing.assert_close(action.raw_action, action.input_scale[None])
    torch.testing.assert_close(action.filtered_action, action.clip[None])
    torch.testing.assert_close(
        action.action_l2, action.input_scale.square().sum()[None]
    )
    torch.testing.assert_close(action.action_rate_l2, action.action_l2)

    action.process_actions(policy)
    torch.testing.assert_close(action.filtered_action, action.clip[None])
    assert action.action_rate_l2.item() == 0.0
    ref_pos, ref_quat, ref_finger, *_ = _reference()
    target = action.processed_target.reshape(1, 2, 29)
    torch.testing.assert_close(target[:, :, :3], ref_pos + 0.2)
    residual = from_euler_xyz(torch.ones(1, 2, 3))
    torch.testing.assert_close(target[:, :, 3:7], multiply(ref_quat, residual))
    torch.testing.assert_close(target[:, :, 7:], ref_finger + 1.0)
    assert not torch.allclose(
        target[0, 0, 3:7], multiply(residual[0, 0], ref_quat[0, 0])
    )

    action.process_actions(-policy)
    torch.testing.assert_close(action.raw_action, -action.input_scale[None])
    torch.testing.assert_close(action.filtered_action, -0.7 * action.clip[None])


def test_substep_wrist_child_frame_and_link_origin_voc() -> None:
    action = cast(Any, object.__new__(ResidualHandPoseAction))
    action.cfg = SimpleNamespace(wrist_kp=25.0, wrist_tau_max=45.0)
    action.processed_target = torch.zeros(1, 58)
    action.ball_qpos = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    action.ball_qvel = torch.tensor([[0, 1, 2], [3, 4, 5]])
    action.object_qvel = torch.empty(0, dtype=torch.int64)
    action.object_reference_ids = torch.tensor([0])
    action.object_mass = torch.tensor([2.0])
    action.object_inertia = torch.tensor([6.0])
    action.object_body_ids = torch.tensor([0])
    action.joint_force = torch.zeros(1, 6)
    action.body_force = torch.zeros(1, 1, 6)
    action.object_wrench = torch.zeros(1, 1, 6)
    action._device = torch.device("cpu")
    action.gravity_w = torch.tensor([0.0, 0.0, 9.81])
    action._env = SimpleNamespace(num_envs=1)
    current_z90 = torch.tensor([2.0**-0.5, 0.0, 0.0, 2.0**-0.5])
    local_x = from_euler_xyz(torch.tensor([0.04, 0.0, 0.0]))
    target_quat = multiply(current_z90, local_x)
    targets = action.processed_target.reshape(1, 2, 29)
    targets[0, 0, 3:7] = target_quat
    targets[0, 1, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    qpos = torch.zeros(1, 8)
    qpos[0, :4] = current_z90
    qpos[0, 4:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    captured = {}

    class Backend:
        def get_state_views(self, names):
            return {"qpos": qpos, "qvel": torch.zeros(1, 6)}

        def set_generalized_force_tensor(self, *, joint_f, body_f):
            captured["joint"] = joint_f.clone()
            captured["body"] = body_f.clone()

    action._env.backend = Backend()
    reference = list(_reference())
    reference[3] = torch.tensor([[[1.0, -1.0, 0.0]]])
    reference[4] = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    action.active_reference = tuple(reference)
    action.command = SimpleNamespace(
        current_reference=lambda: tuple(reference),
        applied_voc_scale=torch.tensor([0.5]),
        object_state=lambda: (
            torch.zeros(1, 1, 3),
            torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            torch.tensor([[[1.0, -2.0, 0.5]]]),
            torch.zeros(1, 1, 3),
        ),
    )
    action.apply_actions()
    np.testing.assert_allclose(
        captured["joint"][0, :3].numpy(), [1.0, 0.0, 0.0], atol=1e-6
    )
    expected_force = 0.5 * (
        2.0
        * (
            400.0 * torch.tensor([1.0, -1.0, 0.0])
            - 40.0 * torch.tensor([1.0, -2.0, 0.5])
        )
        + torch.tensor([0.0, 0.0, 2.0 * 9.81])
    )
    expected_force = expected_force.clamp(-200.0, 200.0)
    torch.testing.assert_close(captured["body"][0, 0, :3], expected_force)
    torch.testing.assert_close(captured["body"][0, 0, 3:], torch.zeros(3))


def test_substep_voc_broadcasts_mass_and_inertia_per_object() -> None:
    action = cast(Any, object.__new__(ResidualHandPoseAction))
    action.cfg = SimpleNamespace(wrist_kp=25.0, wrist_tau_max=45.0)
    action._env = SimpleNamespace(num_envs=1)
    action.processed_target = torch.zeros(1, 58)
    targets = action.processed_target.reshape(1, 2, 29)
    targets[:, :, 3] = 1.0
    action.ball_qpos = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    action.object_qvel = torch.empty(0, dtype=torch.int64)
    action.object_reference_ids = torch.tensor([0, 1])
    action.object_mass = torch.tensor([2.0, 3.0])
    action.object_inertia = torch.tensor([4.0, 5.0])
    action.gravity_w = torch.tensor([0.0, 0.0, 9.81])
    action.command = SimpleNamespace(applied_voc_scale=torch.ones(1))
    reference = list(_reference())
    reference[3] = torch.zeros(1, 2, 3)
    reference[4] = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(1, 2, 1)
    reference[6] = torch.zeros(1, 2, 6, 3)
    action.active_reference = tuple(reference)
    qpos = torch.zeros(1, 8)
    qpos[:, 0] = 1.0
    qpos[:, 4] = 1.0
    object_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(1, 2, 1)
    _, drive, wrench = action._evaluate_forces(
        qpos,
        torch.zeros(1, 6),
        torch.zeros(1, 2, 3),
        object_quat,
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
    )
    assert drive.shape == (1, 0)
    torch.testing.assert_close(
        wrench[..., :3],
        torch.tensor([0.0, 0.0, 9.81]) * action.object_mass[None, :, None],
    )
    torch.testing.assert_close(wrench[..., 3:], torch.zeros(1, 2, 3))
