"""Exact observation offsets, frames, contact ordering, and critic context."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from omegaconf import OmegaConf

from chord_unilab.tasks.sharpa_chord.commands import (
    ReferenceCommand,
    body_com_offsets,
)
from chord_unilab.tasks.sharpa_chord.math import conjugate, from_euler_xyz, multiply
from chord_unilab.tasks.sharpa_chord.observations import (
    PolicyObservation,
    critic_context,
)


def _observation():
    observation = object.__new__(PolicyObservation)
    observation.finger_qpos = torch.arange(44)
    observation.finger_qvel = torch.arange(44)
    observation.finger_lower = torch.zeros(44)
    observation.finger_upper = torch.ones(44)
    wrist_pos = torch.tensor([[[0.1, 0.2, 0.3], [-0.1, 0.0, 0.4]]])
    wrist_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]])
    ref_pos = wrist_pos + 0.1
    ref_quat = from_euler_xyz(torch.tensor([0.0, 0.0, 0.2]))[None, None].expand(1, 2, 4)
    ref_finger = torch.full((1, 2, 22), 0.5)
    object_pos = torch.tensor([[[0.4, -0.2, 0.7]]])
    object_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    contact = SimpleNamespace(
        position_b=torch.full((1, 2, 1, 17, 3), 0.3),
        direction_b=torch.nn.functional.normalize(
            torch.tensor([1.0, 2.0, 3.0]), dim=0
        ).expand(1, 2, 1, 17, 3),
    )
    command = cast(Any, object.__new__(ReferenceCommand))
    vars(command).update(
        num_bodies=1,
        wrist_state=lambda: (
            wrist_pos,
            wrist_quat,
            torch.ones(1, 2, 3),
            torch.full((1, 2, 3), 0.2),
        ),
        object_state=lambda: (
            object_pos,
            object_quat,
            torch.full((1, 1, 3), 0.4),
            torch.full((1, 1, 3), -0.3),
        ),
        current_reference=lambda: (
            ref_pos,
            ref_quat,
            ref_finger,
            object_pos + 0.2,
            object_quat,
            torch.zeros(1, 0),
            torch.zeros(1, 2, 6, 3),
        ),
        contact_state=lambda: contact,
        object_com=torch.zeros(1, 1, 3),
    )
    observation.command = command
    return observation, command, ref_quat


def _env(command):
    action = SimpleNamespace(
        raw_action=torch.arange(56, dtype=torch.float32)[None],
        processed_target=torch.arange(58, dtype=torch.float32)[None],
    )
    return cast(Any, SimpleNamespace)(
        num_envs=1,
        action_manager=SimpleNamespace(get_term=lambda name: action),
        backend=SimpleNamespace(
            get_state_views=lambda names: {
                "qpos": torch.full((1, 44), 0.25),
                "qvel": torch.full((1, 44), 2.0),
            }
        ),
        episode_length_buf=torch.zeros(1, dtype=torch.int64),
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )


def test_policy_layout_and_first_step_contact_zero() -> None:
    observation, command, ref_quat = _observation()
    env = _env(command)
    result = observation.compute(env)
    assert result.shape == (1, 504)
    torch.testing.assert_close(
        result[0, :6], torch.tensor([0.1, 0.2, 0.3, -0.1, 0.0, 0.4])
    )
    torch.testing.assert_close(
        result[0, 6:14], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )
    torch.testing.assert_close(result[0, 26:70], torch.full((44,), -0.5))
    torch.testing.assert_close(result[0, 70:114], torch.full((44,), 2.0))
    expected_command_quat = multiply(
        conjugate(torch.tensor([1.0, 0.0, 0.0, 0.0])), ref_quat[0, 0]
    )
    torch.testing.assert_close(result[0, 124:128], expected_command_quat)
    torch.testing.assert_close(
        result[0, 186:242], env.action_manager.get_term("x").raw_action[0]
    )
    torch.testing.assert_close(
        result[0, 242:300], env.action_manager.get_term("x").processed_target[0]
    )
    torch.testing.assert_close(result[0, 300:], torch.zeros(204))

    env.episode_length_buf[:] = 1
    result = observation.compute(env)
    torch.testing.assert_close(result[0, 300:351], torch.full((51,), 0.3))
    torch.testing.assert_close(result[0, 351:402], result[0, 453:504])


def test_critic_context_is_ten_columns() -> None:
    _, command, _ = _observation()
    command.applied_voc_scale = torch.tensor([0.25])
    command.target_voc_scale = torch.tensor(0.1)
    command.steps_since_reset = torch.tensor([10])
    command.timestep = torch.tensor([5])
    command.num_frames = 11
    command.cfg = SimpleNamespace(voc_decay_steps=20)
    result = critic_context(_env(command))
    assert result.shape == (1, 10)
    torch.testing.assert_close(result[0, :4], torch.tensor([0.25, 0.1, 0.5, 0.5]))
    torch.testing.assert_close(result[0, 4:7], torch.full((3,), 0.4))
    torch.testing.assert_close(result[0, 7:], torch.full((3,), -0.3))


def test_critic_linear_velocity_uses_com_not_origin() -> None:
    _, command, _ = _observation()
    command.applied_voc_scale = torch.tensor([1.0])
    command.target_voc_scale = torch.tensor(1.0)
    command.steps_since_reset = torch.tensor([20])
    command.timestep = torch.tensor([0])
    command.num_frames = 10
    command.cfg = SimpleNamespace(voc_decay_steps=20)
    command.object_com = torch.tensor([[[0.1, 0.0, 0.0]]])
    z90 = from_euler_xyz(torch.tensor([0.0, 0.0, 1.5707963267948966]))
    command.object_state = lambda: (
        torch.zeros(1, 1, 3),
        z90.reshape(1, 1, 4),
        torch.tensor([[[1.0, 2.0, 3.0]]]),
        torch.tensor([[[0.0, 0.0, 2.0]]]),
    )
    result = critic_context(_env(command))
    torch.testing.assert_close(result[0, 4:7], torch.tensor([0.8, 2.0, 3.0]))
    torch.testing.assert_close(result[0, 7:], torch.tensor([0.0, 0.0, 2.0]))


def test_wrist_orientation_error_does_not_enable_termination() -> None:
    config = OmegaConf.load(
        Path(__file__).parents[1]
        / "src/chord_unilab/conf/flashsac/task/sharpa_chord/base.yaml"
    )
    nonzero_wrist_orientation_error = 0.6
    assert nonzero_wrist_orientation_error > 0.0
    assert "wrist_orientation" not in config.env.terminations
    assert "object_orientation" in config.env.terminations


def test_body_com_offsets_preserve_env_and_variant_rows() -> None:
    class Backend:
        def get_body_id(self, name):
            return {"a": 1, "b": 0}[name]

        def get_body_ipos(self, env_ids):
            np.testing.assert_array_equal(env_ids, [0, 1])
            return np.array(
                [
                    [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                    [[40.0, 50.0, 60.0], [10.0, 20.0, 30.0]],
                ],
                dtype=np.float32,
            )

    env = SimpleNamespace(device=torch.device("cpu"), backend=Backend())
    result = body_com_offsets(cast(Any, env), ("a", "b"), 2)
    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
            ]
        ),
    )
