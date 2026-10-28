"""FlashChord's ordered 504-D policy observation and critic context."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from unilab.managers import ManagerTermBase

from .actions import ResidualHandPoseAction
from .commands import ReferenceCommand, reference_command
from .math import conjugate, multiply, rotate, rotate_inverse, unique
from .scene import CONTACT_SIDES, POLICY_SIDES

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


class PolicyObservation(ManagerTermBase):
    """Consume public body/joint/contact tensors; preserve all 13 enabled groups.

    SimBackend capabilities: named qpos/qvel indices, joint ranges, and qpos/qvel
    state views. Tracked bodies and contacts come from ReferenceCommand's cached
    public views; critic_context consumes the same command state directly.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.command = reference_command(
            env, cfg.params.get("command_name", "reference")
        )
        self._delegate = getattr(self.command, "_policy_observation", None)
        if self._delegate is None:
            self.command._policy_observation = self
            bindings = {
                binding.side: binding for binding in self.command.assembly.hand_bindings
            }
            names = tuple(
                f"{bindings[side].entity_name}/{name}"
                for side in POLICY_SIDES
                for name in bindings[side].joint_names
            )
            self.finger_qpos = torch.as_tensor(
                env.backend.get_joint_state_qpos_indices(names),
                dtype=torch.int64,
                device=env.device,
            )
            self.finger_qvel = torch.as_tensor(
                env.backend.get_joint_state_qvel_indices(names),
                dtype=torch.int64,
                device=env.device,
            )
            ranges = env.backend.get_joint_range(names=names)
            if ranges is None:
                raise ValueError("finger observations require joint ranges")
            self.finger_lower = torch.as_tensor(
                np.asarray(ranges[:, 0]), dtype=torch.float32, device=env.device
            )
            self.finger_upper = torch.as_tensor(
                np.asarray(ranges[:, 1]), dtype=torch.float32, device=env.device
            )

    @staticmethod
    def _indices(env: ManagerBasedRlEnv, values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, device=env.device)

    def __call__(
        self, env: ManagerBasedRlEnv, command_name: str = "reference"
    ) -> torch.Tensor:
        del command_name
        if self._delegate is not None:
            return self._delegate(env)
        return self.compute(env)

    def compute(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        command = self.command
        action = cast(
            "ResidualHandPoseAction",
            env.action_manager.get_term("residual_hand_pose"),
        )
        wrist_pos, wrist_quat, wrist_lin_com, wrist_ang = command.wrist_state()
        object_pos, object_quat, _, _ = command.object_state()
        ref_pos, ref_quat, ref_finger, ref_obj_pos, ref_obj_quat, *_ = (
            command.current_reference()
        )
        state = env.backend.get_state_views(("qpos", "qvel"))
        finger_pos = state["qpos"][:, self.finger_qpos].reshape(env.num_envs, 2, 22)
        finger_vel = state["qvel"][:, self.finger_qvel].reshape(env.num_envs, 2, 22)
        lower = self.finger_lower.reshape(2, 22)
        upper = self.finger_upper.reshape(2, 22)
        finger_normalized = (
            2.0 * (finger_pos - lower) / (upper - lower).clamp_min(1.0e-8) - 1.0
        )
        unique_wrist = unique(wrist_quat)
        linear_origin = wrist_lin_com + torch.linalg.cross(
            wrist_ang, wrist_pos - wrist_pos
        )
        wrist_velocity = torch.cat(
            (
                rotate_inverse(unique_wrist, linear_origin),
                rotate_inverse(unique_wrist, wrist_ang),
            ),
            dim=-1,
        )
        unique_ref_wrist = unique(ref_quat)
        command_wrist = torch.cat(
            (
                rotate_inverse(unique_wrist, ref_pos - wrist_pos),
                multiply(conjugate(unique_wrist), unique_ref_wrist),
            ),
            dim=-1,
        )
        command_object_pos = rotate_inverse(object_quat, ref_obj_pos - object_pos)
        command_object_quat = multiply(conjugate(object_quat), unique(ref_obj_quat))
        contacts = command.contact_state()
        contact_groups: list[torch.Tensor] = []
        for side in CONTACT_SIDES:
            index = POLICY_SIDES.index(side)
            position = contacts.position_b[:, index].reshape(env.num_envs, -1)
            direction = contacts.direction_b[:, index].reshape(env.num_envs, -1)
            if env.episode_length_buf is not None:
                first = (env.episode_length_buf == 0)[:, None]
                position = torch.where(first, 0.0, position)
                direction = torch.where(first, 0.0, direction)
            contact_groups.extend((position, direction))
        groups = (
            wrist_pos.reshape(env.num_envs, -1),
            unique_wrist.reshape(env.num_envs, -1),
            wrist_velocity.reshape(env.num_envs, -1),
            finger_normalized.reshape(env.num_envs, -1),
            finger_vel.reshape(env.num_envs, -1),
            object_pos.reshape(env.num_envs, -1),
            object_quat.reshape(env.num_envs, -1),
            command_wrist.reshape(env.num_envs, -1),
            (ref_finger - finger_pos).reshape(env.num_envs, -1),
            command_object_pos.reshape(env.num_envs, -1),
            command_object_quat.reshape(env.num_envs, -1),
            action.raw_action,
            action.processed_target,
            *contact_groups,
        )
        observation = torch.cat(groups, dim=-1)
        if command.num_bodies == 1 and observation.shape[1] != 504:
            raise RuntimeError(
                f"single-body FlashChord observation must be 504-D, got {observation.shape[1]}"
            )
        return observation


def critic_context(
    env: ManagerBasedRlEnv, command_name: str = "reference"
) -> torch.Tensor:
    command: ReferenceCommand = reference_command(env, command_name)
    progress = (
        command.steps_since_reset.to(torch.float32) / command.cfg.voc_decay_steps
        if command.cfg.voc_decay_steps > 0
        else torch.ones_like(command.applied_voc_scale)
    ).clamp(0.0, 1.0)
    phase = command.timestep.to(torch.float32) / max(command.num_frames - 1, 1)
    _, object_quat, linear_origin, angular = command.object_state()
    com_offset_w = rotate(object_quat, command.object_com)
    linear_com = linear_origin + torch.linalg.cross(angular, com_offset_w)
    return torch.cat(
        (
            command.applied_voc_scale[:, None],
            command.target_voc_scale.expand(env.num_envs, 1),
            progress[:, None],
            phase.clamp(0.0, 1.0)[:, None],
            linear_com.reshape(env.num_envs, -1),
            angular.reshape(env.num_envs, -1),
        ),
        dim=-1,
    )
