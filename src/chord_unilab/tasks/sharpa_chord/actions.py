"""Rational residual actions and per-substep wrist/VOC control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from unilab.managers import ActionTerm, ActionTermCfg

from .commands import reference_command
from .math import conjugate, from_euler_xyz, multiply, normalize, to_rotvec
from .scene import POLICY_SIDES

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ResidualHandPoseActionCfg(ActionTermCfg):
    command_name: str = "reference"
    wrist_pos_scale: float = 0.05
    wrist_ori_scale: float = 0.15
    finger_joint_pos_scale: float = 0.15
    wrist_pos_clip: float = 0.2
    wrist_ori_clip: float = 1.0
    finger_joint_pos_clip: float = 1.0
    ema: float = 0.3
    wrist_kp: float = 25.0
    wrist_tau_max: float = 45.0

    def build(self, env: ManagerBasedRlEnv) -> ResidualHandPoseAction:
        return ResidualHandPoseAction(self, env)


class ResidualHandPoseAction(ActionTerm):
    """Consume 56 normalized residuals and publish 58 absolute pose targets.

    SimBackend capabilities: scene layout, actuator names/joints/gains, body mass
    and IDs, body-inertia reset defaults, qpos/qvel state views, and generalized
    force tensors. The primary Entity also provides the public control buffer.
    """

    requires_substep_state_feedback: ClassVar[bool] = True
    uses_tensor_state_packet: ClassVar[bool] = False

    def __init__(self, cfg: ResidualHandPoseActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.command = reference_command(env, cfg.command_name)
        self.active_reference: tuple[torch.Tensor, ...] | None = None
        self._device = torch.device(env.device)
        self.gravity_w = torch.tensor(
            [0.0, 0.0, 9.81], dtype=torch.float32, device=self._device
        )
        self._raw_action = torch.zeros(
            (env.num_envs, 56), dtype=torch.float32, device=self._device
        )
        self.filtered_action = torch.zeros_like(self._raw_action)
        self.processed_target = torch.zeros(
            (env.num_envs, 58), dtype=torch.float32, device=self._device
        )
        self.action_l2 = torch.zeros(env.num_envs, device=self._device)
        self.action_rate_l2 = torch.zeros(env.num_envs, device=self._device)
        scales: list[list[float]] = []
        clips: list[list[float]] = []
        input_scales: list[list[float]] = []
        for _ in POLICY_SIDES:
            scales.extend(
                (
                    [cfg.wrist_pos_scale] * 3,
                    [cfg.wrist_ori_scale] * 3,
                    [cfg.finger_joint_pos_scale] * 22,
                )
            )
            clips.extend(
                (
                    [cfg.wrist_pos_clip] * 3,
                    [cfg.wrist_ori_clip] * 3,
                    [cfg.finger_joint_pos_clip] * 22,
                )
            )
            input_scales.extend(
                (
                    [cfg.wrist_pos_clip / ((1.0 - cfg.ema) * cfg.wrist_pos_scale)] * 3,
                    [cfg.wrist_ori_clip / ((1.0 - cfg.ema) * cfg.wrist_ori_scale)] * 3,
                    [
                        cfg.finger_joint_pos_clip
                        / ((1.0 - cfg.ema) * cfg.finger_joint_pos_scale)
                    ]
                    * 22,
                )
            )
        self.scale = self._vector(scales)
        self.clip = self._vector(clips)
        self.input_scale = self._vector(input_scales)
        layout = env.backend.get_scene_layout()
        actuator_names = env.backend.get_actuator_names()
        actuator_joints = env.backend.get_actuator_joint_names()
        position_rows = {
            joint: row
            for row, (actuator, joint) in enumerate(
                zip(actuator_names, actuator_joints, strict=True)
            )
            if actuator.rsplit("/", 1)[-1] == f"{joint.rsplit('/', 1)[-1]}_pos"
        }
        velocity_rows = {
            joint: row
            for row, (actuator, joint) in enumerate(
                zip(actuator_names, actuator_joints, strict=True)
            )
            if actuator.rsplit("/", 1)[-1] == f"{joint.rsplit('/', 1)[-1]}_vel"
        }
        wrist_pos_joints: list[str] = []
        finger_joints: list[str] = []
        ball_qpos: list[int] = []
        ball_qvel: list[int] = []
        for side in POLICY_SIDES:
            binding = next(
                binding
                for binding in self.command.assembly.hand_bindings
                if binding.side == side
            )
            entity = layout.get_entity(binding.entity_name)
            slide = {
                joint.name: joint
                for joint in entity.root_joints
                if joint.kind == "slide"
            }
            wrist_pos_joints.extend(
                f"{binding.entity_name}/{slide[f'{binding.side}_wrist_{axis}'].name}"
                for axis in ("x", "y", "z")
            )
            balls = [joint for joint in entity.root_joints if joint.kind == "ball"]
            if len(balls) != 1:
                raise ValueError(f"{binding.entity_name} requires one wrist ball joint")
            ball_qpos.extend(balls[0].qpos_indices)
            ball_qvel.extend(balls[0].qvel_indices)
            finger_joints.extend(
                f"{binding.entity_name}/{name}" for name in binding.joint_names
            )
        self.wrist_pos_ctrl = self._indices(position_rows, wrist_pos_joints)
        self.finger_ctrl = self._indices(position_rows, finger_joints)
        self.ball_qpos = torch.tensor(
            ball_qpos, dtype=torch.int64, device=self._device
        ).reshape(2, 4)
        self.ball_qvel = torch.tensor(
            ball_qvel, dtype=torch.int64, device=self._device
        ).reshape(2, 3)
        object_joints = [
            f"{binding.entity_name}/{name}"
            for binding in self.command.assembly.bindings
            for name in binding.joint_names
        ]
        self.object_ctrl = self._indices(position_rows, object_joints)
        self.object_vel_ctrl = self._indices(velocity_rows, object_joints)
        object_layout_joints = [
            joint
            for binding in self.command.assembly.bindings
            for joint in layout.get_entity(binding.entity_name).joints
        ]
        self.object_qpos = self._flat_indices(
            [joint.qpos_indices for joint in object_layout_joints]
        )
        self.object_qvel = self._flat_indices(
            [joint.qvel_indices for joint in object_layout_joints]
        )
        kp, kd = env.backend.get_actuator_gains()
        self.object_kp = self._gain(kp, [position_rows[name] for name in object_joints])
        self.object_kd = self._gain(kd, [velocity_rows[name] for name in object_joints])
        body_mass = np.asarray(env.backend.get_body_mass(), dtype=np.float32)
        if body_mass.ndim == 2:
            body_mass = body_mass[0]
        inertia = np.asarray(
            env.backend.get_reset_term_default("body_inertia"), dtype=np.float32
        )
        if inertia.ndim == 3:
            inertia = inertia[0]
        masses = []
        inertias = []
        root_ids = []
        for object_binding in self.command.assembly.bindings:
            ids = [
                env.backend.get_body_id(body) for body in object_binding.bodies.values()
            ]
            root = env.backend.get_body_id(object_binding.root_body)
            masses.append(float(body_mass[ids].sum()))
            inertias.append(float(inertia[root].sum() / 3.0))
            root_ids.append(root)
        self.object_mass = self._vector(masses)
        self.object_inertia = self._vector(inertias)
        self.object_body_ids = torch.tensor(
            root_ids, dtype=torch.int64, device=self._device
        )
        reference_names = self.command.reference.object_body_names()
        self.object_reference_ids = torch.tensor(
            [
                reference_names.index(binding.spec.root_reference_name)
                for binding in self.command.assembly.bindings
            ],
            dtype=torch.int64,
            device=self._device,
        )
        self.object_wrench = torch.zeros(
            (env.num_envs, self.object_reference_ids.numel(), 6),
            dtype=torch.float32,
            device=self._device,
        )
        self.joint_force = torch.zeros(
            (env.num_envs, layout.nv), dtype=torch.float32, device=self._device
        )
        self.body_force = torch.zeros(
            (
                env.num_envs,
                len(env.backend.get_body_mass())
                if np.asarray(env.backend.get_body_mass()).ndim == 1
                else np.asarray(env.backend.get_body_mass()).shape[1],
                6,
            ),
            dtype=torch.float32,
            device=self._device,
        )
        self._control = self._entity.data.control_buffer
        if not isinstance(self._control, torch.Tensor):
            raise TypeError(
                "ResidualHandPoseAction requires the public Torch control plane"
            )
        self.reset()
        self._force_evaluator = self._evaluate_forces
        if self._device.type == "cuda":
            self._force_evaluator = torch.compile(self._evaluate_forces, dynamic=False)

    def _vector(self, values) -> torch.Tensor:
        flat = [
            value
            for group in values
            for value in (group if isinstance(group, list) else [group])
        ]
        return torch.tensor(flat, dtype=torch.float32, device=self._device)

    def _indices(self, mapping: dict[str, int], names: list[str]) -> torch.Tensor:
        try:
            values = [mapping[name] for name in names]
        except KeyError as exc:
            raise ValueError(f"missing actuator for {exc.args[0]}") from exc
        return torch.tensor(values, dtype=torch.int64, device=self._device)

    def _flat_indices(self, groups) -> torch.Tensor:
        values = [index for group in groups for index in group]
        return torch.tensor(values, dtype=torch.int64, device=self._device)

    def _gain(self, gains, rows: list[int]) -> torch.Tensor:
        values = np.asarray(gains, dtype=np.float32)
        if values.ndim == 2:
            values = values[0]
        return torch.as_tensor(values[rows].copy(), device=self._device)

    @property
    def action_dim(self) -> int:
        return 56

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    def process_actions(self, actions: torch.Tensor) -> None:
        invalidate = getattr(self.command, "invalidate_current_reference", None)
        if callable(invalidate):
            invalidate()
        self._action_finite_valid.copy_(torch.isfinite(actions).all())
        normalized = actions.clamp(-1.0, 1.0)
        magnitude = normalized.abs()
        mapped = normalized / ((1.0 - magnitude) + magnitude / self.input_scale)
        self.action_l2.copy_(mapped.square().sum(dim=-1))
        self.action_rate_l2.copy_((mapped - self._raw_action).square().sum(dim=-1))
        self._raw_action.copy_(mapped)
        self.filtered_action.copy_(
            (
                self.cfg.ema * self.filtered_action
                + (1.0 - self.cfg.ema) * mapped * self.scale
            ).clamp(-self.clip, self.clip)
        )
        reference = self.command.current_reference()
        self.active_reference = reference
        ref_wrist_pos, ref_wrist_quat, ref_finger, *_ = reference
        filtered = self.filtered_action.reshape(self.num_envs, 2, 28)
        targets = self.processed_target.view(self.num_envs, 2, 29)
        target_pos = targets[:, :, :3]
        torch.add(ref_wrist_pos, filtered[:, :, :3], out=target_pos)
        residual_quat = from_euler_xyz(filtered[:, :, 3:6])
        target_quat = normalize(multiply(ref_wrist_quat, residual_quat))
        targets[:, :, 3:7].copy_(target_quat)
        target_finger = targets[:, :, 7:]
        torch.add(ref_finger, filtered[:, :, 6:], out=target_finger)
        self._control[:, self.wrist_pos_ctrl] = target_pos.reshape(self.num_envs, 6)
        self._control[:, self.finger_ctrl] = target_finger.reshape(self.num_envs, 44)
        ref_articulation = reference[5]
        if self.object_ctrl.numel():
            self._control[:, self.object_ctrl] = ref_articulation
            self._control[:, self.object_vel_ctrl] = 0.0

    def _evaluate_forces(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        linear_origin: torch.Tensor,
        angular: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        targets = self.processed_target.reshape(self.num_envs, 2, 29)
        current = normalize(qpos[:, self.ball_qpos])
        target = targets[:, :, 3:7]
        target = torch.where(
            (target * current).sum(dim=-1, keepdim=True) < 0.0, -target, target
        )
        error = to_rotvec(multiply(conjugate(current), target))
        torque = (self.cfg.wrist_kp * error).clamp(
            -self.cfg.wrist_tau_max, self.cfg.wrist_tau_max
        )
        assert self.active_reference is not None
        if self.object_qvel.numel():
            ref_articulation = self.active_reference[5]
            drive = (
                self.object_kp * (ref_articulation - qpos[:, self.object_qpos])
                - self.object_kd * qvel[:, self.object_qvel]
            )
            drive = (self.command.applied_voc_scale[:, None] - 1.0) * drive
        else:
            drive = qpos.new_zeros((qpos.shape[0], 0))
        _, _, _, object_target_pos, object_target_quat, *_ = self.active_reference
        object_target_pos = object_target_pos[:, self.object_reference_ids]
        object_target_quat = object_target_quat[:, self.object_reference_ids]
        object_pos = object_pos[:, self.object_reference_ids]
        object_quat = object_quat[:, self.object_reference_ids]
        linear_origin = linear_origin[:, self.object_reference_ids]
        angular = angular[:, self.object_reference_ids]
        force = self.object_mass * (
            400.0 * (object_target_pos - object_pos) - 40.0 * linear_origin
        )
        force = force + self.gravity_w * self.object_mass
        orientation_error = to_rotvec(
            multiply(object_target_quat, conjugate(object_quat))
        )
        torque_object = self.object_inertia * (
            400.0 * orientation_error - 40.0 * angular
        )
        scale = self.command.applied_voc_scale[:, None, None]
        force = (scale * force).clamp(-200.0, 200.0)
        torque_object = (scale * torque_object).clamp(-50.0, 50.0)
        return torque, drive, torch.cat((force, torque_object), dim=-1)

    def apply_actions(self) -> None:
        views = self._env.backend.get_state_views(("qpos", "qvel"))
        qpos, qvel = views["qpos"], views["qvel"]
        if self.active_reference is None:
            self.active_reference = self.command.current_reference()
        object_pos, object_quat, linear_origin, angular = self.command.object_state()
        evaluator = getattr(self, "_force_evaluator", self._evaluate_forces)
        torque, drive, object_wrench = evaluator(
            qpos, qvel, object_pos, object_quat, linear_origin, angular
        )
        self.joint_force.zero_()
        self.joint_force[:, self.ball_qvel] = torque
        if self.object_qvel.numel():
            self.joint_force[:, self.object_qvel] = drive
        self.object_wrench.copy_(object_wrench)
        self.body_force.zero_()
        self.body_force.index_copy_(1, self.object_body_ids, self.object_wrench)
        self._env.backend.set_generalized_force_tensor(
            joint_f=self.joint_force, body_f=self.body_force
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_action[ids] = 0.0
        self.filtered_action[ids] = 0.0
        self.processed_target[ids] = 0.0
        target = self.processed_target.reshape(self.num_envs, 2, 29)
        target[ids, :, 3] = 1.0
        self.action_l2[ids] = 0.0
        self.action_rate_l2[ids] = 0.0
        self.active_reference = None
        self.joint_force[ids] = 0.0
        self.body_force[ids] = 0.0
        self._env.backend.set_generalized_force_tensor(
            joint_f=self.joint_force, body_f=self.body_force
        )
