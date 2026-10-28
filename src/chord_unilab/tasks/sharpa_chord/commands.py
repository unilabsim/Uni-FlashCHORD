"""Reference command, random reset, VOC settling, and curriculum state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from unilab.managers import CommandTerm, CommandTermCfg

from .contacts import ContactSlots, ContactTracker
from .scene import POLICY_SIDES, hand_body_names, scene_from_env

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


@dataclass(frozen=True)
class HandResetBinding:
    side: int
    position: torch.Tensor
    orientation: torch.Tensor
    fingers: torch.Tensor
    velocity: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


@dataclass(frozen=True)
class ObjectResetBinding:
    body: int
    root: torch.Tensor
    articulations: torch.Tensor
    reference_articulations: torch.Tensor
    velocity: torch.Tensor


VOC_THRESHOLDS = (
    19660800,
    34406400,
    49152000,
    63897600,
    78643200,
    93388800,
    108134400,
    122880000,
    137625600,
    200007680,
    250003456,
)
VOC_SCALES = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.0, 0.0, 0.0, 0.0)


@dataclass(kw_only=True)
class ReferenceCommandCfg(CommandTermCfg):
    voc_decay_steps: int = 20
    reset_finger_openness: float = 0.7
    reset_to_first_frame_probability: float = 0.1
    first_frame_voc_threshold: float = 0.1
    curriculum_enabled: bool = True
    initial_voc_scale: float = 1.0
    always_reset_to_first_frame: bool = False

    def build(self, env: ManagerBasedRlEnv) -> ReferenceCommand:
        return ReferenceCommand(self, env)


def body_com_offsets(
    env: ManagerBasedRlEnv, body_names: tuple[str, ...], num_envs: int
) -> torch.Tensor:
    body_ids = [env.backend.get_body_id(name) for name in body_names]
    body_ipos = env.backend.get_body_ipos(env_ids=np.arange(num_envs, dtype=np.int64))
    if (
        body_ipos.ndim != 3
        or body_ipos.shape[0] != num_envs
        or body_ipos.shape[2] != 3
        or any(body_id >= body_ipos.shape[1] for body_id in body_ids)
    ):
        raise ValueError(
            f"object body_ipos shape {body_ipos.shape}; expected "
            f"({num_envs}, nbody, 3) containing all selected body IDs"
        )
    return torch.as_tensor(
        body_ipos[:, body_ids, :], dtype=torch.float32, device=env.device
    )


class ReferenceCommand(CommandTerm):
    """Own the exact reference clock and the 20-step frozen-settling timeline.

    SimBackend capabilities: tracked-body views, body IDs and local COM offsets,
    named joint ranges, qpos/qvel views, and selected tensor state resets. Contact
    access is delegated to ContactTracker; curriculum uses public command state.
    """

    def __init__(self, cfg: ReferenceCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        scene = scene_from_env(env)
        self.scene = scene
        self.reference = scene.reference
        self.assembly = scene.assembly
        self.num_frames = self.reference.num_frames
        self.num_hands = len(POLICY_SIDES)
        self.num_bodies = len(self.reference.object_body_names())
        self.timestep = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self._device
        )
        self.episode_start_frame = torch.zeros_like(self.timestep)
        self.steps_since_reset = torch.zeros_like(self.timestep)
        self.applied_voc_scale = torch.ones(
            self.num_envs, dtype=torch.float32, device=self._device
        )
        self.target_voc_scale = torch.tensor(cfg.initial_voc_scale, device=self._device)
        self.first_frame_probability = float(cfg.reset_to_first_frame_probability)
        self.curriculum_stage = 0
        self._last_published_stage = -1
        self.contact = ContactTracker(env, scene)
        self._reward_runtime: Any = None
        self._policy_observation: Any = None
        self._current_reference: tuple[torch.Tensor, ...] | None = None
        self._contact_state: ContactSlots | None = None
        self._contact_state_step: int | None = None
        event_modes = getattr(env.event_manager, "available_modes", ())
        self._contact_cache_allowed = not ({"step", "interval"} & set(event_modes))
        self._wrist_views = env.backend.get_tracked_body_views(
            tuple(f"{side}_hand/{side}_hand_C_MC" for side in POLICY_SIDES),
            device=env.device,
        )
        keypoint_names = tuple(
            f"{side}_hand/{body}"
            for side in POLICY_SIDES
            for body in hand_body_names(side)
        )
        self._keypoint_views = env.backend.get_tracked_body_views(
            keypoint_names, device=env.device
        )
        self._object_views = env.backend.get_tracked_body_views(
            tuple(
                body
                for binding in self.assembly.bindings
                for body in binding.bodies.values()
            ),
            device=env.device,
        )
        self.object_com = body_com_offsets(
            env, tuple(self._object_views.body_names), self.num_envs
        )
        finger_limits: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for binding in self.assembly.hand_bindings:
            names = tuple(
                f"{binding.entity_name}/{name}" for name in binding.joint_names
            )
            ranges = env.backend.get_joint_range(names=names)
            if ranges is None or ranges.shape != (len(names), 2):
                raise ValueError(f"missing finger limits for {binding.entity_name}")
            finger_limits[binding.side] = (
                torch.as_tensor(ranges[:, 0], dtype=torch.float32, device=self._device),
                torch.as_tensor(ranges[:, 1], dtype=torch.float32, device=self._device),
            )
        self.ref_wrist_pos = self._tensor(
            np.stack(
                [self.reference.wrist_pos_w(side) for side in POLICY_SIDES], axis=1
            )
        )
        self.ref_wrist_quat = self._tensor(
            np.stack(
                [self.reference.wrist_quat_w(side) for side in POLICY_SIDES], axis=1
            )
        )
        self.ref_finger = self._tensor(
            np.stack(
                [self.reference.finger_joint_pos(side) for side in POLICY_SIDES], axis=1
            )
        )
        self.ref_object_pos = self._tensor(self.reference.object_body_pos_w())
        self.ref_object_quat = self._tensor(self.reference.object_body_quat_w())
        self.ref_object_articulation = self._tensor(
            self.reference.object_articulation()
        )
        keypoints = []
        for side in POLICY_SIDES:
            names = self.reference.frame_names(side)
            expected = hand_body_names(side)
            indices = [names.index(name) for name in expected]
            keypoints.append(self.reference.frame_pos_w(side)[:, indices])
        self.ref_hand_keypoints = self._tensor(np.stack(keypoints, axis=1))
        self.ref_object_radius = self._tensor(self.reference.object_mesh_radius())
        self._reset_hands, self._reset_objects = self._tensor_reset_bindings(
            env, finger_limits
        )

    def _tensor_reset_bindings(
        self,
        env: ManagerBasedRlEnv,
        finger_limits: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[tuple[HandResetBinding, ...], tuple[ObjectResetBinding, ...]]:
        layout = env.backend.get_scene_layout()

        def indices(groups) -> torch.Tensor:
            return torch.tensor(
                [index for group in groups for index in group],
                dtype=torch.int64,
                device=self._device,
            )

        binding: Any
        hands = []
        for side_index, side in enumerate(POLICY_SIDES):
            binding = next(
                binding
                for binding in self.assembly.hand_bindings
                if binding.side == side
            )
            entity = layout.get_entity(binding.entity_name)
            slides = {
                joint.name: joint
                for joint in entity.root_joints
                if joint.kind == "slide"
            }
            balls = [joint for joint in entity.root_joints if joint.kind == "ball"]
            if len(balls) != 1:
                raise ValueError(f"{binding.entity_name} requires one wrist ball joint")
            lower, upper = finger_limits[side]
            hands.append(
                HandResetBinding(
                    side=side_index,
                    position=indices(
                        [slides[f"{side}_wrist_{axis}"].qpos_indices for axis in "xyz"]
                    ),
                    orientation=indices([balls[0].qpos_indices]),
                    fingers=indices([joint.qpos_indices for joint in entity.joints]),
                    velocity=indices(
                        [
                            *(joint.qvel_indices for joint in entity.root_joints),
                            *(joint.qvel_indices for joint in entity.joints),
                        ]
                    ),
                    lower=lower,
                    upper=upper,
                )
            )
        reference_names = self.reference.object_body_names()
        objects = []
        for binding in self.assembly.bindings:
            entity = layout.get_entity(binding.entity_name)
            if len(entity.root_qpos_indices) != 7:
                raise ValueError(
                    f"{binding.entity_name} requires a floating object root"
                )
            objects.append(
                ObjectResetBinding(
                    body=reference_names.index(binding.spec.root_reference_name),
                    root=indices([entity.root_qpos_indices]),
                    articulations=indices(
                        [joint.qpos_indices for joint in entity.joints]
                    ),
                    reference_articulations=indices(
                        [
                            (articulation.reference_index,)
                            for articulation in binding.spec.articulations
                        ]
                    ),
                    velocity=indices(
                        [
                            entity.root_qvel_indices,
                            *(joint.qvel_indices for joint in entity.joints),
                        ]
                    ),
                )
            )
        return tuple(hands), tuple(objects)

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(values), dtype=torch.float32, device=self._device
        )

    @property
    def command(self) -> torch.Tensor:
        return self.timestep[:, None].to(torch.float32)

    @property
    def frames(self) -> torch.Tensor:
        return self.timestep.clamp(0, self.num_frames - 1)

    def current_reference(self) -> tuple[torch.Tensor, ...]:
        if self._current_reference is not None:
            return self._current_reference
        frames = self.frames
        self._current_reference = (
            self.ref_wrist_pos[frames],
            self.ref_wrist_quat[frames],
            self.ref_finger[frames],
            self.ref_object_pos[frames],
            self.ref_object_quat[frames],
            self.ref_object_articulation[frames],
            self.ref_hand_keypoints[frames],
        )
        return self._current_reference

    def invalidate_current_reference(self) -> None:
        self._current_reference = None

    def wrist_state(self) -> tuple[torch.Tensor, ...]:
        views = self._wrist_views
        return views.pos_w, views.quat_w, views.lin_vel_w, views.ang_vel_w

    def object_state(self) -> tuple[torch.Tensor, ...]:
        views = self._object_views
        return views.pos_w, views.quat_w, views.lin_vel_w, views.ang_vel_w

    def hand_keypoint_state(self) -> torch.Tensor:
        return self._keypoint_views.pos_w.reshape(self.num_envs, self.num_hands, 6, 3)

    def contact_state(self) -> ContactSlots:
        step = int(self._env.common_step_counter)
        if (
            self._contact_cache_allowed
            and self._contact_state is not None
            and self._contact_state_step == step
        ):
            return self._contact_state
        wrist_pos, wrist_quat, _, _ = self.wrist_state()
        object_pos, object_quat, _, _ = self.object_state()
        result = self.contact.read(wrist_pos, wrist_quat, object_pos, object_quat)
        if self._contact_cache_allowed:
            self._contact_state = result
            self._contact_state_step = step
        else:
            self._contact_state = None
            self._contact_state_step = None
        return result

    def reset(
        self,
        env_ids: torch.Tensor | slice | None = None,
        *,
        publish_metrics: bool = True,
    ) -> dict[str, float]:
        del publish_metrics
        self.invalidate_current_reference()
        self._contact_state = None
        self._contact_state_step = None
        reset_mask = self._reset_mask_buffer.resolve(
            env_ids, owner="ReferenceCommand.reset"
        )
        random_values = self._env.rng.uniform(
            0.0, 1.0, (self.num_envs, 3), dtype=torch.float32
        )
        frames = torch.floor(random_values[:, 0] * max(self.num_frames - 1, 1)).to(
            torch.int64
        )
        frames = frames.clamp(max=max(self.num_frames - 1, 0))
        if self.cfg.always_reset_to_first_frame:
            frames.zero_()
        else:
            first = (self.target_voc_scale < self.cfg.first_frame_voc_threshold) & (
                random_values[:, 1] < self.first_frame_probability
            )
            frames = torch.where(first, 0, frames)
        finger_scale = random_values[:, 2] * self.cfg.reset_finger_openness
        ids = reset_mask.nonzero(as_tuple=False).flatten()
        frame_ids = frames[ids]
        views = self._env.backend.get_state_views(("qpos", "qvel"))
        reset_qpos = views["qpos"][ids].clone()
        reset_qvel = views["qvel"][ids].clone()
        binding: Any
        for binding in self._reset_hands:
            reset_qpos[:, binding.position] = self.ref_wrist_pos[
                frame_ids, binding.side
            ]
            reset_qpos[:, binding.orientation] = self.ref_wrist_quat[
                frame_ids, binding.side
            ]
            fingers = self.ref_finger[frame_ids, binding.side]
            fingers = fingers * finger_scale[ids, None]
            fingers = torch.minimum(
                torch.maximum(fingers, binding.lower), binding.upper
            )
            reset_qpos[:, binding.fingers] = fingers
            reset_qvel[:, binding.velocity] = 0.0
        for binding in self._reset_objects:
            reset_qpos[:, binding.root] = torch.cat(
                (
                    self.ref_object_pos[frame_ids, binding.body],
                    self.ref_object_quat[frame_ids, binding.body],
                ),
                dim=-1,
            )
            reset_qpos[:, binding.articulations] = self.ref_object_articulation[
                frame_ids
            ][:, binding.reference_articulations]
            reset_qvel[:, binding.velocity] = 0.0
        self._env.backend.set_state_tensor(ids, reset_qpos, reset_qvel)
        self.timestep[ids] = frames[ids]
        self.episode_start_frame[ids] = frames[ids]
        self.steps_since_reset[ids] = 0
        self.applied_voc_scale[ids] = (
            self.target_voc_scale if self.cfg.voc_decay_steps == 0 else 1.0
        )
        self.time_left[ids] = self.cfg.resampling_time_range[0]
        self.command_counter[ids] = 0
        return {}

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        del reset_mask

    def _update_command(self) -> None:
        self.invalidate_current_reference()
        self.update_curriculum()
        self._publish_curriculum()
        active = ~self._env.reset_buf
        self.steps_since_reset.add_(active.to(torch.int64))
        settled = self.steps_since_reset >= self.cfg.voc_decay_steps
        self.applied_voc_scale.copy_(
            torch.where(
                settled, self.target_voc_scale, torch.ones_like(self.target_voc_scale)
            )
        )
        self.timestep.add_((active & settled).to(torch.int64))
        self.command_counter.add_(active.to(torch.int64))

    def _publish_curriculum(self) -> None:
        if self._last_published_stage == self.curriculum_stage:
            return
        metrics = voc_curriculum(self._env)
        cast(Any, self._env).state.info["log"].update(
            {f"Curriculum/voc/{key}": value for key, value in metrics.items()}
        )
        self._last_published_stage = self.curriculum_stage

    def _update_metrics(self) -> None:
        pass

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        del reset_mask

    def post_compute_reset(self) -> None:
        self.post_compute()

    def update_curriculum(self) -> None:
        if not self.cfg.curriculum_enabled:
            return
        environment_steps = int(self._env.common_step_counter) * self.num_envs
        stage = sum(environment_steps >= threshold for threshold in VOC_THRESHOLDS)
        if stage == self.curriculum_stage:
            return
        self.curriculum_stage = stage
        self.target_voc_scale.fill_(VOC_SCALES[stage])
        self.first_frame_probability = (
            0.5
            if stage >= len(VOC_THRESHOLDS) - 1
            else self.cfg.reset_to_first_frame_probability
        )
        settled = self.steps_since_reset >= self.cfg.voc_decay_steps
        self.applied_voc_scale.copy_(
            torch.where(settled, self.target_voc_scale, self.applied_voc_scale)
        )


def reference_command(
    env: ManagerBasedRlEnv, command_name: str = "reference"
) -> ReferenceCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ReferenceCommand):
        raise TypeError(f"{command_name} must be ReferenceCommand")
    return command


def voc_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None = None,
    command_name: str = "reference",
) -> dict[str, float]:
    del env_ids
    command = reference_command(env, command_name)
    return {
        "target_voc_scale": float(command.target_voc_scale.item()),
        "stage": float(command.curriculum_stage),
        "first_frame_probability": float(command.first_frame_probability),
    }
