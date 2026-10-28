"""Raw FlashChord tracking/contact terms; RewardManager applies 0.05*weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from unilab.managers import ManagerTermBase

from .actions import ResidualHandPoseAction
from .commands import ReferenceCommand, reference_command
from .math import rotate, rotate_inverse
from .scene import POLICY_SIDES
from .wrench import wrench_support_cuda

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

TERM_NAMES = (
    "object_keypoints",
    "hand_keypoints",
    "hand_joint_pos",
    "contact_wrench_support",
    "missed_contact",
    "unintended_contact",
    "termination",
    "action_rate_l2",
    "action_l2",
)
ACTIVE = 1.0e-3
WORKSPACE_CHUNK_SIZE = 16
WORKSPACE_CONE_EDGES = 8
AXES = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)


@dataclass(frozen=True)
class WrenchWorkspace:
    """Persistent exact-operation scratch for the live wrench evaluation."""

    product: torch.Tensor
    force_values: torch.Tensor
    torque_values: torch.Tensor
    chunk_output: torch.Tensor
    zero: torch.Tensor
    chunk_size: int
    cone_edges: int


def sample_wrench_basis(num_basis: int, radius: float, seed: int) -> np.ndarray:
    """Sample FlashChord's radius-scaled wrench directions on the 6-D sphere."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((num_basis, 6))
    basis[:, 3:] /= radius
    basis /= np.linalg.norm(basis, axis=-1, keepdims=True).clip(min=1.0e-8)
    return basis.astype(np.float32)


def wrench_support(
    position_o: torch.Tensor,
    direction_o: torch.Tensor,
    basis: torch.Tensor,
    radius: torch.Tensor,
    *,
    friction: float = 0.1,
    cone_edges: int = 8,
    chunk_size: int = 16,
    output: torch.Tensor | None = None,
    workspace: WrenchWorkspace | None = None,
) -> torch.Tensor:
    """Evaluate clamped friction-cone wrench support in [N,H,B,K]."""
    n, h, b, _, _ = position_o.shape
    directions = basis.shape[1]
    if workspace is not None and (
        workspace.chunk_size != chunk_size or workspace.cone_edges != cone_edges
    ):
        raise ValueError("wrench workspace chunk or cone configuration does not match")
    if position_o.device.type == "cuda":
        return wrench_support_cuda(
            position_o,
            direction_o,
            basis,
            radius,
            friction=friction,
            cone_edges=cone_edges,
            output=output,
        )
    if output is None:
        output = position_o.new_zeros((n, h, b, directions))
    elif tuple(output.shape) != (n, h, b, directions):
        raise ValueError(
            "wrench support output shape "
            f"{tuple(output.shape)} does not match {(n, h, b, directions)}"
        )
    normal_length = torch.linalg.vector_norm(direction_o, dim=-1, keepdim=True)
    normal = direction_o / normal_length.clamp_min(1.0e-12)
    enabled = normal_length > 1.0e-3
    sign = torch.where(normal[..., 2:3] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + normal[..., 2:3])
    bb = normal[..., 0:1] * normal[..., 1:2] * a
    t1 = torch.cat(
        (
            1.0 + sign * normal[..., 0:1].square() * a,
            sign * bb,
            -sign * normal[..., 0:1],
        ),
        dim=-1,
    )
    t2 = torch.cat(
        (bb, sign + normal[..., 1:2].square() * a, -normal[..., 1:2]),
        dim=-1,
    )
    t1 = t1 / torch.linalg.vector_norm(t1, dim=-1, keepdim=True).clamp_min(1.0e-12)
    t2 = t2 / torch.linalg.vector_norm(t2, dim=-1, keepdim=True).clamp_min(1.0e-12)
    theta = torch.arange(cone_edges, device=position_o.device) * (
        2.0 * torch.pi / cone_edges
    )
    rays = normal[..., None, :] + friction * (
        torch.cos(theta)[..., None] * t1[..., None, :]
        + torch.sin(theta)[..., None] * t2[..., None, :]
    )
    rays = torch.cat((rays, normal[..., None, :]), dim=-2)
    rays = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )
    torque = (
        torch.linalg.cross(position_o[..., None, :], rays)
        / radius[None, None, :, None, None, None]
    )
    for start in range(0, directions, chunk_size):
        stop = min(start + chunk_size, directions)
        chunk = basis[:, start:stop]
        if workspace is not None:
            width = stop - start
            product = workspace.product[..., :width, :]
            force_value = workspace.force_values[..., :width]
            torque_value = workspace.torque_values[..., :width]
            torch.mul(
                rays[..., None, :],
                chunk[None, None, :, None, None, :, :3],
                out=product,
            )
            torch.sum(product, dim=-1, out=force_value)
            torch.mul(
                torque[..., None, :],
                chunk[None, None, :, None, None, :, 3:],
                out=product,
            )
            torch.sum(product, dim=-1, out=torque_value)
            torch.add(force_value, torque_value, out=force_value)
            torch.where(
                enabled[..., None], force_value, workspace.zero, out=force_value
            )
            chunk_output = workspace.chunk_output[..., :width]
            torch.amax(force_value, dim=(3, 4), out=chunk_output)
            chunk_output.clamp_min_(0.0)
            output[..., start:stop].copy_(chunk_output)
            continue
        force_value = (
            rays[..., None, :] * chunk[None, None, :, None, None, :, :3]
        ).sum(dim=-1)
        torque_value = (
            torque[..., None, :] * chunk[None, None, :, None, None, :, 3:]
        ).sum(dim=-1)
        values = torch.where(enabled[..., None], force_value + torque_value, 0.0)
        output[:, :, :, start:stop] = values.amax(dim=(3, 4)).clamp_min(0.0)
    return output


def reference_support(command: ReferenceCommand, basis: torch.Tensor) -> torch.Tensor:
    reference = command.reference
    supports = []
    for side in POLICY_SIDES:
        pos_w = command._tensor(reference.contact_pos_w(side))
        normal_w = command._tensor(reference.contact_normal_w(side))
        parts = torch.as_tensor(reference.contact_part_ids(side), device=pos_w.device)
        body_supports = []
        for body in range(command.num_bodies):
            position_o = rotate_inverse(
                command.ref_object_quat[:, body, None, :],
                pos_w - command.ref_object_pos[:, body, None, :],
            )
            direction_o = rotate_inverse(
                command.ref_object_quat[:, body, None, :], normal_w
            )
            direction_o = direction_o * (parts == body + 1)[..., None]
            body_supports.append(
                wrench_support(
                    position_o[:, None, None, :, :],
                    direction_o[:, None, None, :, :],
                    basis[body : body + 1],
                    command.ref_object_radius[body : body + 1],
                )[:, 0, 0]
            )
        supports.append(torch.stack(body_supports, dim=1))
    return torch.stack(supports, dim=1)


class RewardRuntime:
    def __init__(self, command: ReferenceCommand):
        self.command = command
        radii = command.ref_object_radius.detach().cpu().tolist()
        self.reference_basis = torch.as_tensor(
            np.stack(
                [
                    sample_wrench_basis(512, float(radius), body)
                    for body, radius in enumerate(radii)
                ]
            ),
            device=command.ref_object_pos.device,
        )
        self.live_basis = torch.as_tensor(
            np.stack(
                [
                    sample_wrench_basis(512, 1.0, body)
                    for body in range(command.num_bodies)
                ]
            ),
            device=command.ref_object_pos.device,
        )
        self.axes = AXES.to(device=command.ref_object_pos.device)
        self.ref_support = reference_support(command, self.reference_basis)
        support_shape = (
            command.applied_voc_scale.shape[0],
            len(POLICY_SIDES),
            command.num_bodies,
            512,
        )
        self.live_support = torch.empty(
            support_shape, device=command.ref_object_pos.device
        )
        self.expected_support = torch.empty(
            support_shape, device=command.ref_object_pos.device
        )
        self.terms = {name: command.applied_voc_scale.clone() for name in TERM_NAMES}
        self.wrench_workspace: WrenchWorkspace | None = None
        self._step = -1

    def evaluate(self, env: ManagerBasedRlEnv) -> None:
        step = int(env.common_step_counter)
        if step == self._step:
            return
        self._step = step
        command = self.command
        ref_pos, ref_quat, ref_finger, ref_obj_pos, ref_obj_quat, _, ref_hand_kp = (
            command.current_reference()
        )
        del ref_pos, ref_quat
        object_pos, object_quat, _, _ = command.object_state()
        actual_axes = self.axes.expand(env.num_envs, command.num_bodies, 6, 3)
        object_actual = object_pos[:, :, None, :] + rotate(
            object_quat[:, :, None, :], actual_axes
        )
        object_target = ref_obj_pos[:, :, None, :] + rotate(
            ref_obj_quat[:, :, None, :], actual_axes
        )
        object_error = torch.linalg.vector_norm(object_actual - object_target, dim=-1)
        self.terms["object_keypoints"].copy_(
            torch.exp(-object_error / 0.1).mean(dim=(1, 2))
        )
        hand_error = torch.linalg.vector_norm(
            command.hand_keypoint_state() - ref_hand_kp, dim=-1
        )
        self.terms["hand_keypoints"].copy_(
            torch.exp(-hand_error / 0.1).mean(dim=(1, 2))
        )
        state = env.backend.get_state_views(("qpos",))
        observation = command._policy_observation
        finger = state["qpos"][:, observation.finger_qpos].reshape(env.num_envs, 2, 22)
        hand_joint_l2 = torch.linalg.vector_norm(finger - ref_finger, dim=-1)
        self.terms["hand_joint_pos"].copy_(torch.exp(-hand_joint_l2).mean(dim=1))
        contact = command.contact_state()
        if self.wrench_workspace is None and contact.position_o.device.type == "cpu":
            chunk_shape = (
                *contact.position_o.shape[:-1],
                WORKSPACE_CONE_EDGES + 1,
                WORKSPACE_CHUNK_SIZE,
            )
            self.wrench_workspace = WrenchWorkspace(
                product=contact.position_o.new_empty((*chunk_shape, 3)),
                force_values=contact.position_o.new_empty(chunk_shape),
                torque_values=contact.position_o.new_empty(chunk_shape),
                chunk_output=contact.position_o.new_empty(
                    (*contact.position_o.shape[:3], WORKSPACE_CHUNK_SIZE)
                ),
                zero=contact.position_o.new_zeros(()),
                chunk_size=WORKSPACE_CHUNK_SIZE,
                cone_edges=WORKSPACE_CONE_EDGES,
            )
        support = wrench_support(
            contact.position_o,
            contact.direction_o,
            self.live_basis,
            command.ref_object_radius,
            output=self.live_support,
            workspace=self.wrench_workspace,
        )
        expected = self.expected_support
        torch.index_select(
            self.ref_support.reshape(self.ref_support.shape[0], -1),
            0,
            command.frames,
            out=expected.reshape(expected.shape[0], -1),
        )
        cmd_active = expected > ACTIVE
        cur_active = support > ACTIVE
        better = ((1.0 - 0.1) * expected - support).clamp_min(0.0)
        too_large = (support - (1.0 + 0.1) * expected).clamp_min(0.0)
        shaped = torch.exp(-(better.square() + too_large.square()) / 0.1)
        cmd_num = cmd_active.sum(dim=-1)
        support_sum = (shaped * (cmd_active & cur_active)).sum(dim=-1)
        body_values = support_sum / cmd_num.clamp_min(1)
        body_active = cmd_active.any(dim=-1)
        hand_values = (body_values * body_active).sum(dim=-1) / body_active.sum(
            dim=-1
        ).clamp_min(1)
        hand_active = body_active.any(dim=-1)
        self.terms["contact_wrench_support"].copy_(
            (hand_values * hand_active).sum(dim=-1)
            / hand_active.sum(dim=-1).clamp_min(1)
        )
        missed = ((~cur_active) & cmd_active).sum(dim=-1).to(torch.float32)
        body_missed = missed / cmd_num.clamp_min(1)
        hand_missed = (body_missed * body_active).sum(dim=-1) / body_active.sum(
            dim=-1
        ).clamp_min(1)
        self.terms["missed_contact"].copy_(
            (hand_missed * hand_active).sum(dim=-1)
            / hand_active.sum(dim=-1).clamp_min(1)
        )
        unintended_binary = (
            ((~cmd_active.any(dim=-1)) & cur_active.any(dim=-1))
            .to(torch.float32)
            .mean(dim=-1)
        )
        support_square = support.square().mean(dim=-1)
        inactive_body = ~cmd_active.any(dim=-1)
        continuous = (support_square * inactive_body).sum(dim=-1) / inactive_body.sum(
            dim=-1
        ).clamp_min(1)
        self.terms["unintended_contact"].copy_(
            (unintended_binary + continuous).sum(dim=-1)
        )
        self.terms["termination"].copy_(
            env.termination_manager.terminated.to(torch.float32)
        )
        action = cast(
            "ResidualHandPoseAction",
            env.action_manager.get_term("residual_hand_pose"),
        )
        self.terms["action_rate_l2"].copy_(action.action_rate_l2)
        self.terms["action_l2"].copy_(action.action_l2)


class SharpaRewardTerm(ManagerTermBase):
    """Return one unweighted raw term from the shared per-step evaluation.

    SimBackend capabilities: qpos state views. ReferenceCommand supplies cached
    tracked-body/contact views, and RewardManager supplies weights and reward dt.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.command = reference_command(
            env, cfg.params.get("command_name", "reference")
        )
        self.term_name = cfg.params["term_name"]
        if self.term_name not in TERM_NAMES:
            raise ValueError(f"unknown FlashChord reward term {self.term_name}")
        if self.command._reward_runtime is None:
            self.command._reward_runtime = RewardRuntime(self.command)
        self.runtime = self.command._reward_runtime

    def __call__(
        self, env: ManagerBasedRlEnv, term_name: str, command_name: str = "reference"
    ) -> torch.Tensor:
        del term_name, command_name
        self.runtime.evaluate(env)
        return self.runtime.terms[self.term_name]
