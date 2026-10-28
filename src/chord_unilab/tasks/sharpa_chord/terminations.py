"""FlashChord tracking failures and reference-end truncation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from unilab.managers import ManagerTermBase

from .commands import reference_command
from .math import geodesic_angle

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


class TrackingTermination(ManagerTermBase):
    """Compare each wrist/object against its current reference pose.

    SimBackend capabilities: none directly; consumes ReferenceCommand's cached
    tracked-body views. reference_end likewise consumes only the reference clock.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.command = reference_command(
            env, cfg.params.get("command_name", "reference")
        )
        self.kind = cfg.params["kind"]

    def __call__(
        self, env: ManagerBasedRlEnv, kind: str, command_name: str = "reference"
    ) -> torch.Tensor:
        del command_name
        ref_pos, ref_quat, _, ref_obj_pos, ref_obj_quat, *_ = (
            self.command.current_reference()
        )
        if kind == "wrist_position":
            wrist_pos, _, _, _ = self.command.wrist_state()
            return (torch.linalg.vector_norm(wrist_pos - ref_pos, dim=-1) > 0.15).any(
                dim=-1
            )
        if kind == "wrist_orientation":
            _, wrist_quat, _, _ = self.command.wrist_state()
            return (geodesic_angle(wrist_quat, ref_quat) > 0.0).any(dim=-1)
        object_pos, object_quat, _, _ = self.command.object_state()
        if kind == "object_position":
            return (
                torch.linalg.vector_norm(object_pos - ref_obj_pos, dim=-1) > 0.1
            ).any(dim=-1)
        if kind == "object_orientation":
            return (geodesic_angle(object_quat, ref_obj_quat) > 0.5).any(dim=-1)
        raise ValueError(f"unknown tracking termination {kind!r}")


def reference_end(
    env: ManagerBasedRlEnv, command_name: str = "reference"
) -> torch.Tensor:
    command = reference_command(env, command_name)
    return command.timestep >= command.num_frames - 1
