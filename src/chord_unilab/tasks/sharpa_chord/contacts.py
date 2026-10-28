"""Fixed-slot hand/object contacts through public BodyPairContactData views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from unisim import BodyPairContactData

from .math import rotate_inverse
from .scene import CONTACT_LINKS, POLICY_SIDES, contact_sensor_name

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

    from .scene import SharpaSceneCfg


@dataclass
class ContactSlots:
    """World forces on objects and their wrist/object-frame representations."""

    position_w: torch.Tensor
    force_w: torch.Tensor
    position_b: torch.Tensor
    direction_b: torch.Tensor
    position_o: torch.Tensor
    direction_o: torch.Tensor
    active: torch.Tensor


class ContactTracker:
    """Bind ordered object-first sensors once, then read live device tensors.

    SimBackend capabilities: named sensor views returning BodyPairContactData.
    """

    def __init__(self, env: ManagerBasedRlEnv, scene: SharpaSceneCfg):
        self.num_bodies = len(scene.reference.object_body_names())
        self.slots_per_hand = len(CONTACT_LINKS) * self.num_bodies
        self._views: list[list[BodyPairContactData]] = []
        for side in POLICY_SIDES:
            side_views = []
            for body_index in range(self.num_bodies):
                for link_index in range(len(CONTACT_LINKS)):
                    view = env.backend.get_sensor_view(
                        contact_sensor_name(side, link_index, body_index)
                    )
                    if not isinstance(view, BodyPairContactData):
                        raise TypeError(
                            f"contact sensor {side}/{link_index}/{body_index} is not BodyPairContactData"
                        )
                    side_views.append(view)
            self._views.append(side_views)

    def read(
        self,
        wrist_pos_w: torch.Tensor,
        wrist_quat_w: torch.Tensor,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
    ) -> ContactSlots:
        num_envs = wrist_pos_w.shape[0]
        shape = (num_envs, len(POLICY_SIDES), self.num_bodies, len(CONTACT_LINKS))
        position_w = torch.stack(
            [view.position_w for views in self._views for view in views], dim=1
        ).reshape(*shape, 3)
        force_w = torch.stack(
            [view.force_w for views in self._views for view in views], dim=1
        ).reshape(*shape, 3)
        active = torch.stack(
            [view.active for views in self._views for view in views], dim=1
        ).reshape(shape)
        position_w = position_w * active[..., None]
        force_w = force_w * active[..., None]
        direction_w = force_w / torch.linalg.vector_norm(
            force_w, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        wrist_quat = wrist_quat_w[:, :, None, None, :]
        object_quat = object_quat_w[:, None, :, None, :]
        position_b = rotate_inverse(
            wrist_quat, position_w - wrist_pos_w[:, :, None, None, :]
        )
        direction_b = rotate_inverse(wrist_quat, direction_w)
        position_o = rotate_inverse(
            object_quat, position_w - object_pos_w[:, None, :, None, :]
        )
        position_b = position_b * active[..., None]
        position_o = position_o * active[..., None]
        direction_o = rotate_inverse(object_quat, direction_w)
        return ContactSlots(
            position_w,
            force_w,
            position_b,
            direction_b,
            position_o,
            direction_o,
            active,
        )
