"""Cold-path FlashChord scene, sensor, and reference assembly."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from unilab.base.entity import EntityCfg
from unilab.base.scene import SceneCfg

from chord_unilab.assets.compose_sharpa_scene import (
    SharpaSceneAssembly,
    assemble_sharpa_scene,
)
from chord_unilab.assets.convert_object import cache_root
from chord_unilab.data.loader import load_chord_reference
from chord_unilab.data.reference import ChordReference

POLICY_SIDES = ("right", "left")
CONTACT_SIDES = ("left", "right")
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
CONTACT_LINKS = (
    "hand_C_MC",
    "thumb_MC",
    "thumb_PP",
    "thumb_DP",
    "index_PP",
    "index_MP",
    "index_DP",
    "middle_PP",
    "middle_MP",
    "middle_DP",
    "ring_PP",
    "ring_MP",
    "ring_DP",
    "pinky_MC",
    "pinky_PP",
    "pinky_MP",
    "pinky_DP",
)


def contact_sensor_name(side: str, link_index: int, body_index: int) -> str:
    return f"chord_contact_{side}_{link_index:02d}_{body_index:02d}"


def hand_body_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_hand_C_MC",
        *(f"{side}_{digit}_DP" for digit in DIGITS),
    )


@lru_cache(maxsize=8)
def load_scene_reference(
    parquet_path: str,
    motion_speed: float,
    start_frame: int,
    end_frame: int,
) -> ChordReference:
    reference = load_chord_reference(
        parquet_path, control_fps=20.0, motion_speed=motion_speed
    )
    return reference.frame_window(start_frame, end_frame)


@lru_cache(maxsize=8)
def assemble_cached_scene(
    reference: ChordReference,
    include_support: bool,
    include_floor: bool,
) -> SharpaSceneAssembly:
    return assemble_sharpa_scene(
        reference, include_support=include_support, include_floor=include_floor
    )


def _sensor_fragment(assembly: SharpaSceneAssembly, reference: ChordReference) -> Path:
    declarations: list[str] = []
    object_bodies = [
        body for binding in assembly.bindings for body in binding.spec.bodies
    ]
    for side in POLICY_SIDES:
        for link_index, link in enumerate(CONTACT_LINKS):
            hand_body = f"{side}_hand/{side}_{link}"
            for body_index, body in enumerate(object_bodies):
                owner = next(
                    binding
                    for binding in assembly.bindings
                    if body.reference_name in binding.bodies
                )
                object_body = owner.bodies[body.reference_name]
                name = contact_sensor_name(side, link_index, body_index)
                declarations.append(
                    f'<contact name="{name}" body1="{object_body}" body2="{hand_body}" '
                    'data="found force pos" num="1"/>'
                )
    payload = "<mujoco><sensor>" + "".join(declarations) + "</sensor></mujoco>"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    path = cache_root() / "scene" / "fragments" / f"sharpa_chord_{digest}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_text(encoding="utf-8") != payload:
        path.write_text(payload, encoding="utf-8")
    return path


@dataclass
class SharpaSceneCfg(SceneCfg):
    """Assemble one data-driven scene; all asset work happens at config time."""

    reference_path: str = "???"
    motion_speed: float = 0.5
    motion_start_frame: int = 0
    motion_end_frame: int = -1
    include_support: bool = True
    include_floor: bool = True
    deferred_contact_publication: bool = False
    reference: ChordReference = field(init=False)
    assembly: SharpaSceneAssembly = field(init=False)

    def __post_init__(self) -> None:
        if self.reference_path == "???":
            raise ValueError("env.scene.reference_path must name a CHORD parquet")
        self.reference = load_scene_reference(
            self.reference_path,
            self.motion_speed,
            self.motion_start_frame,
            self.motion_end_frame,
        )
        if abs(self.reference.fps - 20.0) > 1.0e-6:
            raise ValueError("FlashChord reference must be resampled to 20 Hz")
        self.assembly = assemble_cached_scene(
            self.reference, self.include_support, self.include_floor
        )
        self.entity_assets = self.assembly.scene_cfg.entity_assets
        self.primary_entity = "left_hand"
        self.contact_excludes = self.assembly.scene_cfg.contact_excludes
        self.fragment_files = [str(_sensor_fragment(self.assembly, self.reference))]
        entities: dict[str, Any] = {}
        for binding in self.assembly.hand_bindings:
            side = binding.side
            entities[binding.entity_name] = EntityCfg(
                root_body_name=f"{side}_hand/{side}_wrist_px_body",
                physical_entity=binding.entity_name,
                joint_names=tuple(
                    f"{binding.entity_name}/{name}" for name in binding.joint_names
                ),
                body_names=tuple(
                    f"{binding.entity_name}/{body}" for body in hand_body_names(side)
                ),
                actuator_names=(),
            )
        for object_binding in self.assembly.bindings:
            entities[object_binding.entity_name] = EntityCfg(
                root_body_name=object_binding.root_body,
                physical_entity=object_binding.entity_name,
                joint_names=tuple(
                    f"{object_binding.entity_name}/{name}"
                    for name in object_binding.joint_names
                ),
                body_names=tuple(object_binding.bodies.values()),
                actuator_names=(),
            )
        self.entities = entities
        super().__post_init__()


def scene_from_env(env: Any) -> SharpaSceneCfg:
    scene = env.cfg.scene
    if not isinstance(scene, SharpaSceneCfg):
        raise TypeError("SharpaChord terms require env.scene to be SharpaSceneCfg")
    return scene
