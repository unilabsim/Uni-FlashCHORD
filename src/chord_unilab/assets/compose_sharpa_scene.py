"""Assemble dual Sharpa hands and reference-driven scene components.

``assemble_sharpa_scene`` is the formal entry point. Object entity names and
kinds come from ``ObjectAssetSpec``: every component has a floating root, zero
articulations produce a rigid entity, and one or more produce an articulation.
Support is a USDA-derived kinematic rigid entity; the optional floor is fixed.
Cross-entity hand/support excludes are generated from compiled geom ownership.

The assembly returns the UniSim ``SceneCfg``, object and hand reference bindings,
support and asset manifests, and source paths. Reference initial/per-frame state
is handled only by ``reference_state``. Callers explicitly choose the parquet;
there is no dataset-specific scene factory or default-parquet fallback.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco

from chord_unilab.assets import convert_floor, convert_object, convert_support
from chord_unilab.assets.actuators import append_position_velocity_actuators
from chord_unilab.assets.compose_sharpa_dual import (
    PARAMS,
    ROBOT_DIR,
    WRIST_Y_DEFAULT,
    WRIST_Z_DEFAULT,
    mount_with_hand,
    stamp_geom_filters,
)
from chord_unilab.assets.convert_object import cache_root
from chord_unilab.assets.convert_support import SupportAsset
from chord_unilab.assets.physics import DEFAULT_SCENE_PHYSICS, ScenePhysics
from chord_unilab.data.reference import ChordReference, ObjectAssetSpec
from chord_unilab.data.support_assets import resolve_support_asset

SCENE_SCHEMA = "chord-sharpa-scene-v8"

HAND_ENTITY_NAMES = ("left_hand", "right_hand")

SIM_DT = 0.01

_SLIDES = (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1"))


@dataclass(frozen=True)
class ObjectComponentBinding:
    spec: ObjectAssetSpec
    entity_name: str
    source_path: Path
    bodies: dict[str, str]
    articulations: dict[int, str]

    @property
    def kind(self) -> Literal["rigid", "articulation"]:
        return "articulation" if self.spec.articulations else "rigid"

    @property
    def root_body(self) -> str:
        return self.bodies[self.spec.root_reference_name]

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.spec.name}_{articulation.simulation_joint_name}"
            for articulation in self.spec.articulations
        )


@dataclass(frozen=True)
class HandReferenceBinding:
    side: str
    entity_name: str
    reference_joint_names: tuple[str, ...]
    joint_names: tuple[str, ...]


@dataclass(frozen=True)
class SharpaSceneAssembly:
    scene_cfg: Any
    bindings: tuple[ObjectComponentBinding, ...]
    hand_bindings: tuple[HandReferenceBinding, ...]
    support: SupportAsset | None
    manifest: dict[str, Any]
    physics: ScenePhysics
    entity_sources: dict[str, Path]

    @property
    def entity_names(self) -> tuple[str, ...]:
        return tuple(self.entity_sources)

    def initial_state(
        self, reference: ChordReference, *, env_ids: tuple[int, ...] = (0,)
    ):
        from chord_unilab.assets.reference_state import reference_state_patch

        return reference_state_patch(reference, self, 0, env_ids=env_ids)


def _fmt(values) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _option_block() -> dict[str, str]:
    # SolverMuJoCo settings from flash_chord SimConfig (runtime/sim.py); the
    # timestep is overridden by compose_scene(sim_dt) but authored for
    # standalone compilation of the entity sources.
    return {
        "timestep": f"{SIM_DT:.9g}",
        "solver": "Newton",
        "integrator": "implicitfast",
        "cone": "pyramidal",
        "impratio": "20",
        "iterations": "100",
        "ls_iterations": "50",
    }


def _default_block(root: ET.Element) -> None:
    contact = PARAMS["contact"]
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        {
            "solref": _fmt(contact["solref"]),
            "solimp": _fmt(contact["solimp"]),
            "friction": _fmt(contact["friction"]),
            "margin": f"{contact['margin']:.9g}",
            "gap": f"{contact['gap']:.9g}",
        },
    )
    # Same default-joint actuatorgravcomp as the dual composer; ball joints
    # override to false explicitly inside mount_with_hand's subtree.
    ET.SubElement(default, "joint", {"actuatorgravcomp": "true"})


def _hand_excludes(side: str) -> list[list[str]]:
    """The side's 202 same-side pairs from the GT hand exclude set."""
    return [
        [body_a, body_b]
        for body_a, body_b in PARAMS["excludes"]
        if body_a.startswith(f"{side}_") and body_b.startswith(f"{side}_")
    ]


def _collision_body_names(source: Path) -> tuple[str, ...]:
    model = mujoco.MjModel.from_xml_path(str(source))
    names = []
    for body_id in range(1, model.nbody):
        has_collision = any(
            model.geom_bodyid[geom_id] == body_id and model.geom_contype[geom_id] != 0
            for geom_id in range(model.ngeom)
        )
        if has_collision:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name is not None:
                names.append(name)
    return tuple(names)


def scene_contact_exclude_pairs(
    entity_names: tuple[str, ...], *, sources: dict[str, Path]
) -> tuple[tuple[str, str], ...]:
    """Generate hand/support excludes from actual collision geom ownership."""
    requested = set(entity_names)
    if "support" not in requested:
        return ()
    support_bodies = _collision_body_names(sources["support"])
    pairs = {
        (f"{hand}/{body}", f"support/{support_body}")
        for hand in HAND_ENTITY_NAMES
        if hand in requested
        for body in _collision_body_names(sources[hand])
        for support_body in support_bodies
    }
    return tuple(sorted(pairs))


def build_hand_entity_xml(side: str, *, cache_dir: Path | None = None) -> ET.Element:
    """One mounted hand as a self-contained single-entity MJCF document."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be left or right, got {side!r}")
    root = ET.Element("mujoco", {"model": f"{side}_hand"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", _option_block())
    _default_block(root)

    contact_elem = ET.SubElement(root, "contact")
    contact_elem.append(
        ET.Comment(
            f" {len(_hand_excludes(side))} same-side pairs from flash_chord's "
            "compiled mj_model exclude_signature; cross-entity pairs cannot be "
            "expressed inside one entity source (GAP-EXCLUDES, unisim #434) "
        )
    )
    for body_a, body_b in _hand_excludes(side):
        ET.SubElement(contact_elem, "exclude", {"body1": body_a, "body2": body_b})

    worldbody = ET.SubElement(root, "worldbody")
    assets: list[ET.Element] = []
    worldbody.append(mount_with_hand(side, assets))
    asset = ET.SubElement(root, "asset")
    for mesh in assets:
        # Mesh files are relative to the single-hand source's directory; the
        # entity XML lives in the external cache, so pin absolute paths.
        mesh_file = mesh.get("file")
        if mesh_file is not None:
            mesh.set("file", str((ROBOT_DIR / "mjcf" / mesh_file).resolve()))
        asset.append(mesh)

    # First pass: enumerate the side's finger hinges for the actuator block
    # and the keyframe width (mirrors compose_sharpa_dual's two-pass flow).
    tmp_root = ET.fromstring(ET.tostring(root))
    tmp_model = mujoco.MjModel.from_xml_string(
        ET.tostring(tmp_root, encoding="unicode")
    )
    # Stamp the newton collision-group bitmasks onto collision geoms (same
    # two-pass rule as the dual composer: stamping does not change the
    # kinematic layout, so the first-pass compile serves both).
    stamp_geom_filters(root, tmp_model, side=side)
    finger_joints = [
        name
        for j in range(tmp_model.njnt)
        if tmp_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        and (name := mujoco.mj_id2name(tmp_model, mujoco.mjtObj.mjOBJ_JOINT, j))
    ]

    actuator = ET.SubElement(root, "actuator")
    for name, _axis in _SLIDES:
        gains = PARAMS["actuators"][f"{side}_wrist_{name}"]
        append_position_velocity_actuators(
            actuator,
            name=f"{side}_wrist_{name}",
            joint=f"{side}_wrist_{name}",
            kp=float(gains["kp"]),
            kd=float(gains["kd"]),
        )
    for joint_name in finger_joints:
        gains = PARAMS["actuators"][joint_name]
        append_position_velocity_actuators(
            actuator,
            name=joint_name,
            joint=joint_name,
            kp=float(gains["kp"]),
            kd=float(gains["kd"]),
        )

    # Keyframe "home": flash_chord seeds wrist_y=±0.25, wrist_z=0.5, ball unit
    # quaternion, fingers zero (same defaults as the dual composer's keyframe).
    qpos = [0.0] * tmp_model.nq
    for name, value in (
        (f"{side}_wrist_y", WRIST_Y_DEFAULT[side]),
        (f"{side}_wrist_z", WRIST_Z_DEFAULT),
    ):
        joint_id = mujoco.mj_name2id(tmp_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[tmp_model.jnt_qposadr[joint_id]] = value
    ball_id = mujoco.mj_name2id(
        tmp_model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_rotvec"
    )
    qpos[tmp_model.jnt_qposadr[ball_id]] = 1.0
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe, "key", {"name": "home", "qpos": " ".join(f"{v:g}" for v in qpos)}
    )
    return root


def _validate_hand_entity(xml_path: Path, side: str) -> None:
    """Structural + load-bearing checks (hard-fail), mirroring the dual's."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    counts = {
        mujoco.mjtJoint.mjJNT_HINGE: 0,
        mujoco.mjtJoint.mjJNT_BALL: 0,
        mujoco.mjtJoint.mjJNT_SLIDE: 0,
    }
    for j in range(model.njnt):
        counts[model.jnt_type[j]] += 1
    print(
        f"{side}_hand: nbody={model.nbody} njnt={model.njnt} "
        f"(hinge={counts[mujoco.mjtJoint.mjJNT_HINGE]} "
        f"ball={counts[mujoco.mjtJoint.mjJNT_BALL]} "
        f"slide={counts[mujoco.mjtJoint.mjJNT_SLIDE]}) nu={model.nu} "
        f"nexclude={model.nexclude} nq={model.nq} nv={model.nv}"
    )
    expected = (
        counts[mujoco.mjtJoint.mjJNT_HINGE],
        counts[mujoco.mjtJoint.mjJNT_BALL],
        counts[mujoco.mjtJoint.mjJNT_SLIDE],
    )
    if expected != (22, 1, 3):
        raise ValueError(
            f"{side}_hand: expected 22 hinge + 1 ball + 3 slide, got {expected}"
        )
    if model.nbody != 27 or model.nu != 50:  # world + 3 dummies + palm + 22 fingers
        raise ValueError(
            f"{side}_hand: expected nbody=27 nu=50, got {model.nbody}/{model.nu}"
        )
    if model.nexclude != len(_hand_excludes(side)):
        raise ValueError(
            f"{side}_hand: expected {len(_hand_excludes(side))} excludes, got {model.nexclude}"
        )


def hand_entity_path(side: str, *, cache_dir: Path | None = None) -> Path:
    """Cache location of the side's mounted hand entity XML; builds if stale.

    The file content depends only on the robot asset inputs, so the hand
    directory is content-keyed by the single-hand source plus the GT hand
    params slice (any change rebuilds).
    """
    out_dir = (cache_dir or cache_root()) / "scene" / "hands"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{side}_hand.xml"
    root = build_hand_entity_xml(side, cache_dir=cache_dir)
    payload = ET.tostring(root, encoding="unicode")
    if out_path.is_file():
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        if (
            ET.tostring(ET.parse(out_path, parser).getroot(), encoding="unicode")
            == payload
        ):
            return out_path
    tmp_path = out_path.with_suffix(".tmp.xml")
    ET.ElementTree(root).write(tmp_path, xml_declaration=True, encoding="unicode")
    _validate_hand_entity(tmp_path, side)
    tmp_path.replace(out_path)
    print(f"wrote {out_path}")
    return out_path


def _validate_reference_assets(reference: ChordReference) -> None:
    assets = reference.object_assets()
    names = tuple(asset.name for asset in assets)
    reserved = set(HAND_ENTITY_NAMES) | {"support", "floor"}
    if not assets or len(set(names)) != len(names) or reserved.intersection(names):
        raise ValueError(
            f"object component names must be unique and unreserved, got {names}"
        )
    convert_object.validate_object_asset_spec(reference, assets[0])
    for spec in assets[1:]:
        convert_object.validate_object_asset_spec(reference, spec)
    bodies = tuple(body.reference_name for spec in assets for body in spec.bodies)
    articulations = tuple(
        articulation.reference_index
        for spec in assets
        for articulation in spec.articulations
    )
    if len(set(bodies)) != len(bodies) or set(bodies) != set(
        reference.object_body_names()
    ):
        raise ValueError(
            "object asset bodies must cover every reference body exactly once"
        )
    expected = set(range(reference.object_articulation().shape[1]))
    if len(set(articulations)) != len(articulations) or set(articulations) != expected:
        raise ValueError(
            "object asset articulations must cover every reference column exactly once"
        )


def _object_binding(spec: ObjectAssetSpec, source: Path) -> ObjectComponentBinding:
    model = mujoco.MjModel.from_xml_path(str(source))
    root_id = next(
        body_id
        for body_id in range(1, model.nbody)
        if model.body_parentid[body_id] == 0
    )
    root_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_id)
    if root_name is None:
        raise ValueError(f"object component {spec.name!r} has no named root body")
    root_link = root_name.removeprefix(f"{spec.name}_")
    bodies = {
        body.reference_name: (
            f"{spec.name}/{spec.name}_{body.simulation_name or root_link}"
        )
        for body in spec.bodies
    }
    articulations = {
        articulation.reference_index: (
            f"{spec.name}/{spec.name}_{articulation.simulation_joint_name}"
        )
        for articulation in spec.articulations
    }
    return ObjectComponentBinding(spec, spec.name, source, bodies, articulations)


def _hand_binding(
    reference: ChordReference, side: str, source: Path
) -> HandReferenceBinding:
    reference_names = reference.finger_joint_names(side)
    model = mujoco.MjModel.from_xml_path(str(source))
    source_names = tuple(
        name
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    )
    if len(reference_names) != len(source_names) or set(reference_names) != set(
        source_names
    ):
        raise ValueError(f"{side} reference finger joints do not match the hand asset")
    return HandReferenceBinding(side, f"{side}_hand", reference_names, reference_names)


def assemble_sharpa_scene(
    reference: ChordReference,
    *,
    include_support: bool = True,
    include_floor: bool = True,
    physics: ScenePhysics = DEFAULT_SCENE_PHYSICS,
    cache_dir: Path | None = None,
) -> SharpaSceneAssembly:
    """Build the unique data-driven scene configuration and reset bindings."""
    from unisim.dr.types import ModelSourceDescriptor
    from unisim.entities import EntityInitialState, SceneEntitySpec
    from unisim.scene import SceneCfg

    _validate_reference_assets(reference)
    sources = {
        f"{side}_hand": hand_entity_path(side, cache_dir=cache_dir)
        for side in ("left", "right")
    }
    bindings = []
    for spec in reference.object_assets():
        source = convert_object.convert_object(
            reference, spec, cache_dir=cache_dir, physics=physics
        )
        sources[spec.name] = source
        bindings.append(_object_binding(spec, source))
    support = resolve_support_asset(reference) if include_support else None
    support_asset = None
    if include_support:
        if support is None:
            raise FileNotFoundError(
                f"support is enabled but none resolved for {reference.metadata.source_path}"
            )
        support_asset = convert_support.load_support_asset(support)
        sources["support"] = convert_support.convert_support(
            support, cache_dir=cache_dir, physics=physics
        )
    if include_floor:
        sources["floor"] = convert_floor.convert_floor(
            cache_dir=cache_dir, physics=physics
        )

    specs = []
    for name, source in sources.items():
        if name in HAND_ENTITY_NAMES:
            kind, root_mode = "articulation", "floating"
        elif name == "support":
            kind, root_mode = "rigid", "kinematic"
        elif name == "floor":
            kind, root_mode = "rigid", "fixed"
        else:
            binding = next(
                binding for binding in bindings if binding.entity_name == name
            )
            kind, root_mode = binding.kind, "floating"
        specs.append(
            SceneEntitySpec(
                name,
                ModelSourceDescriptor(str(source)),
                kind=kind,
                root_mode=root_mode,
                initial_state=EntityInitialState(),
            )
        )
    cfg = SceneCfg(
        entity_assets=tuple(specs),
        contact_excludes=scene_contact_exclude_pairs(tuple(sources), sources=sources),
    )
    hands = tuple(
        _hand_binding(reference, side, sources[f"{side}_hand"])
        for side in ("left", "right")
    )
    manifest = {
        "schema": SCENE_SCHEMA,
        "reference_source": reference.metadata.source_path,
        "entities": list(sources),
        "objects": [
            {
                "name": binding.entity_name,
                "kind": binding.kind,
                "source": str(binding.source_path),
                "sha256": hashlib.sha256(binding.source_path.read_bytes()).hexdigest(),
            }
            for binding in bindings
        ],
        "support": None
        if support_asset is None
        else {
            "path": str(support_asset.path),
            "sha256": support_asset.sha256,
            "shapes": [asdict(shape) for shape in support_asset.shapes],
        },
        "physics": repr(physics),
    }
    return SharpaSceneAssembly(
        cfg, tuple(bindings), hands, support_asset, manifest, physics, sources
    )
