from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chord_unilab.assets import compose_sharpa_scene, convert_object, convert_support
from chord_unilab.assets.reference_state import reference_state_patch
from chord_unilab.data.reference import (
    ChordReference,
    HandReference,
    ObjectArticulationSpec,
    ObjectAssetSpec,
    ObjectBodySpec,
    ObjectJointDriveSpec,
    ObjectJointPhysicsSpec,
    ObjectReference,
    ReferenceMetadata,
)
from chord_unilab.data.support_assets import resolve_support_asset


def _model(path: Path):
    mujoco = pytest.importorskip("mujoco")
    return mujoco.MjModel.from_xml_path(str(path))


_FINGERS = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)


def _hand(side: str, frames: int = 2) -> HandReference:
    return HandReference(
        wrist_pos_w=np.full((frames, 3), 0.25 if side == "left" else -0.25),
        wrist_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1)),
        finger_joint_pos=np.full((frames, len(_FINGERS)), 0.125),
        finger_joint_names=tuple(f"{side}_{name}" for name in _FINGERS),
        contact_pos_w=np.zeros((frames, 1, 3)),
        contact_normal_w=np.zeros((frames, 1, 3)),
        contact_part_ids=np.zeros((frames, 1), dtype=np.int32),
        frame_pos_w=np.zeros((frames, 1, 3)),
        frame_names=("palm",),
    )


def _reference(
    specs: tuple[ObjectAssetSpec, ...],
    body_names: tuple[str, ...],
    articulation_count: int,
    *,
    source_path: str = "/synthetic/reference.parquet",
) -> ChordReference:
    frames = 2
    positions = np.zeros((frames, len(body_names), 3))
    for body_index in range(len(body_names)):
        positions[:, body_index, 0] = 0.5 + body_index
        positions[1, body_index, 1] = 0.75 + body_index
    quaternions = np.zeros((frames, len(body_names), 4))
    quaternions[..., 0] = 1.0
    return ChordReference(
        metadata=ReferenceMetadata(source_path=source_path, source_fps=20.0),
        playback_fps=20.0,
        hands={"left": _hand("left"), "right": _hand("right")},
        object_ref=ObjectReference(
            name="synthetic",
            body_names=body_names,
            mesh_paths=tuple("" for _ in body_names),
            urdf_paths=tuple(spec.urdf_path for spec in specs),
            mesh_radius=np.ones(len(body_names)),
            body_pos_w=positions,
            body_quat_w=quaternions,
            articulation=np.full((frames, articulation_count), 0.4),
            assets=specs,
        ),
    )


def _inertial() -> str:
    return (
        '<inertial><origin xyz="0 0 0"/><mass value="1"/>'
        '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>'
        "</inertial>"
    )


def _rigid_urdf(path: Path, link: str = "part") -> ObjectAssetSpec:
    path.write_text(
        f'<robot name="{path.stem}"><link name="{link}">{_inertial()}'
        '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
        f"</link></robot>"
    )
    return ObjectAssetSpec(
        name=path.stem,
        urdf_path=str(path),
        bodies=(ObjectBodySpec(reference_name=path.stem, simulation_name=None),),
        root_reference_name=path.stem,
    )


def test_rigid_object_has_free_root_and_no_actuators(tmp_path: Path) -> None:
    spec = _rigid_urdf(tmp_path / "rigid.urdf")
    reference = _reference((spec,), ("rigid",), 0)
    fragment = convert_object.convert_object(reference, spec, cache_dir=tmp_path)
    model = _model(fragment)
    assert (model.nbody, model.njnt, model.nu, model.nq, model.nv, model.nkey) == (
        2,
        1,
        0,
        7,
        6,
        1,
    )
    np.testing.assert_array_equal(model.key_qpos[0], [0, 0, 0, 1, 0, 0, 0])
    assert not (tmp_path / "rigid.patched.urdf").exists()
    manifest = json.loads(fragment.with_name("manifest.json").read_text())
    assert "parquet" not in manifest and "source_path" not in manifest
    assert "gt_params" not in manifest


def test_multiple_articulations_drive_structure_and_home_width(tmp_path: Path) -> None:
    urdf = tmp_path / "multi.urdf"
    joints = []
    links = []
    bodies = []
    articulations = []
    for index in range(2):
        child = f"child{index}"
        links.append(
            f'<link name="{child}">{_inertial()}'
            '<collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision></link>'
        )
        joints.append(
            f'<joint name="joint{index}" type="revolute"><parent link="root"/>'
            f'<child link="{child}"/><axis xyz="0 0 1"/>'
            '<limit lower="0" upper="1" effort="10" velocity="1"/></joint>'
        )
        bodies.append(ObjectBodySpec(reference_name=child, simulation_name=child))
        articulations.append(
            ObjectArticulationSpec(
                simulation_joint_name=f"joint{index}",
                reference_index=index,
                physics=ObjectJointPhysicsSpec(armature=0.01, friction=0.1),
                drive=ObjectJointDriveSpec(kp=50.0, kd=2.0, effort_limit=50.0),
            )
        )
    urdf.write_text(
        f'<robot name="multi"><link name="root">{_inertial()}'
        '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>'
        + "".join(links + joints)
        + "</robot>"
    )
    spec = ObjectAssetSpec(
        name="multi",
        urdf_path=str(urdf),
        bodies=(ObjectBodySpec(reference_name="root", simulation_name="root"), *bodies),
        root_reference_name="root",
        articulations=tuple(articulations),
    )
    reference = _reference((spec,), ("root", "child0", "child1"), 2)
    fragment = convert_object.convert_object(reference, spec, cache_dir=tmp_path)
    model = _model(fragment)
    assert (model.nbody, model.njnt, model.nu, model.nq, model.nv) == (4, 3, 4, 9, 8)
    assert model.key_qpos.shape == (1, 9)
    np.testing.assert_array_equal(model.key_qpos[0], [0, 0, 0, 1, 0, 0, 0, 0, 0])


def test_multi_rigid_assembly_and_shared_reset_patch(tmp_path: Path) -> None:
    first = _rigid_urdf(tmp_path / "first.urdf")
    second = _rigid_urdf(tmp_path / "second.urdf")
    reference = _reference((first, second), ("first", "second"), 0)
    assembly = compose_sharpa_scene.assemble_sharpa_scene(
        reference,
        include_support=False,
        include_floor=False,
        cache_dir=tmp_path,
    )
    assert assembly.entity_names == ("left_hand", "right_hand", "first", "second")
    assert [binding.kind for binding in assembly.bindings] == ["rigid", "rigid"]
    assert [spec.kind for spec in assembly.scene_cfg.entity_assets] == [
        "articulation",
        "articulation",
        "rigid",
        "rigid",
    ]
    request = reference_state_patch(reference, assembly, 1, env_ids=(0, 2))
    patches = {patch.entity: patch for patch in request.patches}
    assert set(patches) == set(assembly.entity_names)
    for name in ("first", "second"):
        patch = patches[name]
        assert patch.root_velocity is not None and not patch.root_velocity.any()
        assert patch.joint_positions is not None and patch.joint_positions.shape == (
            2,
            0,
        )
        body_index = reference.object_body_names().index(name)
        assert patch.root_pose is not None
        np.testing.assert_allclose(
            patch.root_pose[1],
            np.concatenate(
                (
                    reference.object_body_pos_w()[1, body_index],
                    reference.object_body_quat_w()[1, body_index],
                )
            ),
        )
    assert patches["left_hand"].root_velocity is not None
    assert not patches["left_hand"].root_velocity.any()


def test_support_resolver_prefers_sequence_adjacent_robot_specific(
    tmp_path: Path,
) -> None:
    sequence = (
        tmp_path
        / "human_motion_data/demo/processed/sequence_id=sequence/robot_name=other"
    )
    sequence.mkdir(parents=True)
    stage = tmp_path / "human_motion_data/demo/reconstructed_stage"
    stage.mkdir()
    generic = stage / "sequence_support.usda"
    other = stage / "sequence_other_support.usda"
    specific = stage / "sequence_sharpa_wave_support.usda"
    for path in (generic, other, specific):
        path.write_text("#usda 1.0\n")
    spec = _rigid_urdf(tmp_path / "part.urdf")
    reference = _reference(
        (spec,),
        ("part",),
        0,
        source_path=str(sequence / "reference.parquet"),
    )
    reference = replace(
        reference,
        metadata=replace(
            reference.metadata,
            source_dataset="demo",
            sequence_id="sequence",
            robot_name="sharpa_wave",
        ),
    )
    assert resolve_support_asset(reference) == specific
    specific.unlink()
    assert resolve_support_asset(reference) == generic


def test_support_usda_shapes_and_kinematic_scene_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usda = tmp_path / "support.usda"
    usda.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)
def Xform "World" {
    def Cylinder "Disk" {
        double radius = 0.2
        double height = 0.04
        double3 xformOp:translate = (0, 0, 0.3)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cylinder "Second" {
        double radius = 0.1
        double height = 0.02
        double3 xformOp:translate = (0.5, 0, 0.1)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "Pad" {
        double size = 2
        double3 xformOp:translate = (0, 0.4, 0.2)
        double3 xformOp:scale = (0.3, -0.2, 0.1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
}
"""
    )
    fragment = convert_support.convert_support(usda, cache_dir=tmp_path)
    model = _model(fragment)
    assert (model.nbody, model.njnt, model.ngeom, model.nu) == (2, 1, 6, 0)
    collision_ids = (0, 2, 4)
    np.testing.assert_allclose(model.geom_size[collision_ids[0]][:2], [0.2, 0.02])
    np.testing.assert_allclose(model.geom_size[collision_ids[1]][:2], [0.1, 0.01])
    np.testing.assert_allclose(model.geom_size[collision_ids[2]], [0.3, 0.2, 0.1])
    np.testing.assert_allclose(model.geom_pos[collision_ids[2]], [0, 0.4, 0.2])
    for collision_id in collision_ids:
        visual_id = collision_id + 1
        assert model.geom_contype[visual_id] == 0
        assert model.geom_conaffinity[visual_id] == 0
        assert model.geom_type[visual_id] == model.geom_type[collision_id]
        np.testing.assert_allclose(
            model.geom_size[visual_id], model.geom_size[collision_id]
        )
        np.testing.assert_allclose(
            model.geom_pos[visual_id], model.geom_pos[collision_id]
        )

    spec = _rigid_urdf(tmp_path / "part.urdf")
    reference = _reference((spec,), ("part",), 0)
    monkeypatch.setattr(
        compose_sharpa_scene, "resolve_support_asset", lambda _reference: usda
    )
    assembly = compose_sharpa_scene.assemble_sharpa_scene(
        reference,
        include_support=True,
        include_floor=False,
        cache_dir=tmp_path,
    )
    assert assembly.support is not None
    support_spec = next(
        spec for spec in assembly.scene_cfg.entity_assets if spec.name == "support"
    )
    assert (support_spec.kind, support_spec.root_mode) == ("rigid", "kinematic")
