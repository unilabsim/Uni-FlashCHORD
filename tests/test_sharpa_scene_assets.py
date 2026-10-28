"""Sharpa scene structure and Newton integration tests."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from unisim.mjcf_compiler import compose_scene
from unisim.scene import SceneCfg

from chord_unilab.assets import compose_sharpa_scene, convert_object
from chord_unilab.data import load_chord_reference


def _assembly(cache_dir: Path) -> compose_sharpa_scene.SharpaSceneAssembly:
    reference = load_chord_reference(
        convert_object.DEFAULT_PARQUET,
        control_fps=convert_object.GT_CONTROL_FPS,
        motion_speed=convert_object.GT_MOTION_SPEED,
    )
    return compose_sharpa_scene.assemble_sharpa_scene(reference, cache_dir=cache_dir)


def _scene_cfg(
    assembly: compose_sharpa_scene.SharpaSceneAssembly,
    entity_names: tuple[str, ...] | None = None,
) -> SceneCfg:
    if entity_names is None:
        return assembly.scene_cfg
    known = {spec.name: spec for spec in assembly.scene_cfg.entity_assets}
    return SceneCfg(
        entity_assets=tuple(known[name] for name in entity_names),
        contact_excludes=compose_sharpa_scene.scene_contact_exclude_pairs(
            entity_names, sources=assembly.entity_sources
        ),
    )


def _load_newton_deps():
    try:
        from unisim.backend.newton.dependencies import (
            NewtonDependencyError,
            load_newton_dependencies,
        )
    except ImportError as exc:  # pragma: no cover - unisim without newton extra
        pytest.skip(f"unisim newton backend unavailable: {exc}")
    try:
        return load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))


def test_hand_entity_xml_structure(tmp_path: Path) -> None:
    """Mounted-hand sources preserve joints, contacts, meshes and home state."""
    mujoco = pytest.importorskip("mujoco")  # untyped module (no stubs)

    from chord_unilab.assets.compose_sharpa_dual import (
        WRIST_Y_DEFAULT,
        WRIST_Z_DEFAULT,
    )

    for side in ("left", "right"):
        path = compose_sharpa_scene.hand_entity_path(side, cache_dir=tmp_path)
        assert path.is_file()
        root = ET.parse(path).getroot()
        assert len(root.findall("./contact/exclude")) == 202  # same-side GT pairs
        assert len(root.findall("./worldbody//body")) == 26  # 3 dummies+palm+22
        assert len(root.findall("./actuator/position")) == 25  # 3 slides+22 hinges
        assert len(root.findall("./actuator/velocity")) == 25
        joints = root.findall("./worldbody//joint")
        assert sum(j.get("type", "hinge") == "hinge" for j in joints) == 22
        assert sum(j.get("type") == "ball" for j in joints) == 1
        assert sum(j.get("type") == "slide" for j in joints) == 3
        for joint in joints:
            if joint.get("type") != "ball":
                assert joint.get("solreflimit") == "-10000 -10"
        # Newton collision-group bitmasks are stamped onto every collision geom.
        for geom in root.findall("./worldbody//geom"):
            if geom.get("contype") != "0":
                assert geom.get("contype") is not None
                assert geom.get("conaffinity") is not None
            if "elastomer" in (geom.get("mesh") or ""):
                np.testing.assert_allclose(
                    [float(value) for value in geom.get("quat", "").split()],
                    [0.5, 0.5, -0.5, 0.5],
                    atol=2e-6,
                    err_msg=f"{geom.get('mesh')}: fixed-child RPY missing",
                )

        model = mujoco.MjModel.from_xml_path(str(path))
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        assert key_id >= 0
        key_qpos = model.key_qpos[key_id]
        joint_adr = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): int(
                model.jnt_qposadr[j]
            )
            for j in range(model.njnt)
        }
        assert key_qpos[joint_adr[f"{side}_wrist_y"]] == pytest.approx(
            WRIST_Y_DEFAULT[side]
        )
        assert key_qpos[joint_adr[f"{side}_wrist_z"]] == pytest.approx(WRIST_Z_DEFAULT)
        ball_adr = joint_adr[f"{side}_wrist_rotvec"]
        np.testing.assert_allclose(
            key_qpos[ball_adr : ball_adr + 4], [1, 0, 0, 0], atol=1e-12
        )


def test_subset_scene_loads_in_unisim_newton(tmp_path: Path) -> None:
    """The composed rigid-object subset materializes, reads state and resets."""
    pytest.importorskip("torch")
    deps = _load_newton_deps()
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("newton smoke requires a CUDA Warp device")

    import unisim
    from unisim.entities import EntityStatePatch, SceneResetRequest

    assembly = _assembly(tmp_path)
    object_name = assembly.bindings[0].entity_name
    subset = (object_name, "support", "floor")
    backend = unisim.create_backend(
        "newton",
        scene=_scene_cfg(assembly, subset),
        num_envs=2,
        sim_dt=0.01,
        newton_device=str(device),
        newton_use_cuda_graph=False,
    )
    try:
        backend.materialize()
        layout = backend.get_physics_state_layout()
        parts = layout.split_state(backend.get_physics_state())
        assert parts.qpos.shape == (2, 14)  # rigid object 7 + support 7
        assert parts.qvel.shape == (2, 12)

        states = {name: backend.get_entity_state(name) for name in subset}
        assert states[object_name]["root_pose"].shape == (2, 7)
        assert states[object_name]["joint_positions"].shape == (2, 0)
        assert states["support"]["root_pose"].shape == (2, 7)
        assert states["support"]["joint_positions"].shape == (2, 0)
        assert states["floor"]["root_pose"].shape == (2, 7)
        assert states["floor"]["joint_positions"].shape == (2, 0)
        np.testing.assert_allclose(
            states[object_name]["root_pose"][0], [0, 0, 0, 1, 0, 0, 0], atol=1e-6
        )
        defaults = backend.get_entity_default_state(object_name)
        assert defaults["root_pose"].shape == (2, 7)
        assert defaults["joint_positions"].shape == (2, 0)
        np.testing.assert_allclose(
            defaults["root_pose"][0], [0, 0, 0, 1, 0, 0, 0], atol=1e-6
        )

        pose = np.array([[0.1, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        patch = EntityStatePatch(object_name, root_pose=pose)
        backend.reset_entities(SceneResetRequest(env_ids=(1,), patches=(patch,)))
        moved = backend.get_entity_state(object_name)
        np.testing.assert_allclose(moved["root_pose"][1], pose[0], atol=1e-6)
        np.testing.assert_allclose(  # env 0 untouched
            moved["root_pose"][0], [0, 0, 0, 1, 0, 0, 0], atol=1e-6
        )
    finally:
        backend.cleanup_scene_assets()


def test_full_scene_loads_in_unisim_newton(tmp_path: Path) -> None:
    """The full scene materializes, exposes public entity state and resets."""
    pytest.importorskip("torch")
    deps = _load_newton_deps()
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("newton smoke requires a CUDA Warp device")

    import unisim
    from unisim.entities import EntityStatePatch, SceneResetRequest

    assembly = _assembly(tmp_path)
    backend = unisim.create_backend(
        "newton",
        scene=assembly.scene_cfg,
        num_envs=2,
        sim_dt=0.01,
        newton_device=str(device),
        newton_use_cuda_graph=False,
    )
    try:
        backend.materialize()
        layout = backend.get_physics_state_layout()
        parts = layout.split_state(backend.get_physics_state())
        assert parts.qpos.shape == (2, 72)  # hands 58 + rigid object 7 + support 7
        assert parts.qvel.shape == (2, 68)

        states = {
            name: backend.get_entity_state(name) for name in assembly.entity_names
        }
        for side in compose_sharpa_scene.HAND_ENTITY_NAMES:
            assert states[side]["root_pose"].shape == (2, 7)
            assert states[side]["joint_positions"].shape == (2, 22)
            # Composed defaults are the mount origin (zero-relative source
            # columns; no default keyframe is declared); GT's wrist mount
            # defaults (y=∓0.25, z=0.5) live in the "home" keyframe and are
            # applied by runtime resets.
            np.testing.assert_allclose(
                states[side]["root_pose"][0], [0, 0, 0, 1, 0, 0, 0], atol=1e-6
            )
        object_name = assembly.bindings[0].entity_name
        assert states[object_name]["root_pose"].shape == (2, 7)
        assert states[object_name]["joint_positions"].shape == (2, 0)
        assert states["support"]["root_pose"].shape == (2, 7)
        assert states["floor"]["root_pose"].shape == (2, 7)

        pose = np.array([[0.1, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        fingers = np.full((1, 22), 0.3, dtype=np.float32)
        patch = EntityStatePatch("left_hand", root_pose=pose, joint_positions=fingers)
        backend.reset_entities(SceneResetRequest(env_ids=(1,), patches=(patch,)))
        moved = backend.get_entity_state("left_hand")
        np.testing.assert_allclose(moved["root_pose"][1], pose[0], atol=1e-6)
        np.testing.assert_allclose(moved["joint_positions"][1], fingers[0], atol=1e-6)
        np.testing.assert_allclose(  # env 0 untouched
            moved["root_pose"][0], [0, 0, 0, 1, 0, 0, 0], atol=1e-6
        )
        other = backend.get_entity_state("right_hand")
        np.testing.assert_allclose(  # other entity untouched
            other["root_pose"][1], [0, 0, 0, 1, 0, 0, 0], atol=1e-6
        )
    finally:
        backend.cleanup_scene_assets()


def _source_hand_mesh_files(side: str) -> dict[str, Path]:
    source_path = compose_sharpa_scene.ROBOT_DIR / "mjcf" / f"{side}_sharpawave.xml"
    source_root = ET.parse(source_path).getroot()
    source_asset = source_root.find("asset")
    assert source_asset is not None
    return {
        f"{side}_{mesh.get('name')}": (
            source_path.parent / str(mesh.get("file"))
        ).resolve()
        for mesh in source_asset.findall("mesh")
    }


def test_hand_entity_references_raw_source_meshes(tmp_path: Path) -> None:
    """Visual and collision geoms keep the source STL assets unchanged."""
    for side in ("left", "right"):
        path = compose_sharpa_scene.hand_entity_path(side, cache_dir=tmp_path)
        root = ET.parse(path).getroot()
        asset = root.find("asset")
        assert asset is not None
        by_name = {m.get("name"): m for m in asset.findall("mesh")}
        source_files = _source_hand_mesh_files(side)
        assert by_name.keys() == source_files.keys()
        for name, source_file in source_files.items():
            mesh = by_name[name]
            assert source_file.suffix.lower() == ".stl"
            assert mesh.get("file") == str(source_file)
            assert "maxhullvert" not in mesh.attrib

        source_root = ET.parse(
            compose_sharpa_scene.ROBOT_DIR / "mjcf" / f"{side}_sharpawave.xml"
        ).getroot()
        source_refs = [
            (f"{side}_{geom.get('mesh')}", geom.get("contype") == "0")
            for geom in source_root.iter("geom")
            if geom.get("mesh") is not None
        ]
        entity_refs = [
            (geom.get("mesh"), geom.get("contype") == "0")
            for geom in root.iter("geom")
            if geom.get("mesh") is not None
        ]
        assert entity_refs == source_refs
        assert sum(not visual for _, visual in entity_refs) == 26


def test_hand_entity_raw_meshes_compose_into_full_scene(tmp_path: Path) -> None:
    """Raw hand sources compose; Newton owns convex-hull construction."""
    assembly = _assembly(tmp_path)
    composed = compose_scene(
        assembly.scene_cfg,
        num_envs=1,
        sim_dt=compose_sharpa_scene.SIM_DT,
        preserve_physical_kinematic_free_joint=True,
    )
    try:
        assert Path(composed.model_file).is_file()
    finally:
        composed.close()
