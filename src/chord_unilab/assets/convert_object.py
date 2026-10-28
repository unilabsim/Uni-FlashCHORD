"""Convert one ``ObjectAssetSpec`` component into a self-contained MJCF fragment.

A component always has one free root. Its entity kind, actuator count, home
keyframe width and structural validation derive from zero or more declared
scalar articulations. Independent rigid components are converted independently.
Reference arrays are used only to validate bindings; initial and per-frame poses
belong to :mod:`chord_unilab.assets.reference_state` and never enter this cache.

URDF inertia, origins, scales and visual meshes are preserved. Collision meshes
are decomposed with the GT CoACD parameters into original-order NPZ arrays and
indexed OBJ files, without STL round-trips or ``maxhullvert``. Patched URDFs are
written in a managed temporary directory so source assets remain read-only.

The fragment cache key contains the schema, URDF bytes, complete object spec and
explicit physics configuration. Conversion does not depend on dataset-specific
scene defaults.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from chord_unilab.assets.actuators import append_position_velocity_actuators
from chord_unilab.assets.convert_sharpawave import root_inertial
from chord_unilab.assets.physics import (
    DEFAULT_SCENE_PHYSICS,
    ScenePhysics,
    contact_mjcf,
)
from chord_unilab.data import load_chord_reference
from chord_unilab.data.reference import ChordReference, ObjectAssetSpec

SCENE_PARAMS_PATH = (
    Path(__file__).parent / "robots" / "sharpawave" / "flash_chord_scene_params.json"
)

CACHE_ENV_VAR = "CHORD_CACHE_DIR"
FRAGMENT_SCHEMA = "chord-object-fragment-v6"

DEFAULT_PARQUET = (
    Path(__file__).resolve().parents[3]
    / "data/assets/human_motion_data/ego_recon/processed"
    / "sequence_id=tissue_box_simple/robot_name=sharpa_wave"
)
GT_CONTROL_FPS = 20.0
GT_MOTION_SPEED = 0.5

COACD_PARAMS = {"threshold": 0.05, "max_convex_hull": -1, "preprocess_mode": "auto"}
_COACD_CACHE_VERSION = 1


def cache_root() -> Path:
    """Fragment/hull cache root (outside the repo; env CHORD_CACHE_DIR overrides)."""
    return Path(os.environ.get(CACHE_ENV_VAR, Path.home() / ".cache" / "chord_unilab"))


def default_parquet() -> Path | None:
    """CHORD_TEST_PARQUET or the repository tissue demo, if it exists."""
    override = os.environ.get("CHORD_TEST_PARQUET")
    candidate = Path(override) if override else DEFAULT_PARQUET
    return candidate if candidate.exists() else None


def convex_cache_key(vertices: np.ndarray, faces: np.ndarray, params: dict) -> str:
    """flash_chord assets/mesh.py key: sha1(version | f64 vertices | i64 faces | params)."""
    h = hashlib.sha1()
    h.update(f"v{_COACD_CACHE_VERSION}".encode())
    h.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
    h.update(repr(sorted(params.items())).encode())
    return h.hexdigest()[:16]


def _load_mesh_arrays(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh: Any = trimesh.load(str(mesh_path), force="mesh")
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def _write_indexed_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for vertex in np.asarray(vertices, dtype=np.float32):
            values = " ".join(f"{float(value):.9g}" for value in vertex)
            stream.write(f"v {values}\n")
        for face in np.asarray(faces, dtype=np.int32).reshape(-1, 3):
            indices = " ".join(str(int(value) + 1) for value in face)
            stream.write(f"f {indices}\n")


def _convex_cache_dir(
    vertices: np.ndarray, faces: np.ndarray, cache_dir: Path | None
) -> Path:
    key = convex_cache_key(vertices, faces, COACD_PARAMS)
    return (cache_dir or cache_root()) / "convex" / key


def _read_hulls(cache_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    with np.load(cache_path, allow_pickle=False) as cached:
        return [
            (
                np.ascontiguousarray(cached[f"v{i}"], dtype=np.float32),
                np.ascontiguousarray(cached[f"f{i}"], dtype=np.int32).reshape(-1, 3),
            )
            for i in range(int(cached["n"]))
        ]


def _reference_hull_cache(reference: ChordReference | None, key: str) -> Path | None:
    if reference is None or reference.metadata.sequence_id is None:
        return None
    from chord_unilab.data.reference_assets import reference_coacd_cache

    try:
        cache = reference_coacd_cache(reference.metadata.sequence_id, key)
    except KeyError:
        return None
    return None if cache is None else cache.path


def convex_hulls(
    mesh_path: Path,
    *,
    cache_dir: Path | None = None,
    reference: ChordReference | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load reference hulls or decompose once, preserving exact array ordering."""
    vertices, faces = _load_mesh_arrays(mesh_path)
    key = convex_cache_key(vertices, faces, COACD_PARAMS)
    out_dir = _convex_cache_dir(vertices, faces, cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "hulls.npz"
    import fcntl

    with (out_dir / "hulls.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            reference_cache = (
                None if cache_path.exists() else _reference_hull_cache(reference, key)
            )
            if cache_path.exists():
                hulls = _read_hulls(cache_path)
            elif reference_cache is not None:
                hulls = _read_hulls(reference_cache)
            else:
                import coacd

                coacd.set_log_level("error")
                parts = coacd.run_coacd(coacd.Mesh(vertices, faces), **COACD_PARAMS)
                hulls = [
                    (
                        np.ascontiguousarray(part_vertices, dtype=np.float32),
                        np.ascontiguousarray(part_faces, dtype=np.int32).reshape(-1, 3),
                    )
                    for part_vertices, part_faces in parts
                ]
                if not hulls:
                    raise ValueError(
                        f"CoACD returned no hulls for collision mesh {mesh_path}"
                    )
                arrays: dict[str, np.ndarray] = {
                    "n": np.asarray(len(hulls), dtype=np.int64)
                }
                for index, (hull_vertices, hull_faces) in enumerate(hulls):
                    arrays[f"v{index}"] = hull_vertices
                    arrays[f"f{index}"] = hull_faces
                temporary = out_dir / "hulls.tmp.npz"
                np.savez(temporary, allow_pickle=False, **arrays)
                os.replace(temporary, cache_path)
            if not hulls:
                raise ValueError(f"empty collision hull cache for {mesh_path}")
            for index, (hull_vertices, hull_faces) in enumerate(hulls):
                obj_path = out_dir / f"hull_{index:03d}.obj"
                temporary_obj = out_dir / f"hull_{index:03d}.tmp.obj"
                _write_indexed_obj(temporary_obj, hull_vertices, hull_faces)
                os.replace(temporary_obj, obj_path)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return hulls


def _convex_hull_paths(
    mesh_path: Path,
    cache_dir: Path | None,
    reference: ChordReference | None = None,
) -> tuple[Path, ...]:
    vertices, faces = _load_mesh_arrays(mesh_path)
    out_dir = _convex_cache_dir(vertices, faces, cache_dir)
    hulls = convex_hulls(mesh_path, cache_dir=cache_dir, reference=reference)
    return tuple(out_dir / f"hull_{index:03d}.obj" for index in range(len(hulls)))


def _resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        raise ValueError(f"package:// collision mesh is unsupported: {filename}")
    path = Path(filename.removeprefix("file://"))
    if not path.is_absolute():
        path = urdf_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"collision mesh does not exist: {path}")
    return path


def validate_object_asset_spec(
    reference: ChordReference, spec: ObjectAssetSpec
) -> None:
    """Check one component's bindings against the reference schema, not its frames."""
    body_names = set(reference.object_body_names())
    missing = {body.reference_name for body in spec.bodies} - body_names
    if missing:
        raise ValueError(
            f"object asset {spec.name!r} has unknown reference bodies {sorted(missing)}"
        )
    articulation_count = reference.object_articulation().shape[1]
    invalid = tuple(
        articulation.reference_index
        for articulation in spec.articulations
        if articulation.reference_index >= articulation_count
    )
    if invalid:
        raise ValueError(
            f"object asset {spec.name!r} has articulation columns {invalid}, "
            f"but the reference has {articulation_count}"
        )


class _FragmentInputs:
    def __init__(
        self,
        reference: ChordReference,
        spec: ObjectAssetSpec,
        physics: ScenePhysics,
    ) -> None:
        validate_object_asset_spec(reference, spec)
        self.reference = reference
        self.spec = spec
        self.physics = physics
        self.urdf_path = Path(spec.urdf_path)
        if not self.urdf_path.is_file():
            raise ValueError(f"resolved object URDF does not exist: {self.urdf_path}")
        self.root_link = _urdf_root_link(self.urdf_path)
        self.root_body = f"{spec.name}_{self.root_link}"
        self.free_joint = f"{spec.name}_floating_base"
        self.hinge_joints = {
            articulation.reference_index: f"{spec.name}_{articulation.simulation_joint_name}"
            for articulation in spec.articulations
        }

    def cache_key(self) -> str:
        h = hashlib.sha1()
        h.update(FRAGMENT_SCHEMA.encode())
        h.update(self.urdf_path.read_bytes())
        h.update(repr(self.spec).encode())
        h.update(repr(self.physics).encode())
        return h.hexdigest()[:16]

    def fragment_dir(self, cache_dir: Path | None = None) -> Path:
        return (cache_dir or cache_root()) / "objects" / self.cache_key()


def _prepare(
    parquet_path: str | Path,
    control_fps: float,
    motion_speed: float,
    physics: ScenePhysics,
) -> tuple[ChordReference, _FragmentInputs]:
    reference = load_chord_reference(
        parquet_path, control_fps=control_fps, motion_speed=motion_speed
    )
    assets = reference.object_assets()
    if len(assets) != 1:
        raise ValueError(
            f"parquet compatibility conversion expects one object asset, got {len(assets)}"
        )
    return reference, _FragmentInputs(reference, assets[0], physics)


def fragment_path_for(
    parquet_path: str | Path,
    *,
    control_fps: float = GT_CONTROL_FPS,
    motion_speed: float = GT_MOTION_SPEED,
    cache_dir: Path | None = None,
    physics: ScenePhysics = DEFAULT_SCENE_PHYSICS,
) -> Path:
    """Compatibility cache path for a single-component parquet."""
    _, inputs = _prepare(parquet_path, control_fps, motion_speed, physics)
    return inputs.fragment_dir(cache_dir) / f"{inputs.spec.name}.xml"


def _urdf_root_link(urdf_path: Path) -> str:
    root = ET.parse(urdf_path).getroot()
    if root.get("name") is None:
        raise ValueError(f"{urdf_path}: <robot> has no name")
    children = {
        child.get("link")
        for joint in root.findall("joint")
        if (child := joint.find("child")) is not None
    }
    roots = [
        name
        for link in root.findall("link")
        if (name := link.get("name")) is not None and name not in children
    ]
    if len(roots) != 1:
        raise ValueError(f"{urdf_path}: expected exactly one root link, got {roots}")
    return roots[0]


def _replace_collision_meshes(
    root: ET.Element,
    urdf_path: Path,
    cache_dir: Path | None,
    reference: ChordReference | None = None,
) -> None:
    from copy import deepcopy

    parents = {child: parent for parent in root.iter() for child in parent}
    for collision in tuple(root.iter("collision")):
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            continue
        filename = mesh.get("filename")
        if filename is None:
            raise ValueError(f"{urdf_path}: collision mesh has no filename")
        source = _resolve_mesh_path(urdf_path, filename)
        hull_paths = _convex_hull_paths(source, cache_dir, reference)
        parent = parents[collision]
        index = list(parent).index(collision)
        for offset, hull_path in enumerate(hull_paths):
            replacement = deepcopy(collision)
            replacement_mesh = replacement.find("geometry/mesh")
            assert replacement_mesh is not None
            replacement_mesh.set("filename", str(hull_path))
            parent.insert(index + offset, replacement)
        parent.remove(collision)


def _convert_urdf(
    urdf_path: Path,
    cache_dir: Path | None = None,
    reference: ChordReference | None = None,
) -> ET.Element:
    """URDF -> MJCF without writing beside the read-only source asset."""
    root = ET.parse(urdf_path).getroot()
    mujoco_element = root.find("mujoco")
    if mujoco_element is None:
        mujoco_element = ET.Element("mujoco")
        root.insert(0, mujoco_element)
    compiler = mujoco_element.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_element, "compiler")
    compiler.set("discardvisual", "false")
    _replace_collision_meshes(root, urdf_path, cache_dir, reference)
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename is not None:
            mesh.set("filename", str(_resolve_mesh_path(urdf_path, filename)))
    with tempfile.TemporaryDirectory(prefix="chord-object-urdf-") as temporary_dir:
        patched_path = Path(temporary_dir) / f"{urdf_path.stem}.patched.urdf"
        ET.ElementTree(root).write(
            patched_path, xml_declaration=True, encoding="unicode"
        )
        spec = mujoco.MjSpec.from_file(str(patched_path))
        spec.compile()
    return ET.fromstring(spec.to_xml())


def _fmt(values: list[float]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _rewrap_root(converted: ET.Element, inputs: _FragmentInputs) -> None:
    """Re-wrap the welded root link as a real body with inertial + freejoint."""
    worldbody = converted.find("worldbody")
    assert worldbody is not None
    root_body = ET.Element("body", {"name": inputs.root_body})
    root_body.append(
        ET.Comment(
            " inertial restored from the URDF root link (MuJoCo's URDF parser "
            "discards it); free root added per flash_chord scene/objects.py "
            "add_urdf(floating=True) "
        )
    )
    ET.SubElement(
        root_body, "inertial", root_inertial(inputs.urdf_path, inputs.root_link)
    )
    ET.SubElement(root_body, "freejoint", {"name": inputs.free_joint})
    for child in list(worldbody):
        worldbody.remove(child)
        root_body.append(child)
    worldbody.append(root_body)


def _rename_and_stamp(converted: ET.Element, inputs: _FragmentInputs) -> None:
    """Apply component namespacing and the declarative articulation contract."""
    prefix = f"{inputs.spec.name}_"
    worldbody = converted.find("worldbody")
    assert worldbody is not None
    for body in worldbody.iter("body"):
        name = body.get("name")
        if name != inputs.root_body:
            body.set("name", f"{prefix}{name}")
    articulations = {
        f"{prefix}{articulation.simulation_joint_name}": articulation
        for articulation in inputs.spec.articulations
    }
    for joint in worldbody.iter("joint"):
        name = joint.get("name")
        if name is None:
            continue
        joint.set("name", f"{prefix}{name}")
        articulation = articulations.get(joint.get("name", ""))
        if articulation is None:
            continue
        joint.set("armature", f"{articulation.physics.armature:.9g}")
        joint.set("frictionloss", f"{articulation.physics.friction:.9g}")
        joint.set("damping", "0")
        joint.set("solreflimit", _fmt(list(inputs.physics.joint_solreflimit)))
        joint.set("solimplimit", _fmt(list(inputs.physics.joint_solimplimit)))
        joint.set(
            "actuatorfrcrange",
            f"{-articulation.drive.effort_limit:.9g} {articulation.drive.effort_limit:.9g}",
        )


def _stamp_geom_coloring(converted: ET.Element, inputs: _FragmentInputs) -> None:
    """Assign object collision ownership without dataset-specific snapshots."""
    group = inputs.physics.collision.object_group
    conaffinity = inputs.physics.collision.conaffinity(group)
    for geom in converted.iter("geom"):
        if geom.get("contype") == "0":
            geom.set("conaffinity", "0")
        else:
            geom.set("contype", str(group))
            geom.set("conaffinity", str(conaffinity))


def _add_excludes(root: ET.Element) -> None:
    """Explicit parent-child <exclude> pairs, mirroring newton's importer.

    MuJoCo filters parent-child contacts implicitly at collision time, but the
    GT full-scene model lists them in ``exclude_signature`` (newton's importer
    materializes them), so the fragment states them explicitly to match.
    """
    worldbody = root.find("worldbody")
    assert worldbody is not None
    pairs: list[tuple[str, str]] = []

    def walk(parent: ET.Element) -> None:
        for body in parent.findall("body"):
            if parent.tag == "body":
                pairs.append((str(parent.get("name")), str(body.get("name"))))
            walk(body)

    walk(worldbody)
    if not pairs:
        return
    contact = ET.SubElement(root, "contact")
    contact.append(
        ET.Comment(
            " newton's importer lists parent-child pairs in exclude_signature; "
            "MuJoCo would filter these implicitly "
        )
    )
    for parent_name, child_name in pairs:
        ET.SubElement(contact, "exclude", {"body1": parent_name, "body2": child_name})


def _assemble(converted: ET.Element, inputs: _FragmentInputs) -> ET.Element:
    root = ET.Element("mujoco", {"model": inputs.spec.name})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", inputs.physics.solver.as_mjcf())
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contact_mjcf(inputs.physics.contact))
    converted_asset = converted.find("asset")
    if converted_asset is not None:
        asset = ET.SubElement(root, "asset")
        for child in list(converted_asset):
            converted_asset.remove(child)
            asset.append(child)
    converted_worldbody = converted.find("worldbody")
    assert converted_worldbody is not None
    worldbody = ET.SubElement(root, "worldbody")
    for child in list(converted_worldbody):
        converted_worldbody.remove(child)
        worldbody.append(child)
    _add_excludes(root)

    if inputs.spec.articulations:
        actuator = ET.SubElement(root, "actuator")
        for articulation in inputs.spec.articulations:
            name = f"{inputs.spec.name}_{articulation.simulation_joint_name}"
            append_position_velocity_actuators(
                actuator,
                name=name,
                joint=name,
                kp=float(articulation.drive.kp),
                kd=float(articulation.drive.kd),
            )

    qpos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * len(inputs.spec.articulations)
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", {"name": "home", "qpos": _fmt(qpos)})
    return root


def _validate(xml_path: Path, inputs: _FragmentInputs) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    articulation_count = len(inputs.spec.articulations)
    print(
        f"{inputs.spec.name}: nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} "
        f"nu={model.nu} nq={model.nq} nv={model.nv} nkey={model.nkey}"
    )
    expected = (
        1 + articulation_count,
        1 + len(inputs.spec.bodies),
        2 * articulation_count,
    )
    if (model.njnt, model.nbody, model.nu) != expected:
        raise ValueError(
            f"expected njnt/nbody/nu={expected}, got {model.njnt}/{model.nbody}/{model.nu}"
        )
    if (model.nq, model.nv) != (7 + articulation_count, 6 + articulation_count):
        raise ValueError(
            f"unexpected free-root/articulation widths nq={model.nq} nv={model.nv}"
        )
    required_bodies = {inputs.root_body} | {
        f"{inputs.spec.name}_{body.simulation_name}"
        for body in inputs.spec.bodies
        if body.simulation_name is not None
    }
    for name in required_bodies:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"{name} missing after conversion")
        if not np.isfinite(model.body_mass[body_id]) or model.body_mass[body_id] <= 0.0:
            raise ValueError(f"{name} must have positive finite URDF inertial mass")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, inputs.free_joint) < 0:
        raise ValueError(f"{inputs.free_joint} missing after conversion")
    movable = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    }
    if movable != set(inputs.hinge_joints.values()):
        raise ValueError(
            f"articulation declaration mismatch: imported={sorted(movable)}, "
            f"declared={sorted(inputs.hinge_joints.values())}"
        )
    for articulation in inputs.spec.articulations:
        name = inputs.hinge_joints[articulation.reference_index]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(f"{name} must be a scalar hinge or slide joint")
        dof_id = int(model.jnt_dofadr[joint_id])
        if not np.isclose(model.dof_armature[dof_id], articulation.physics.armature):
            raise ValueError(f"{name}: armature drifted from its spec")
    if model.nkey != 1 or model.key_qpos.shape[1] != model.nq:
        raise ValueError(f"expected one home keyframe of width nq, got {model.nkey}")
    expected_home = np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * articulation_count
    )
    if not np.allclose(model.key_qpos[0], expected_home, atol=1e-12, rtol=0.0):
        raise ValueError("object home keyframe must be independent of reference frames")


def fragment_path(
    reference: ChordReference,
    spec: ObjectAssetSpec,
    *,
    cache_dir: Path | None = None,
    physics: ScenePhysics = DEFAULT_SCENE_PHYSICS,
) -> Path:
    inputs = _FragmentInputs(reference, spec, physics)
    return inputs.fragment_dir(cache_dir) / f"{spec.name}.xml"


def convert_object(
    reference: ChordReference,
    spec: ObjectAssetSpec,
    *,
    cache_dir: Path | None = None,
    physics: ScenePhysics = DEFAULT_SCENE_PHYSICS,
) -> Path:
    """Build one spec-declared object component into the external cache."""
    inputs = _FragmentInputs(reference, spec, physics)
    out_dir = inputs.fragment_dir(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}.xml"

    converted = _convert_urdf(inputs.urdf_path, cache_dir, inputs.reference)
    _rewrap_root(converted, inputs)
    _rename_and_stamp(converted, inputs)
    _stamp_geom_coloring(converted, inputs)
    root = _assemble(converted, inputs)

    tmp_path = out_path.with_suffix(".tmp.xml")
    ET.ElementTree(root).write(tmp_path, xml_declaration=True, encoding="unicode")
    _validate(tmp_path, inputs)
    os.replace(tmp_path, out_path)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": FRAGMENT_SCHEMA,
                "cache_key": inputs.cache_key(),
                "object_spec": repr(spec),
                "physics": repr(physics),
                "urdf_path": str(inputs.urdf_path),
                "urdf_sha256": hashlib.sha256(
                    inputs.urdf_path.read_bytes()
                ).hexdigest(),
                "generated_at": datetime.now(UTC).isoformat(),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {out_path}")
    return out_path


def convert_object_from_parquet(
    parquet_path: str | Path,
    *,
    control_fps: float = GT_CONTROL_FPS,
    motion_speed: float = GT_MOTION_SPEED,
    cache_dir: Path | None = None,
    physics: ScenePhysics = DEFAULT_SCENE_PHYSICS,
) -> Path:
    """Thin compatibility wrapper for the original single-component API."""
    reference, inputs = _prepare(parquet_path, control_fps, motion_speed, physics)
    return convert_object(reference, inputs.spec, cache_dir=cache_dir, physics=physics)


def main() -> None:
    import sys

    parquet = Path(sys.argv[1]) if len(sys.argv) > 1 else default_parquet()
    if parquet is None:
        raise ValueError(
            f"no parquet given and none found (set CHORD_TEST_PARQUET or pass one; "
            f"default {DEFAULT_PARQUET} missing)"
        )
    convert_object_from_parquet(parquet)


if __name__ == "__main__":
    main()
