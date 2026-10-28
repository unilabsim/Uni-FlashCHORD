"""Convert the SharpaWave hand URDFs into single-hand MJCF.

Source: ``robots/sharpawave/urdf/{left,right}_sharpawave.urdf`` (verbatim copies of
flash_chord's authoritative hand models). Regenerate with::

    uv run python -m chord_unilab.assets.convert_sharpawave

Only the mesh path prefix is rewritten (flash_chord's asset layout → our sibling
``meshes/`` dir); every physical property comes from the URDF untouched.

MuJoCo's URDF parser welds the root link (palm) into the worldbody and discards
its inertial, so after conversion we re-wrap the palm as a real body, restore the
URDF root-link inertial, and attach the ball wrist joint
(``{side}_wrist_rotvec``) that flash_chord adds via ``add_urdf(base_joint=BALL)``.

NOTE (架构取舍): the ball joint conceptually belongs to the floating-mount layer,
but MJCF ``<include>`` cannot inject a joint into included content, so it lives
in this single-hand file. The 3-slide translation mount stays in the scene layer
(``compose_sharpa_dual.py``).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROBOT_DIR = Path(__file__).parent / "robots" / "sharpawave"
MESH_SOURCE_PREFIX = "../../meshes/sharpawave/"
MESH_TARGET_PREFIX = "../meshes/"

EXPECTED_REVOLUTE_JOINTS = 22
EXPECTED_GEOMS = 54

# flash_chord SharpaHands._add_hand: ball wrist joint dof properties.
BALL_ARMATURE = 0.01
BALL_FRICTIONLOSS = 0.01
# flash_chord SharpaHandsConfig: effort-PD kd is applied as mujoco:dof_passive_damping.
BALL_DAMPING = 6.0


def _attr(element: ET.Element, name: str) -> str:
    value = element.get(name)
    if value is None:
        raise ValueError(f"missing attribute {name!r} on <{element.tag}>")
    return value


def root_inertial(urdf_path: Path, root_link: str) -> dict[str, str]:
    """Read the URDF root link's <inertial> (discarded by MuJoCo's URDF parser)."""
    root = ET.parse(urdf_path).getroot()
    link = root.find(f"link[@name='{root_link}']")
    if link is None:
        raise ValueError(f"{urdf_path}: root link {root_link!r} not found")
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError(f"{urdf_path}: root link {root_link!r} has no <inertial>")
    origin = inertial.find("origin")
    inertia = inertial.find("inertia")
    mass = inertial.find("mass")
    assert origin is not None and inertia is not None and mass is not None
    i = {
        k: float(_attr(inertia, k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    }
    return {
        "pos": origin.get("xyz", "0 0 0"),
        "mass": _attr(mass, "value"),
        # MJCF fullinertia order: Ixx Iyy Izz Ixy Ixz Iyz.
        "fullinertia": f"{i['ixx']} {i['iyy']} {i['izz']} {i['ixy']} {i['ixz']} {i['iyz']}",
    }


def _urdf_inertia_matrix(urdf_path: Path, root_link: str) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    inertia = root.find(f"link[@name='{root_link}']/inertial/inertia")
    assert inertia is not None
    i = {
        k: float(_attr(inertia, k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    }
    return np.array(
        [
            [i["ixx"], i["ixy"], i["ixz"]],
            [i["ixy"], i["iyy"], i["iyz"]],
            [i["ixz"], i["iyz"], i["izz"]],
        ]
    )


def _rewrap_palm(xml_path: Path, side: str, urdf_path: Path) -> None:
    """Wrap worldbody content into a palm body with URDF inertial + ball joint."""
    tree = ET.parse(xml_path)
    worldbody = tree.getroot().find("worldbody")
    assert worldbody is not None
    palm_name = f"{side}_hand_C_MC"

    palm = ET.Element("body", {"name": palm_name})
    palm.append(
        ET.Comment(
            " inertial restored from the URDF root link (MuJoCo's URDF parser "
            "discards it); ball wrist joint added per flash_chord "
            "SharpaHands._add_hand (base_joint=BALL) "
        )
    )
    ET.SubElement(palm, "inertial", root_inertial(urdf_path, palm_name))
    ET.SubElement(
        palm,
        "joint",
        {
            "name": f"{side}_wrist_rotvec",
            "type": "ball",
            "armature": str(BALL_ARMATURE),
            "frictionloss": str(BALL_FRICTIONLOSS),
            "damping": str(BALL_DAMPING),
        },
    )
    for child in list(worldbody):
        worldbody.remove(child)
        palm.append(child)
    worldbody.append(palm)
    tree.write(xml_path, xml_declaration=True, encoding="unicode")


def _fixed_geom_frames(urdf_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mesh basename -> its fixed-child frame relative to the retained MJCF body.

    MuJoCo's URDF importer folds fixed children into their nearest dynamic
    body. That can preserve the compiled MJ geom pose through mesh recentering
    while losing the explicit parent->child transform Newton's importer needs on
    its native ``shape_transform``. Mapping by mesh basename is unambiguous for
    this asset: every shared collision/visual mesh from a fixed child carries
    one fixed frame per hand.
    """
    root = ET.parse(urdf_path).getroot()
    fixed_parent: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        assert parent is not None and child is not None
        parent_name = _attr(parent, "link")
        child_name = _attr(child, "link")
        xyz = (
            np.zeros(3)
            if origin is None
            else np.asarray(
                [float(value) for value in origin.get("xyz", "0 0 0").split()]
            )
        )
        rpy = (
            np.zeros(3)
            if origin is None
            else np.asarray(
                [float(value) for value in origin.get("rpy", "0 0 0").split()]
            )
        )
        fixed_parent[child_name] = (
            parent_name,
            xyz,
            Rotation.from_euler("xyz", rpy).as_quat(),
        )

    def compose(
        left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        position = left[0] + Rotation.from_quat(left[1]).apply(right[0])
        quaternion = (
            Rotation.from_quat(left[1]) * Rotation.from_quat(right[1])
        ).as_quat()
        return position, quaternion

    frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for link in root.findall("link"):
        chain: list[tuple[np.ndarray, np.ndarray]] = []
        current = _attr(link, "name")
        while current in fixed_parent:
            parent_link, xyz, xyzw = fixed_parent[current]
            chain.append((xyz, xyzw))
            current = parent_link
        if not chain:
            continue
        frame = (np.zeros(3), np.asarray([0.0, 0.0, 0.0, 1.0]))
        for transform in chain:
            frame = compose(frame, transform)
        for geometry in link.findall("visual/geometry/mesh") + link.findall(
            "collision/geometry/mesh"
        ):
            filename = Path(_attr(geometry, "filename")).name
            for key in (filename, Path(filename).stem):
                previous = frames.setdefault(key, (frame[0].copy(), frame[1].copy()))
                if not np.allclose(previous[0], frame[0]) or not np.allclose(
                    previous[1], frame[1], atol=2e-6
                ):
                    raise ValueError(f"mesh {key!r} has conflicting fixed frames")
    return frames


def _restore_fixed_geom_frames(xml_path: Path, urdf_path: Path) -> None:
    """Restore fixed-child RPY lost by URDF fixed-joint folding."""
    tree = ET.parse(xml_path)
    frames = _fixed_geom_frames(urdf_path)
    for geom in tree.getroot().iter("geom"):
        mesh = geom.get("mesh")
        if mesh is None or Path(mesh).name not in frames:
            continue
        position, xyzw = frames[Path(mesh).name]
        if geom.get("pos") not in (None, "0 0 0"):
            raise ValueError(f"fixed-child geom {mesh!r} unexpectedly has a local pos")
        if geom.get("quat") is not None:
            raise ValueError(f"fixed-child geom {mesh!r} unexpectedly has a local quat")
        geom.set("pos", " ".join(f"{value:.9g}" for value in position))
        geom.set("quat", " ".join(f"{value:.9g}" for value in xyzw[[3, 0, 1, 2]]))
    tree.write(xml_path, xml_declaration=True, encoding="unicode")


def _validate(xml_path: Path, side: str, urdf_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    hinge = sum(
        1 for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    )
    ball = sum(
        1 for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL
    )
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_C_MC")
    if palm_id < 0:
        raise ValueError(f"{side}: palm body missing after re-wrap")
    palm_mass = float(model.body_mass[palm_id])
    urdf_mass = float(root_inertial(urdf_path, f"{side}_hand_C_MC")["mass"])

    # Inertia check via eigenvalues (frame rotation between URDF tensor and
    # MuJoCo's diagonalized body_inertia does not affect the spectrum).
    mj_eig = np.sort(model.body_inertia[palm_id])
    urdf_eig = np.sort(
        np.linalg.eigvalsh(_urdf_inertia_matrix(urdf_path, f"{side}_hand_C_MC"))
    )

    print(
        f"{side}: nbody={model.nbody} njnt={model.njnt} hinge={hinge} ball={ball} "
        f"ngeom={model.ngeom} nmesh={model.nmesh} palm_mass={palm_mass:.4f}kg "
        f"total={model.body_subtreemass[0]:.4f}kg"
    )
    if hinge != EXPECTED_REVOLUTE_JOINTS or ball != 1:
        raise ValueError(
            f"{side}: expected {EXPECTED_REVOLUTE_JOINTS} hinge + 1 ball, got {hinge}+{ball}"
        )
    if not np.isclose(palm_mass, urdf_mass, rtol=1e-6):
        raise ValueError(f"{side}: palm mass {palm_mass} != URDF {urdf_mass}")
    if not np.allclose(mj_eig, urdf_eig, rtol=1e-4):
        raise ValueError(f"{side}: palm inertia spectrum {mj_eig} != URDF {urdf_eig}")


def convert_hand(side: str) -> Path:
    if side not in ("left", "right"):
        raise ValueError(f"unknown side {side!r}")
    urdf_path = ROBOT_DIR / "urdf" / f"{side}_sharpawave.urdf"
    mjcf_dir = ROBOT_DIR / "mjcf"
    mjcf_dir.mkdir(exist_ok=True)
    out_path = mjcf_dir / f"{side}_sharpawave.xml"

    patched = urdf_path.read_text().replace(MESH_SOURCE_PREFIX, MESH_TARGET_PREFIX)
    # MuJoCo's URDF parser defaults discardvisual=true, which would drop the
    # 28 visual geoms per hand; flash_chord keeps all 54 shapes per hand.
    # The flag applies at parse time, so inject the URDF <mujoco> extension
    # right after the <robot ...> opening tag.
    opening = patched.index(">", patched.index("<robot")) + 1
    patched = (
        patched[:opening]
        + '\n  <mujoco><compiler discardvisual="false"/></mujoco>'
        + patched[opening:]
    )
    patched_path = urdf_path.with_suffix(".patched.urdf")
    try:
        patched_path.write_text(patched)
        spec = mujoco.MjSpec.from_file(str(patched_path))
        spec.compile()
    finally:
        patched_path.unlink(missing_ok=True)

    out_path.write_text(spec.to_xml())
    _restore_fixed_geom_frames(out_path, urdf_path)
    _rewrap_palm(out_path, side, urdf_path)
    _validate(out_path, side, urdf_path)
    return out_path


def main() -> None:
    for side in ("left", "right"):
        convert_hand(side)


if __name__ == "__main__":
    main()
