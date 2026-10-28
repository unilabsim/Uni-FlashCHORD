"""Compose the dual floating-hand SharpaWave scene MJCF.

Merges the two single-hand files (from
:mod:`chord_unilab.assets.convert_sharpawave`) into one self-contained
``mjcf/sharpa_dual_floating.xml``, replicating
``flash_chord.embodiments.sharpa_hands.SharpaHands._add_hand``::

    3 dummy links (mass 1e-4) + 3 prismatic wrist joints (x/y/z)
        -> palm body with ball wrist joint (from the single-hand file)

MJCF ``<include>`` cannot be nested inside ``<body>`` (parse error in MuJoCo
3.11) and both hand files share one asset namespace, so this composer inlines
the hand content and prefixes mesh asset names per side — mirroring newton's
per-side namespacing when flash_chord adds both URDFs to one model.

All physics parameters are backfilled verbatim from
``robots/sharpawave/flash_chord_params.json``: geom contact params, per-geom
contype/conaffinity bitmasks, joint actfrcrange/armature, the 639
contact-exclusion pairs (incl. the 235 cross-hand pairs — the two hands do
not collide with each other in flash_chord), and the corrected dummy-link
inertia.

Regenerate with::

    uv run python -m chord_unilab.assets.compose_sharpa_dual

Known gaps (declared per repo policy):

- GAP-WRIST-EFFORT: the wrist ball joints are deliberately unactuated.
  flash_chord drives orientation by an effort PD (kp=25, tau_max=45) applied
  per substep through ``control.joint_f`` (runtime/control.py); kd=6 is dof
  damping (kept here). The joint_f application belongs to the task action term.
- GAP-SENSOR: contact-force sensor declarations (fingertip x object geom pairs)
  live in the task scene layer where the object fragment exists — not here.
- The 0.002 s ``opt.timestep`` in flash_chord's mj_model is the untouched
  MuJoCo default: ``SolverMuJoCo.step(dt)`` overwrites it every step. The
  effective dt comes from the recipe's sim group (``configs/sim/rl.yaml``:
  fps=20 x substeps=5 -> 0.01 s, 20 Hz control / 100 Hz physics), set below
  for documentation; the backend overrides it at runtime anyway. NB: the
  SimConfig *class* defaults (fps=50 x substeps=4) are NOT what
  sharpa_flash_sac runs with. Solver capacities there (njmax=512,
  nconmax=256) are backend config, not MJCF fields.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from chord_unilab.assets.actuators import append_position_velocity_actuators

ROBOT_DIR = Path(__file__).parent / "robots" / "sharpawave"
OUT_PATH = ROBOT_DIR / "mjcf" / "sharpa_dual_floating.xml"
PARAMS_PATH = ROBOT_DIR / "flash_chord_params.json"

PARAMS = json.loads(PARAMS_PATH.read_text())

# flash_chord SharpaHandsConfig joint_overrides default positions.
WRIST_Y_DEFAULT = {"left": -0.25, "right": 0.25}
WRIST_Z_DEFAULT = 0.5

_SLIDES = (("x", "px", "1 0 0"), ("y", "py", "0 1 0"), ("z", "pz", "0 0 1"))

_DUMMY_INERTIA = PARAMS["bodies"]["left_wrist_px_body"]["inertia"]
_DUMMY_MASS = PARAMS["bodies"]["left_wrist_px_body"]["mass"]


def _fmt(values: list[float]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def mount_with_hand(side: str, assets: list[ET.Element]) -> ET.Element:
    """Mount chain (3 dummy bodies + slides) with the side's palm inlined."""
    hand_tree = ET.parse(ROBOT_DIR / "mjcf" / f"{side}_sharpawave.xml")
    hand_root = hand_tree.getroot()

    asset = hand_root.find("asset")
    assert asset is not None
    for mesh in asset.findall("mesh"):
        original = mesh.get("name")
        prefixed = f"{side}_{original}"
        mesh.set("name", prefixed)
        assets.append(mesh)
    hand_worldbody = hand_root.find("worldbody")
    assert hand_worldbody is not None
    palm = hand_worldbody.find("body")
    assert palm is not None and palm.get("name") == f"{side}_hand_C_MC"
    for geom in palm.iter("geom"):
        mesh_ref = geom.get("mesh")
        if mesh_ref is not None:
            geom.set("mesh", f"{side}_{mesh_ref}")

    bodies: dict[str, ET.Element] = {}
    for _, tag, _ in _SLIDES:
        body = ET.Element("body", {"name": f"{side}_wrist_{tag}_body"})
        ET.SubElement(
            body,
            "inertial",
            {
                "mass": f"{_DUMMY_MASS:.9g}",
                "pos": "0 0 0",
                "diaginertia": _fmt(_DUMMY_INERTIA),
            },
        )
        bodies[tag] = body
    for name, tag, axis in _SLIDES:
        joint = PARAMS["joints"][f"{side}_wrist_{name}"]
        ET.SubElement(
            bodies[tag],
            "joint",
            {
                "name": f"{side}_wrist_{name}",
                "type": "slide",
                "axis": axis,
                "limited": "false",
                "range": _fmt(joint["range"]),  # +/-1e10, unlimited (limited=false)
                "armature": f"{joint['armature'][0]:.9g}",
                "frictionloss": f"{joint['frictionloss'][0]:.9g}",
                "actuatorfrcrange": _fmt(joint["actfrcrange"]),
            },
        )
    bodies["pz"].append(palm)
    bodies["py"].append(bodies["pz"])
    bodies["px"].append(bodies["py"])

    for body in bodies["px"].iter("body"):
        # flash_chord sets body gravcomp=1.0 on every body (SharpaHands
        # ._apply_control_config); MJCF has no <default> for <body>.
        body.set("gravcomp", "1")
        # Backfill the merged-body inertial verbatim. MuJoCo's URDF importer
        # attaches the fixed child links' geoms (elastomer/fingertip) to the
        # parent body but drops their mass; newton merges the child inertials
        # into the parent (visible on the 10 *_DP bodies). Stamping every body
        # from the extract keeps this exact by construction.
        spec = PARAMS["bodies"].get(body.get("name"))
        if spec is not None:
            for stale in body.findall("inertial"):
                body.remove(stale)
            inertial = ET.SubElement(
                body,
                "inertial",
                {
                    "mass": f"{spec['mass']:.9g}",
                    "pos": _fmt(spec["ipos"]),
                    "diaginertia": _fmt(spec["inertia"]),
                },
            )
            quat = spec["iquat"]
            if any(abs(v) > 1e-12 for v in quat[1:]):
                inertial.set("quat", _fmt(quat))

    for joint in bodies["px"].iter("joint"):
        joint_name = joint.get("name")
        if joint_name is None:
            continue
        spec = PARAMS["joints"][joint_name]
        if joint.get("type") == "ball":
            # kd=6 dof damping / armature / frictionloss already come from the
            # single-hand file; flash_chord leaves the ball actgravcomp=0.
            joint.set("actuatorgravcomp", "false")
            continue
        # Finger hinges: flash_chord overrides URDF effort limits to +/-10 and
        # sets armature=0.01 (SharpaHands joint_overrides).
        joint.set("armature", f"{spec['armature'][0]:.9g}")
        joint.set("actuatorfrcrange", _fmt(spec["actfrcrange"]))
        # Newton's scalar JointDofConfig carries direct limit stiffness/damping;
        # SolverMuJoCo writes them to MuJoCo as negative solreflimit values.
        # Ball joints retain MuJoCo's default limit solver parameters.
        joint.set("solreflimit", _fmt(spec["solref"]))
        joint.set("solimplimit", _fmt(spec["solimp"]))
    return bodies["px"]


def stamp_geom_filters(
    root: ET.Element, model: mujoco.MjModel, side: str | None = None
) -> None:
    """Stamp per-geom contype/conaffinity from the flash_chord collision model.

    Matching happens on the *compiled* model: MuJoCo recenters mesh geoms at
    compile time, so XML-level geom pos/quat (often omitted) differ from the
    dumped ``mj_model.geom_pos/quat``. XML geom elements within a body pair
    with the body's model geoms in document order.

    Only collision geoms participate in matching (visual geoms carry explicit
    ``contype="0"`` from the URDF conversion). A visual geom can share the
    exact compiled transform of a collision geom on the same link, so it must
    never consume a collision entry. Leftover entries on a body (near-spherical
    fingertip meshes whose principal-axis frames are degenerate, so the two
    pipelines recenter them into rotated frames) are assigned to the remaining
    unmatched collision geoms of that body — safe because every multi-geom
    body in flash_chord has uniform (contype, conaffinity) across its geoms
    (verified at extraction time).

    ``side`` scopes the expected GT entries to one hand (``"left"``/``"right"``)
    for single-hand entity sources; ``None`` expects both hands (the dual
    composer).
    """
    worldbody = root.find("worldbody")
    assert worldbody is not None
    unmatched = {
        body: list(entries)
        for body, entries in PARAMS["geoms"].items()
        if side is None or body.startswith(f"{side}_")
    }
    for body in worldbody.iter("body"):
        body_name = body.get("name")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        model_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        xml_geoms = body.findall("geom")
        if len(model_geoms) != len(xml_geoms):
            raise ValueError(
                f"geom count mismatch on {body_name}: {len(model_geoms)} compiled vs {len(xml_geoms)} XML"
            )
        leftover_candidates: list[ET.Element] = []
        for geom_id, geom in zip(model_geoms, xml_geoms, strict=True):
            if geom.get("contype") == "0":  # visual geom: pinned, never matched
                geom.set("conaffinity", "0")
                continue
            pos = model.geom_pos[geom_id]
            quat = model.geom_quat[geom_id]
            entries = unmatched.get(body_name, [])
            match = next(
                (
                    e
                    for e in entries
                    if all(
                        abs(a - b) < 1e-5 for a, b in zip(e["pos"], pos, strict=True)
                    )
                    and abs(sum(a * b for a, b in zip(e["quat"], quat, strict=True)))
                    > 1.0 - 1e-5
                ),
                None,
            )
            if match is None:
                leftover_candidates.append(geom)
                continue
            entries.remove(match)
            geom.set("contype", str(match["contype"]))
            geom.set("conaffinity", str(match["conaffinity"]))
        entries = unmatched.get(body_name, [])
        if entries or leftover_candidates:
            if len(entries) != len(leftover_candidates):
                raise ValueError(
                    f"{body_name}: {len(entries)} unmatched flash_chord geoms vs "
                    f"{len(leftover_candidates)} unmatched collision geoms"
                )
            bits = {
                (e["contype"], e["conaffinity"])
                for e in PARAMS["geoms"].get(body_name, [])
            }
            if len(bits) != 1:
                raise ValueError(
                    f"{body_name}: non-uniform contype/conaffinity, cannot fallback-assign"
                )
            contype, conaffinity = next(iter(bits))
            for geom in leftover_candidates:
                geom.set("contype", str(contype))
                geom.set("conaffinity", str(conaffinity))
            entries.clear()
    leftovers = {body: entries for body, entries in unmatched.items() if entries}
    if leftovers:
        raise ValueError(f"unmatched flash_chord collision geoms: {leftovers}")


def _finger_joint_names(model: mujoco.MjModel, side: str) -> list[str]:
    names = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if (
            model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            and name is not None
            and name.startswith(f"{side}_")
        ):
            names.append(name)
    return names


def compose() -> Path:
    root = ET.Element("mujoco", {"model": "sharpa_dual_floating"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    # SolverMuJoCo settings from flash_chord SimConfig (runtime/sim.py).
    ET.SubElement(
        root,
        "option",
        {
            "timestep": "0.01",  # sharpa_flash_sac sim group (sim/rl.yaml): 20 Hz control x 5 substeps
            "solver": "Newton",
            "integrator": "implicitfast",
            "cone": "pyramidal",
            "impratio": "20",
            "iterations": "100",
            "ls_iterations": "50",
        },
    )

    contact = PARAMS["contact"]
    default = ET.SubElement(root, "default")
    default.append(
        ET.Comment(
            " geom contact params extracted verbatim from flash_chord's "
            "compiled mj_model (uniform across all 52 collision geoms); "
            "visual geoms carry them too but are contype=0 "
        )
    )
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
    # NB: <body> is not a valid <default> child in MJCF; gravcomp is stamped
    # per body in mount_with_hand instead. The ball joints override
    # actuatorgravcomp=false explicitly (flash_chord actgravcomp=0 there).
    ET.SubElement(default, "joint", {"actuatorgravcomp": "true"})

    contact_elem = ET.SubElement(root, "contact")
    contact_elem.append(
        ET.Comment(
            f" {len(PARAMS['excludes'])} pairs extracted from flash_chord's "
            "compiled mj_model exclude_signature (supersedes the 70-pair "
            "sharpa_collision.py _EXCLUDED_CONTACTS, which is a subset); "
            "includes cross-hand pairs — the two hands never collide "
        )
    )
    for body_a, body_b in PARAMS["excludes"]:
        ET.SubElement(contact_elem, "exclude", {"body1": body_a, "body2": body_b})

    worldbody = ET.SubElement(root, "worldbody")
    worldbody.append(
        ET.Comment(
            " GAP-SENSOR: fingertip x object contact-force sensors are declared "
            "in the task scene layer (object fragment is dataset-coupled) "
        )
    )
    assets: list[ET.Element] = []
    for side in ("left", "right"):
        worldbody.append(mount_with_hand(side, assets))
    asset = ET.Element("asset")
    for mesh in assets:
        asset.append(mesh)
    root.insert(1, asset)

    # First pass: compile to match collision geoms, enumerate finger joints
    # and build the keyframe (contype/conaffinity stamping does not change the
    # kinematic layout, so one compile serves all three).
    tmp_path = OUT_PATH.with_suffix(".tmp.xml")
    ET.ElementTree(root).write(tmp_path, xml_declaration=True, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(tmp_path))

    stamp_geom_filters(root, model)

    # Actuation: one exact position+velocity actuator pair per driven scalar
    # joint. Effort limits live at joint level (actuatorfrcrange), as in
    # flash_chord's mj_model (actuator forcerange is unlimited there).
    # GAP-WRIST-EFFORT: no ball actuators by design (see module docstring).
    actuator = ET.SubElement(root, "actuator")
    for side in ("left", "right"):
        for name, _, _ in _SLIDES:
            gains = PARAMS["actuators"][f"{side}_wrist_{name}"]
            append_position_velocity_actuators(
                actuator,
                name=f"{side}_wrist_{name}",
                joint=f"{side}_wrist_{name}",
                kp=float(gains["kp"]),
                kd=float(gains["kd"]),
            )
        for joint_name in _finger_joint_names(model, side):
            gains = PARAMS["actuators"][joint_name]
            append_position_velocity_actuators(
                actuator,
                name=joint_name,
                joint=joint_name,
                kp=float(gains["kp"]),
                kd=float(gains["kd"]),
            )

    # Keyframe: flash_chord seeds wrist_y=±0.25, wrist_z=0.5, all else zero.
    qpos = [0.0] * model.nq
    for side in ("left", "right"):
        for name, value in (
            (f"{side}_wrist_y", WRIST_Y_DEFAULT[side]),
            (f"{side}_wrist_z", WRIST_Z_DEFAULT),
        ):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos[model.jnt_qposadr[joint_id]] = value
        ball_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_rotvec"
        )
        qpos[model.jnt_qposadr[ball_id]] = 1.0  # unit quaternion w
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "home",
            "qpos": " ".join(f"{v:g}" for v in qpos),
        },
    )

    ET.ElementTree(root).write(OUT_PATH, xml_declaration=True, encoding="unicode")
    tmp_path.unlink(missing_ok=True)
    return OUT_PATH


def _validate(path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(path))
    hinge = sum(
        1 for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    )
    ball = sum(
        1 for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL
    )
    slide = sum(
        1 for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
    )
    print(
        f"dual: nbody={model.nbody} njnt={model.njnt} "
        f"(hinge={hinge} ball={ball} slide={slide}) nu={model.nu} "
        f"nexclude={model.nexclude} nq={model.nq} nv={model.nv} "
        f"total_mass={model.body_subtreemass[0]:.4f}kg"
    )
    if (hinge, ball, slide) != (44, 2, 6):
        raise ValueError(
            f"expected 44 hinge + 2 ball + 6 slide, got {hinge}+{ball}+{slide}"
        )
    if model.nu != 2 * (44 + 6):
        raise ValueError(f"expected 100 actuators, got {model.nu}")
    if model.nexclude != len(PARAMS["excludes"]):
        raise ValueError(
            f"expected {len(PARAMS['excludes'])} excludes, got {model.nexclude}"
        )


def main() -> None:
    path = compose()
    _validate(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
