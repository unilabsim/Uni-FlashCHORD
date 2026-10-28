"""Build the fixed floor entity from explicit scene physics configuration."""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import mujoco

from chord_unilab.assets.convert_object import cache_root
from chord_unilab.assets.physics import (
    DEFAULT_SCENE_PHYSICS,
    ScenePhysics,
    contact_mjcf,
)

FRAGMENT_SCHEMA = "chord-floor-fragment-v2"
FLOOR_HALF_EXTENT = 5.0
FLOOR_HALF_DEPTH = 0.1


def cache_key(physics: ScenePhysics = DEFAULT_SCENE_PHYSICS) -> str:
    digest = hashlib.sha1()
    digest.update(FRAGMENT_SCHEMA.encode())
    digest.update(repr(physics).encode())
    return digest.hexdigest()[:16]


def fragment_path(
    *,
    cache_dir: Path | None = None,
    physics: ScenePhysics = DEFAULT_SCENE_PHYSICS,
) -> Path:
    return (cache_dir or cache_root()) / "floor" / cache_key(physics) / "floor.xml"


def _build(physics: ScenePhysics) -> ET.Element:
    root = ET.Element("mujoco", {"model": "floor"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", physics.solver.as_mjcf())
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contact_mjcf(physics.ground_contact))
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "floor"})
    ET.SubElement(
        body, "inertial", {"pos": "0 0 0", "mass": "0", "diaginertia": "0 0 0"}
    )
    group = physics.collision.ground_group
    ET.SubElement(
        body,
        "geom",
        {
            "name": "floor_geom",
            "type": "box",
            "size": f"{FLOOR_HALF_EXTENT:.9g} {FLOOR_HALF_EXTENT:.9g} {FLOOR_HALF_DEPTH:.9g}",
            "pos": f"0 0 {-FLOOR_HALF_DEPTH:.9g}",
            "contype": str(group),
            "conaffinity": str(physics.collision.conaffinity(group)),
        },
    )
    return root


def _validate(xml_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    actual = (model.nbody, model.njnt, model.ngeom, model.nu)
    if actual != (2, 0, 1, 0):
        raise ValueError(f"expected floor topology (2, 0, 1, 0), got {actual}")
    top = float(model.geom_pos[0][2] + model.geom_size[0][2])
    if abs(top) > 1e-9:
        raise ValueError(f"floor top surface must be exactly z=0, got {top}")


def convert_floor(
    *,
    cache_dir: Path | None = None,
    physics: ScenePhysics = DEFAULT_SCENE_PHYSICS,
) -> Path:
    out_path = fragment_path(cache_dir=cache_dir, physics=physics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = _build(physics)
    tmp_path = out_path.with_suffix(".tmp.xml")
    ET.ElementTree(root).write(tmp_path, xml_declaration=True, encoding="unicode")
    _validate(tmp_path)
    os.replace(tmp_path, out_path)
    (out_path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "schema": FRAGMENT_SCHEMA,
                "cache_key": cache_key(physics),
                "physics": repr(physics),
                "generated_at": datetime.now(UTC).isoformat(),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    convert_floor()


if __name__ == "__main__":
    main()
