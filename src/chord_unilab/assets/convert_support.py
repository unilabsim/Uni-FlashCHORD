"""Convert a reference-resolved support USDA into one kinematic rigid component."""

from __future__ import annotations

import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from chord_unilab.assets.convert_object import cache_root
from chord_unilab.assets.physics import (
    DEFAULT_SCENE_PHYSICS,
    ScenePhysics,
    contact_mjcf,
)
from chord_unilab.data import load_chord_reference
from chord_unilab.data.support_assets import resolve_support_asset

FRAGMENT_SCHEMA = "chord-support-fragment-v3"


@dataclass(frozen=True)
class SupportShape:
    name: str
    kind: Literal["cylinder", "box"]
    position: tuple[float, float, float]
    size: tuple[float, ...]


@dataclass(frozen=True)
class SupportAsset:
    path: Path
    sha256: str
    shapes: tuple[SupportShape, ...]


def _vector(
    value, *, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    if value is None:
        return default
    result = tuple(float(component) for component in value)
    if len(result) != 3:
        raise ValueError(f"expected a three-vector, got {result}")
    return result


def load_support_asset(usda_path: str | Path) -> SupportAsset:
    """Read Cylinder/Cube prims with the GT translate and scale semantics."""
    from pxr import Usd  # type: ignore[import-untyped]

    path = Path(usda_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"support surface does not exist: {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"failed to open support surface: {path}")
    shapes: list[SupportShape] = []
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name not in ("Cylinder", "Cube"):
            continue
        label = str(prim.GetPath())
        position = _vector(
            prim.GetAttribute("xformOp:translate").Get(), default=(0.0, 0.0, 0.0)
        )
        if not all(math.isfinite(value) for value in position):
            raise ValueError(f"support shape {label} has a non-finite translation")
        dimensions: tuple[float, ...]
        if type_name == "Cylinder":
            radius_value = prim.GetAttribute("radius").Get()
            height_value = prim.GetAttribute("height").Get()
            if radius_value is None or height_value is None:
                raise ValueError(f"support cylinder {label} requires radius and height")
            radius, height = float(radius_value), float(height_value)
            dimensions = (radius, height)
            shape = SupportShape(
                str(prim.GetName()), "cylinder", position, (radius, 0.5 * height)
            )
        else:
            size_value = prim.GetAttribute("size").Get()
            edge = 2.0 if size_value is None else float(size_value)
            scale = _vector(
                prim.GetAttribute("xformOp:scale").Get(), default=(1.0, 1.0, 1.0)
            )
            if not all(math.isfinite(value) and value != 0.0 for value in scale):
                raise ValueError(
                    f"support shape {label} must have a finite nonzero scale"
                )
            dimensions = tuple(edge * abs(value) for value in scale)
            shape = SupportShape(
                str(prim.GetName()),
                "box",
                position,
                tuple(0.5 * value for value in dimensions),
            )
        if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
            raise ValueError(
                f"support shape {label} must have positive finite dimensions"
            )
        shapes.append(shape)
    if not shapes:
        raise ValueError(f"support surface contains no Cylinder or Cube shapes: {path}")
    return SupportAsset(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), tuple(shapes)
    )


def cache_key(
    asset: SupportAsset, physics: ScenePhysics = DEFAULT_SCENE_PHYSICS
) -> str:
    digest = hashlib.sha1()
    digest.update(FRAGMENT_SCHEMA.encode())
    digest.update(asset.sha256.encode())
    digest.update(repr(physics).encode())
    return digest.hexdigest()[:16]


def _default_support_path() -> Path:
    from chord_unilab.assets.convert_object import (
        GT_CONTROL_FPS,
        GT_MOTION_SPEED,
        default_parquet,
    )

    parquet = default_parquet()
    if parquet is None:
        raise ValueError("support conversion requires a USDA path or reference parquet")
    reference = load_chord_reference(
        parquet, control_fps=GT_CONTROL_FPS, motion_speed=GT_MOTION_SPEED
    )
    support = resolve_support_asset(reference)
    if support is None:
        raise FileNotFoundError(
            f"no support asset resolved for {reference.metadata.source_path}"
        )
    return support


def fragment_path(
    usda_path: str | Path | None = None,
    *,
    cache_dir: Path | None = None,
    physics: ScenePhysics = DEFAULT_SCENE_PHYSICS,
) -> Path:
    asset = load_support_asset(usda_path or _default_support_path())
    return (
        (cache_dir or cache_root())
        / "support"
        / cache_key(asset, physics)
        / "support.xml"
    )


def _build(asset: SupportAsset, physics: ScenePhysics) -> ET.Element:
    root = ET.Element("mujoco", {"model": "support"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", physics.solver.as_mjcf())
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contact_mjcf(physics.contact))
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "support"})
    ET.SubElement(body, "freejoint", {"name": "support_free_joint"})
    group = physics.collision.support_group
    for index, shape in enumerate(asset.shapes):
        geometry = {
            "type": shape.kind,
            "size": " ".join(f"{value:.9g}" for value in shape.size),
            "pos": " ".join(f"{value:.9g}" for value in shape.position),
        }
        ET.SubElement(
            body,
            "geom",
            geometry
            | {
                "name": f"support_{index}_{shape.name}",
                "density": "1000",
                "contype": str(group),
                "conaffinity": str(physics.collision.conaffinity(group)),
            },
        )
        ET.SubElement(
            body,
            "geom",
            geometry
            | {
                "name": f"support_{index}_{shape.name}_visual",
                "density": "0",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    return root


def _validate(xml_path: Path, asset: SupportAsset) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    expected = (2, 1, 2 * len(asset.shapes), 0, 7, 6)
    actual = (model.nbody, model.njnt, model.ngeom, model.nu, model.nq, model.nv)
    if actual != expected:
        raise ValueError(f"support topology {actual} != {expected}")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "support")
    if (
        body_id < 0
        or not np.isfinite(model.body_mass[body_id])
        or model.body_mass[body_id] <= 0.0
    ):
        raise ValueError(
            "support must have positive finite mass derived from its shapes"
        )
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "support_free_joint") < 0:
        raise ValueError("support_free_joint missing after build")


def convert_support(
    usda_path: str | Path | None = None,
    *,
    cache_dir: Path | None = None,
    physics: ScenePhysics = DEFAULT_SCENE_PHYSICS,
) -> Path:
    """Build the USDA-driven support fragment into the external cache."""
    asset = load_support_asset(usda_path or _default_support_path())
    out_path = fragment_path(asset.path, cache_dir=cache_dir, physics=physics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = _build(asset, physics)
    tmp_path = out_path.with_suffix(".tmp.xml")
    ET.ElementTree(root).write(tmp_path, xml_declaration=True, encoding="unicode")
    _validate(tmp_path, asset)
    os.replace(tmp_path, out_path)
    (out_path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "schema": FRAGMENT_SCHEMA,
                "cache_key": cache_key(asset, physics),
                "usda_path": str(asset.path),
                "usda_sha256": asset.sha256,
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
    import sys

    convert_support(Path(sys.argv[1]) if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()
