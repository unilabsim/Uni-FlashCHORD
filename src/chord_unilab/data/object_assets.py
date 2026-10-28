"""Object asset path resolution, ported from flash_chord's assets registry.

Faithful port of the ``resolve_object_asset_specs`` subtree of flash_chord
``src/flash_chord/assets/registry.py``. Parquet-stored mesh/URDF paths come
from the data-generation environment; at load time they are re-rooted by
anchoring on an asset-root boundary.

Declared deviation (GAP-OBJECT-ASSET-ROOT): flash_chord falls back to its
vendored ``ASSETS_DIR`` when the reference-relative anchor misses. This repo
uses the nearest ``assets`` ancestor of the reference parquet itself, including
the versioned benchmark closures under ``data/assets``. For parquets inside a
standard V2D ``assets/human_motion_data/...`` tree both rules resolve
identically.
"""

from __future__ import annotations

from pathlib import Path

from chord_unilab.data.reference import (
    ObjectArticulationSpec,
    ObjectAssetSpec,
    ObjectBodySpec,
    ObjectJointDriveSpec,
    ObjectJointPhysicsSpec,
)

_ASSETS_MARKER = "/assets/"
_OBJECT_ASSETS_MARKER = "/data/object_assets/"


def resolve_asset_path(path: str, *, reference_path: str | Path | None = None) -> str:
    """Re-root a parquet-stored asset path onto the reference's asset closure.

    Anchors on the final ``assets/`` boundary of the reference parquet tree.
    Paths with no known marker, or whose anchored candidate does not exist,
    are returned unchanged (GAP-OBJECT-ASSET-ROOT: no vendored fallback).
    """
    if Path(path).is_file():
        return path
    reference_assets = reference_assets_root(reference_path)
    for marker in (_ASSETS_MARKER, _OBJECT_ASSETS_MARKER):
        idx = path.rfind(marker)
        if idx == -1 or reference_assets is None:
            continue
        candidate = reference_assets / path[idx + len(marker) :]
        if candidate.is_file():
            return str(candidate)
    return path


def reference_assets_root(reference_path: str | Path | None) -> Path | None:
    """Nearest ``assets`` ancestor of the reference parquet, or None."""
    if reference_path is None:
        return None
    resolved = Path(reference_path).expanduser().resolve()
    parts = resolved.parts
    indices = [index for index, part in enumerate(parts) if part == "assets"]
    return None if not indices else Path(*parts[: indices[-1] + 1])


def _v2d_articulated_urdf(
    *,
    source_dataset: str,
    object_name: str,
    mesh_paths: list[str],
    reference_path: str | Path | None,
) -> Path | None:
    """Resolve V2D's ``object_assets/{meshes,urdfs}`` articulated-object layout."""
    if not mesh_paths:
        return None
    parts = Path(mesh_paths[0]).parts
    try:
        marker = parts.index("object_assets")
    except ValueError:
        return None
    tail = parts[marker + 1 :]
    if (
        len(tail) < 4
        or tail[0] != "meshes"
        or tail[1] != source_dataset
        or tail[2] != object_name
    ):
        return None

    relative = Path("object_assets") / "urdfs" / source_dataset / f"{object_name}.urdf"
    candidates = []
    reference_assets = reference_assets_root(reference_path)
    if reference_assets is not None:
        candidates.append(
            reference_assets / "human_motion_data" / source_dataset / relative
        )
    if marker:
        candidates.append(Path(*parts[:marker]) / relative)
    elif Path(mesh_paths[0]).is_absolute():
        candidates.append(Path("/") / relative)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def resolve_object_asset_specs(
    *,
    source_dataset: str | None,
    object_name: str,
    body_names: list[str],
    mesh_paths: list[str],
    urdf_paths: list[str],
    num_articulations: int,
    reference_path: str | Path | None = None,
) -> tuple[ObjectAssetSpec, ...]:
    """Adapt stored asset metadata into declarative object import specifications."""
    if not body_names or len(set(body_names)) != len(body_names):
        raise ValueError("object reference body names must be nonempty and unique")
    if num_articulations < 0:
        raise ValueError("object articulation count must be nonnegative")

    if urdf_paths:
        if len(urdf_paths) != len(body_names):
            raise ValueError(
                f"rigid object URDF count {len(urdf_paths)} does not match "
                f"{len(body_names)} reference bodies"
            )
        if num_articulations:
            raise ValueError(
                "independent rigid object assets cannot bind articulation columns"
            )
        return tuple(
            ObjectAssetSpec(
                name=body_name,
                urdf_path=resolve_asset_path(urdf_path, reference_path=reference_path),
                bodies=(
                    ObjectBodySpec(reference_name=body_name, simulation_name=None),
                ),
                root_reference_name=body_name,
            )
            for body_name, urdf_path in zip(body_names, urdf_paths, strict=True)
        )

    if source_dataset not in ("arctic", "synthbox"):
        raise ValueError(
            "references without per-body URDFs require an explicit supported "
            f"asset adapter; source_dataset={source_dataset!r}"
        )
    articulated_urdf = _v2d_articulated_urdf(
        source_dataset=source_dataset,
        object_name=object_name,
        mesh_paths=mesh_paths,
        reference_path=reference_path,
    )
    if articulated_urdf is None:
        raise ValueError(
            f"no {source_dataset} articulated URDF exists for object {object_name!r}; "
            "expected object_assets/urdfs/<dataset>/<object>.urdf beside the "
            "reference assets"
        )
    if "bottom" not in body_names:
        raise ValueError(
            f"{source_dataset} articulated object reference must contain the root body 'bottom'"
        )
    if num_articulations != 1:
        raise ValueError(
            f"{source_dataset} articulated object {object_name!r} requires exactly "
            f"one articulation column, got {num_articulations}"
        )
    return (
        ObjectAssetSpec(
            name=object_name,
            urdf_path=str(articulated_urdf),
            bodies=tuple(
                ObjectBodySpec(reference_name=body_name, simulation_name=body_name)
                for body_name in body_names
            ),
            root_reference_name="bottom",
            articulations=(
                ObjectArticulationSpec(
                    simulation_joint_name="rotation",
                    reference_index=0,
                    physics=ObjectJointPhysicsSpec(armature=0.01, friction=0.1),
                    drive=ObjectJointDriveSpec(kp=50.0, kd=2.0, effort_limit=50.0),
                ),
            ),
        ),
    )
