"""Resolve support USDA files from reference-local sequence metadata."""

from __future__ import annotations

from pathlib import Path

from chord_unilab.data.reference import ChordReference


def _path_partitions(path: Path, prefix: str) -> tuple[str, ...]:
    return tuple(
        part.name.split("=", 1)[1]
        for part in (path, *path.parents)
        if part.name.startswith(prefix)
    )


def _support_candidates(
    stage_dir: Path, sequence_id: str, robot_name: str | None
) -> tuple[Path, ...]:
    specific = (
        (stage_dir / f"{sequence_id}_{robot_name}_support.usda",) if robot_name else ()
    )
    return specific + (stage_dir / f"{sequence_id}_support.usda",)


def resolve_support_asset(reference: ChordReference) -> Path | None:
    """Resolve sequence-adjacent support, preferring the reference's robot variant."""
    metadata = reference.metadata
    source_path = Path(metadata.source_path).expanduser().resolve()
    sequence_dirs = tuple(
        parent
        for parent in (source_path, *source_path.parents)
        if parent.name.startswith("sequence_id=")
    )
    path_robots = _path_partitions(source_path, "robot_name=")
    robot_name = metadata.robot_name or (path_robots[0] if path_robots else None)
    for sequence_dir in sequence_dirs:
        sequence_id = sequence_dir.name.split("=", 1)[1]
        stage_dir = sequence_dir.parent.parent / "reconstructed_stage"
        for candidate in _support_candidates(stage_dir, sequence_id, robot_name):
            if candidate.is_file():
                return candidate

    parts = source_path.parts
    if (
        "human_motion_data" not in parts
        or metadata.source_dataset is None
        or metadata.sequence_id is None
    ):
        return None
    index = parts.index("human_motion_data")
    dataset_root = Path(*parts[: index + 1]) / metadata.source_dataset
    stage_dir = dataset_root / "reconstructed_stage"
    return next(
        (
            candidate
            for candidate in _support_candidates(
                stage_dir, metadata.sequence_id, robot_name
            )
            if candidate.is_file()
        ),
        None,
    )
