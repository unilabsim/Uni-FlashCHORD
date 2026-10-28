"""ChordReferenceLoader unit tests on synthetic ManoSharpaData parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chord_unilab.data import load_chord_reference

SIDES = ("left", "right")
T_FPS = 30.0


def _write_synthetic_parquet(
    root: Path, *, cells_override: dict | None = None, rows: int = 1
) -> Path:
    """Write a minimal valid ManoSharpaData parquet under Hive partitions."""
    num_frames = 4
    times = np.arange(num_frames)[:, None]
    cells: dict[str, object] = {
        "fps": np.float32(T_FPS),
        "object_name": "box",
        "object_body_names": ["box"],
        "object_mesh_paths": ["box.obj"],
        "object_urdf_paths": ["box.urdf"],
        "object_mesh_radius": np.array([0.1], dtype=np.float32),
        "object_articulation": np.zeros((num_frames, 0), dtype=np.float32),
        "object_body_position": np.concatenate([times, times**2, -times], axis=1)[
            :, None, :
        ].astype(np.float32),
        "object_body_wxyz": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_frames, 1, 1)
        ),
    }
    for side in SIDES:
        cells[f"robot_{side}_wrist_position"] = np.concatenate(
            [times, 2 * times, 3 * times], axis=1
        ).astype(np.float32)
        cells[f"robot_{side}_wrist_wxyz"] = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_frames, 1)
        )
        cells[f"robot_{side}_finger_joints"] = np.tile(
            np.arange(22, dtype=np.float32), (num_frames, 1)
        )
        cells[f"{side}_robot_finger_joint_names"] = [
            f"{side}_joint_{i}" for i in range(22)
        ]
        contacts = np.zeros((num_frames, 16, 3), dtype=np.float32)
        normals = np.zeros((num_frames, 16, 3), dtype=np.float32)
        part_ids = np.zeros((num_frames, 16), dtype=np.int32)
        # frames 0-1: slot 0 active on body 1; frames 2-3: no contact
        contacts[:2, 0] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        normals[:2, 0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        part_ids[:2, 0] = 1
        cells[f"mano_{side}_object_contact_positions"] = contacts
        cells[f"mano_{side}_object_contact_normals"] = normals
        cells[f"mano_{side}_object_contact_part_ids"] = part_ids
        cells[f"robot_{side}_frames"] = np.zeros((num_frames, 3, 7), dtype=np.float32)
        cells[f"{side}_robot_frame_names"] = [f"{side}_frame_{i}" for i in range(3)]
    if cells_override:
        cells.update(cells_override)

    out_dir = root / "sequence_id=test_seq" / "robot_name=test_bot"
    out_dir.mkdir(parents=True)

    def as_cell(value: object) -> object:
        # pyarrow infers nested list columns from Python lists, not ndarrays
        return value.tolist() if isinstance(value, np.ndarray) else value

    arrays = {name: [as_cell(value)] * rows for name, value in cells.items()}
    pq.write_table(pa.table(arrays), out_dir / "data.parquet")
    return out_dir


def test_passthrough_without_resampling(tmp_path: Path) -> None:
    ref = load_chord_reference(_write_synthetic_parquet(tmp_path))
    assert ref.num_frames == 4
    assert ref.fps == T_FPS
    assert not ref.metadata.is_resampled
    assert ref.metadata.sequence_id == "test_seq"
    assert ref.metadata.robot_name == "test_bot"
    for side in SIDES:
        np.testing.assert_array_equal(
            ref.wrist_pos_w(side)[:, 0], np.arange(4, dtype=np.float32)
        )
        assert ref.finger_joint_pos(side).shape == (4, 22)
        np.testing.assert_array_equal(ref.contact_active(side), [1.0, 1.0, 0.0, 0.0])
    assert ref.object_body_pos_w().shape == (4, 1, 3)
    assert ref.object_articulation().shape == (4, 0)
    assert ref.object_assets()[0].urdf_path == "box.urdf"


def test_resample_half_speed_lerp_nearest(tmp_path: Path) -> None:
    # 4 frames @30fps -> duration 0.1s; 0.5x at 30fps control -> 6 samples
    ref = load_chord_reference(
        _write_synthetic_parquet(tmp_path), control_fps=30.0, motion_speed=0.5
    )
    assert ref.num_frames == 6
    assert ref.metadata.is_resampled
    target_times = np.linspace(0.0, 0.1, 6)
    np.testing.assert_allclose(
        ref.wrist_pos_w("left")[:, 0], target_times * T_FPS, rtol=1e-6
    )
    # nearest: contact part_ids stay binary 1/0 across the active boundary
    pids = ref.contact_part_ids("left")[:, 0]
    assert set(np.unique(pids)) <= {0, 1}
    assert pids[0] == 1 and pids[-1] == 0


def test_slerp_midpoint(tmp_path: Path) -> None:
    # wrist rotates 0 -> pi about z across 0.1s; resampled midpoint must be pi/2
    num_frames = 4
    angles = np.linspace(0.0, np.pi, num_frames)
    quats = np.stack(
        [np.cos(angles / 2), np.zeros(4), np.zeros(4), np.sin(angles / 2)], axis=1
    ).astype(np.float32)
    override = {f"robot_{side}_wrist_wxyz": quats for side in SIDES}
    ref = load_chord_reference(
        _write_synthetic_parquet(tmp_path, cells_override=override),
        control_fps=30.0,
        motion_speed=0.5,
    )
    # 6 target samples at t = 0, 0.02, ..., 0.1; source angle(t) = pi * t / 0.1
    for index, t in ((2, 0.04), (3, 0.06)):
        angle = np.pi * t / 0.1
        np.testing.assert_allclose(
            ref.wrist_quat_w("left")[index],
            [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)],
            atol=1e-5,
        )


def test_motion_speed_requires_control_fps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="control_fps is required"):
        load_chord_reference(_write_synthetic_parquet(tmp_path), motion_speed=0.5)


def test_rejects_nonunit_wrist_quaternion(tmp_path: Path) -> None:
    override = {
        "robot_left_wrist_wxyz": np.tile(
            np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32), (4, 1)
        )
    }
    with pytest.raises(ValueError, match="unit length"):
        load_chord_reference(
            _write_synthetic_parquet(tmp_path, cells_override=override)
        )


def test_rejects_contact_part_id_out_of_range(tmp_path: Path) -> None:
    part_ids = np.zeros((4, 16), dtype=np.int32)
    part_ids[0, 0] = 5  # only 1 object body
    with pytest.raises(ValueError, match="integers in"):
        load_chord_reference(
            _write_synthetic_parquet(
                tmp_path, cells_override={"mano_left_object_contact_part_ids": part_ids}
            )
        )


def test_rejects_multi_row_parquet(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one row"):
        load_chord_reference(_write_synthetic_parquet(tmp_path, rows=2))


def test_partition_values_fill_metadata(tmp_path: Path) -> None:
    ref = load_chord_reference(_write_synthetic_parquet(tmp_path))
    assert ref.metadata.sequence_id == "test_seq"
    assert ref.metadata.robot_name == "test_bot"


def test_frame_window_slices_all_time_axes(tmp_path: Path) -> None:
    ref = load_chord_reference(_write_synthetic_parquet(tmp_path))
    assert ref.frame_window() is ref
    windowed = ref.frame_window(1, 3)
    assert windowed.num_frames == 2
    np.testing.assert_array_equal(
        windowed.wrist_pos_w("left"), ref.wrist_pos_w("left")[1:3]
    )
    np.testing.assert_array_equal(
        windowed.object_body_pos_w(), ref.object_body_pos_w()[1:3]
    )
    np.testing.assert_array_equal(
        windowed.contact_part_ids("right"), ref.contact_part_ids("right")[1:3]
    )
    with pytest.raises(ValueError, match="invalid reference frame window"):
        ref.frame_window(3, 3)
