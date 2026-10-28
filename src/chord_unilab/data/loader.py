"""Load a retargeted ManoSharpa parquet into a :class:`ChordReference`.

Faithful port of flash_chord ``load_mano_sharpa``
(``src/flash_chord/data/mano_sharpa.py``) — the ground truth for fidelity
gate 2. The parquet is single-row with each cell a nested per-frame array.
Robot wrist/finger fields are populated only in retargeted (``_processed``)
parquets. Quaternions are wxyz; no coordinate transform is applied at load
(the data is assumed to already be in the Z-up metric world frame).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chord_unilab.data.object_assets import resolve_object_asset_specs
from chord_unilab.data.parquet import read_parquet_row
from chord_unilab.data.reference import (
    ChordReference,
    HandReference,
    ObjectReference,
    ReferenceMetadata,
)
from chord_unilab.data.resampling import (
    lerp as _lerp,
)
from chord_unilab.data.resampling import (
    nearest as _nearest,
)
from chord_unilab.data.resampling import (
    playback_times as _playback_times,
)
from chord_unilab.data.resampling import (
    slerp_tracks_wxyz,
)
from chord_unilab.data.resampling import (
    slerp_wxyz as _slerp_wxyz,
)

_SIDES = ("left", "right")

_QUAT_ATOL = 1.0e-4


def load_chord_reference(
    parquet_path: str | Path,
    control_fps: float | None = None,
    motion_speed: float = 1.0,
) -> ChordReference:
    """Load a ManoSharpa parquet, optionally resampled for control-rate playback.

    ``motion_speed`` scales source-time advancement per control step.
    Positions and joints use linear interpolation, quaternions use SLERP,
    and contact fields use nearest-neighbour sampling.
    """
    if control_fps is None and motion_speed != 1.0:
        raise ValueError("control_fps is required when motion_speed is not 1.0")
    row = read_parquet_row(parquet_path)
    cells = dict(row.cells)
    src_fps = float(cells["fps"])

    obj_pos = _arr(cells, "object_body_position")  # (T, B, 3)
    obj_quat = _arr(cells, "object_body_wxyz")  # (T, B, 4)
    if obj_pos is None or obj_quat is None:
        raise ValueError("parquet missing object_body_position / object_body_wxyz")
    if obj_pos.ndim != 3 or obj_pos.shape[-1] != 3 or not np.all(np.isfinite(obj_pos)):
        raise ValueError(
            f"object_body_position must be finite with shape (T, B, 3), got {obj_pos.shape}"
        )
    if obj_quat.shape != obj_pos.shape[:-1] + (4,) or not np.all(np.isfinite(obj_quat)):
        raise ValueError(
            f"object_body_wxyz must be finite with shape (T, B, 4), got {obj_quat.shape}"
        )
    quaternion_norm = np.linalg.norm(obj_quat, axis=-1, keepdims=True)
    if np.any(quaternion_norm <= np.finfo(obj_quat.dtype).eps):
        raise ValueError("object_body_wxyz contains a zero quaternion")
    if not np.allclose(quaternion_norm, 1.0, atol=_QUAT_ATOL, rtol=0.0):
        raise ValueError("object_body_wxyz quaternions must be unit length")
    obj_quat = np.ascontiguousarray(obj_quat / quaternion_norm)
    num_frames = obj_pos.shape[0]
    num_object_bodies = obj_pos.shape[1]
    object_name = str(cells.get("object_name") or "")
    object_body_names = [str(value) for value in (cells.get("object_body_names") or [])]
    if (
        not object_name
        or len(object_body_names) != num_object_bodies
        or len(set(object_body_names)) != num_object_bodies
        or any(not name for name in object_body_names)
    ):
        raise ValueError(
            "object names must be nonempty, unique, and match the reference body axis"
        )
    object_mesh_paths = [str(value) for value in (cells.get("object_mesh_paths") or [])]
    if len(object_mesh_paths) != num_object_bodies or any(
        not p for p in object_mesh_paths
    ):
        raise ValueError(
            "object mesh paths must be nonempty and match the reference body axis"
        )
    object_urdf_paths = [str(value) for value in (cells.get("object_urdf_paths") or [])]
    object_mesh_radius = np.asarray(
        cells.get("object_mesh_radius") or [], dtype=np.float64
    )
    if (
        object_mesh_radius.shape != (num_object_bodies,)
        or not np.all(np.isfinite(object_mesh_radius))
        or np.any(object_mesh_radius <= 0.0)
    ):
        raise ValueError(
            "object mesh radii must be positive, finite, and match the body axis"
        )
    art = _arr(cells, "object_articulation")
    art = np.zeros((num_frames, 0)) if art is None else art.reshape(num_frames, -1)
    if not np.all(np.isfinite(art)):
        raise ValueError("object articulation trajectory must be finite")
    if object_urdf_paths and art.shape[1]:
        if art.shape[1] != 1 or not np.all(art == 0.0):
            raise ValueError(
                "ManoSharpaData references with independent rigid-object URDFs may "
                "contain only one all-zero articulation placeholder column"
            )
        art = np.zeros((num_frames, 0), dtype=art.dtype)
    source_dataset = _optional_string(
        cells.get("source_dataset")
    ) or _dataset_from_path(row.path)
    object_assets = resolve_object_asset_specs(
        source_dataset=source_dataset,
        object_name=object_name,
        body_names=object_body_names,
        mesh_paths=object_mesh_paths,
        urdf_paths=object_urdf_paths,
        num_articulations=art.shape[1],
        reference_path=row.path,
    )

    t_src, t_tgt = (
        _playback_times(num_frames, src_fps, control_fps, motion_speed)
        if control_fps is not None
        else (None, None)
    )
    resample = control_fps is not None and (
        control_fps != src_fps or motion_speed != 1.0
    )

    def lerp(a: np.ndarray) -> np.ndarray:
        if not resample:
            return a
        assert t_src is not None and t_tgt is not None
        return _lerp(a, t_src, t_tgt)

    def slerp_track(q: np.ndarray) -> np.ndarray:  # q: (T, 4)
        if not resample:
            return q
        assert t_src is not None and t_tgt is not None
        return _slerp_wxyz(q, t_src, t_tgt)

    def slerp_bodies(q: np.ndarray) -> np.ndarray:  # q: (T, B, 4)
        if not resample:
            return q
        assert t_src is not None and t_tgt is not None
        return slerp_tracks_wxyz(q, t_src, t_tgt)

    def near(a: np.ndarray) -> np.ndarray:
        if not resample:
            return a
        assert t_src is not None and t_tgt is not None
        return _nearest(a, t_src, t_tgt)

    hands: dict[str, HandReference] = {}
    for side in _SIDES:
        wp_ = _arr(cells, f"robot_{side}_wrist_position")
        wq_ = _arr(cells, f"robot_{side}_wrist_wxyz")
        fj = _arr(cells, f"robot_{side}_finger_joints")
        if wp_ is None or wp_.shape != (num_frames, 3) or not np.all(np.isfinite(wp_)):
            raise ValueError(
                f"ManoSharpaData {side} wrist position must be finite with shape ({num_frames}, 3)"
            )
        if wq_ is None or wq_.shape != (num_frames, 4) or not np.all(np.isfinite(wq_)):
            raise ValueError(
                f"ManoSharpaData {side} wrist quaternion must be finite with "
                f"shape ({num_frames}, 4)"
            )
        wrist_norm = np.linalg.norm(wq_, axis=-1, keepdims=True)
        if np.any(wrist_norm <= np.finfo(wq_.dtype).eps) or not np.allclose(
            wrist_norm, 1.0, atol=_QUAT_ATOL, rtol=0.0
        ):
            raise ValueError(
                f"ManoSharpaData {side} wrist quaternions must be unit length"
            )
        wq_ = np.ascontiguousarray(wq_ / wrist_norm)
        if (
            fj is None
            or fj.ndim != 2
            or fj.shape[0] != num_frames
            or not np.all(np.isfinite(fj))
        ):
            raise ValueError(
                f"ManoSharpaData {side} finger joints must be finite with shape ({num_frames}, N)"
            )
        finger_names = tuple(
            str(x) for x in (cells.get(f"{side}_robot_finger_joint_names") or [])
        )
        if (
            len(finger_names) != fj.shape[1]
            or len(set(finger_names)) != len(finger_names)
            or any(not name for name in finger_names)
        ):
            raise ValueError(
                f"ManoSharpaData {side} finger joint names must be unique and match the joint axis"
            )

        cp = _arr(cells, f"mano_{side}_object_contact_positions")
        cn = _arr(cells, f"mano_{side}_object_contact_normals")
        cid = _arr(cells, f"mano_{side}_object_contact_part_ids")
        if cp is None or cn is None or cid is None:
            raise ValueError(
                f"ManoSharpaData {side} contact positions, normals, and part IDs "
                "are required together"
            )
        if (
            cp.ndim != 3
            or cp.shape[0] != num_frames
            or cp.shape[-1] != 3
            or cn.shape != cp.shape
        ):
            raise ValueError(
                f"ManoSharpaData {side} contacts must have matching (T, K, 3) "
                "position/normal shapes"
            )
        if cid.shape != cp.shape[:-1]:
            raise ValueError(
                f"ManoSharpaData {side} contact part IDs must match the slot axes"
            )
        if (
            not np.all(np.isfinite(cp))
            or not np.all(np.isfinite(cn))
            or not np.all(np.isfinite(cid))
        ):
            raise ValueError(f"ManoSharpaData {side} contact fields must be finite")
        if (
            np.any(cid != np.floor(cid))
            or np.any(cid < 0)
            or np.any(cid > num_object_bodies)
        ):
            raise ValueError(
                f"ManoSharpaData {side} contact part IDs must be integers in "
                f"[0, {num_object_bodies}]"
            )
        cid = cid.astype(np.int32)
        active_normal_norm = np.linalg.norm(cn, axis=-1)[cid > 0]
        if active_normal_norm.size and not np.allclose(
            active_normal_norm, 1.0, atol=_QUAT_ATOL, rtol=0.0
        ):
            raise ValueError(
                f"ManoSharpaData {side} active contact normals must be unit length"
            )

        fr = _arr(cells, f"robot_{side}_frames")  # (T, F, 7) pos + wxyz per named frame
        frame_names = tuple(
            str(x) for x in (cells.get(f"{side}_robot_frame_names") or [])
        )
        if fr is None or fr.ndim != 3 or fr.shape[0] != num_frames or fr.shape[-1] != 7:
            raise ValueError(
                f"ManoSharpaData {side} robot frames must have shape ({num_frames}, F, 7)"
            )
        if (
            len(frame_names) != fr.shape[1]
            or len(set(frame_names)) != len(frame_names)
            or any(not name for name in frame_names)
        ):
            raise ValueError(
                f"ManoSharpaData {side} robot frame names must be unique and match the frame axis"
            )
        if not np.all(np.isfinite(fr)):
            raise ValueError(f"ManoSharpaData {side} robot frames must be finite")

        # Contacts use NEAREST (not lerp): a slot's position/normal must not be
        # interpolated across an active->inactive transition, which would place
        # it at a garbage midpoint flying off the object.
        hands[side] = HandReference(
            wrist_pos_w=lerp(wp_),
            wrist_quat_w=slerp_track(wq_),
            finger_joint_pos=lerp(fj),
            finger_joint_names=finger_names,
            contact_pos_w=near(cp),
            contact_normal_w=near(cn),
            contact_part_ids=near(cid),
            frame_pos_w=lerp(np.ascontiguousarray(fr[..., :3])),
            frame_names=frame_names,
        )

    if not resample:
        art_out = art
    elif art.shape[1]:
        art_out = lerp(art)
    else:
        assert t_tgt is not None
        art_out = np.zeros((len(t_tgt), 0))

    return ChordReference(
        metadata=ReferenceMetadata(
            source_path=str(row.path),
            source_fps=src_fps,
            is_resampled=resample,
            schema_version=_optional_string(cells.get("schema_version")),
            motion_kind=_optional_string(cells.get("motion_kind")),
            source_dataset=source_dataset,
            sequence_id=_optional_string(cells.get("sequence_id")),
            robot_name=_optional_string(cells.get("robot_name")),
            raw_motion_file=_optional_string(cells.get("raw_motion_file")),
            coord_frame=_optional_string(cells.get("coord_frame")),
        ),
        playback_fps=(control_fps if control_fps is not None else src_fps),
        hands=hands,
        object_ref=ObjectReference(
            name=object_name,
            body_names=tuple(object_body_names),
            mesh_paths=tuple(object_mesh_paths),
            urdf_paths=tuple(object_urdf_paths),
            mesh_radius=object_mesh_radius,
            body_pos_w=lerp(obj_pos),
            body_quat_w=slerp_bodies(obj_quat),
            articulation=art_out,
            assets=object_assets,
        ),
    )


def _arr(cells: dict, name: str) -> np.ndarray | None:
    if name not in cells:
        return None
    a = np.asarray(cells[name])
    return None if a.size == 0 else a


def _optional_string(value: object) -> str | None:
    return None if value is None or value == "" else str(value)


def _dataset_from_path(path: Path) -> str | None:
    parts = path.parts
    try:
        index = parts.index("human_motion_data")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None
