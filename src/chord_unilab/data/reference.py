"""In-memory CHORD reference trajectory (host numpy), loaded from parquet.

Faithful re-implementation of flash_chord's ``ManoSharpaReference``
(``src/flash_chord/data/mano_sharpa.py``) and the reference dataclasses in
``src/flash_chord/data/reference.py``. Field set, dtypes, shapes, and
conventions are identical: positions in metres, quaternions ``wxyz``, time
axis first (``T``), ``B`` object bodies, ``K`` contact slots per hand,
``Nf`` finger joints, ``F`` named robot frames.

Declared structural deviation (approved): the ground truth stores per-side
dicts of bare attributes on a non-frozen dataclass; here the same data lives
in frozen nested dataclasses (:class:`HandReference` per side,
:class:`ObjectReference`) behind method accessors matching flash_chord's
``Reference`` protocol names. Array contents are field-for-field identical.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

_SIDES = ("left", "right")


@dataclass(frozen=True)
class ReferenceMetadata:
    """Source identity kept separate from the resampled playback rate."""

    source_path: str
    source_fps: float
    is_resampled: bool = False
    schema_version: str | None = None
    motion_kind: str | None = None
    source_dataset: str | None = None
    sequence_id: str | None = None
    robot_name: str | None = None
    raw_motion_file: str | None = None
    coord_frame: str | None = None


@dataclass(frozen=True)
class ObjectBodySpec:
    """Map one imported terminal body name to one named reference body."""

    reference_name: str
    simulation_name: str | None

    def __post_init__(self) -> None:
        if not self.reference_name:
            raise ValueError("object reference body names must be nonempty")
        if self.simulation_name == "":
            raise ValueError(
                "object simulation body names must be nonempty when provided"
            )


@dataclass(frozen=True)
class ObjectJointPhysicsSpec:
    """Passive physical properties for one imported scalar object joint."""

    armature: float
    friction: float

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in (self.armature, self.friction)):
            raise ValueError("object articulation physics parameters must be finite")
        if self.armature < 0.0 or self.friction < 0.0:
            raise ValueError(
                "object articulation armature and friction must be nonnegative"
            )


@dataclass(frozen=True)
class ObjectJointDriveSpec:
    """Implicit position-velocity drive parameters for one object articulation."""

    kp: float
    kd: float
    effort_limit: float

    def __post_init__(self) -> None:
        if not all(
            np.isfinite(value) for value in (self.kp, self.kd, self.effort_limit)
        ):
            raise ValueError("object articulation drive parameters must be finite")
        if self.kp < 0.0 or self.kd < 0.0 or self.effort_limit <= 0.0:
            raise ValueError(
                "object articulation drive gains must be nonnegative and effort must be positive"
            )


@dataclass(frozen=True)
class ObjectArticulationSpec:
    """Map one imported terminal joint name to one reference articulation column."""

    simulation_joint_name: str
    reference_index: int
    physics: ObjectJointPhysicsSpec
    drive: ObjectJointDriveSpec

    def __post_init__(self) -> None:
        if not self.simulation_joint_name:
            raise ValueError("object articulation joint names must be nonempty")
        if self.reference_index < 0:
            raise ValueError(
                "object articulation reference indices must be nonnegative"
            )


@dataclass(frozen=True)
class ObjectAssetSpec:
    """Explicit setup-time contract between one URDF component and reference arrays."""

    name: str
    urdf_path: str
    bodies: tuple[ObjectBodySpec, ...]
    root_reference_name: str
    articulations: tuple[ObjectArticulationSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bodies", tuple(self.bodies))
        object.__setattr__(self, "articulations", tuple(self.articulations))
        if not self.name or not self.urdf_path:
            raise ValueError("object asset names and URDF paths must be nonempty")
        if not self.bodies:
            raise ValueError(f"object asset {self.name!r} must bind at least one body")
        reference_names = tuple(body.reference_name for body in self.bodies)
        simulation_names = tuple(
            body.simulation_name
            for body in self.bodies
            if body.simulation_name is not None
        )
        if len(set(reference_names)) != len(reference_names):
            raise ValueError(
                f"object asset {self.name!r} has duplicate reference body bindings"
            )
        if len(set(simulation_names)) != len(simulation_names):
            raise ValueError(
                f"object asset {self.name!r} has duplicate simulation body bindings"
            )
        if (
            any(body.simulation_name is None for body in self.bodies)
            and len(self.bodies) != 1
        ):
            raise ValueError(
                "unnamed simulation-body binding is valid only for a single-body asset"
            )
        if self.root_reference_name not in reference_names:
            raise ValueError(
                f"object asset {self.name!r} root {self.root_reference_name!r} "
                "is not a bound reference body"
            )
        articulation_names = tuple(b.simulation_joint_name for b in self.articulations)
        articulation_indices = tuple(b.reference_index for b in self.articulations)
        if len(set(articulation_names)) != len(articulation_names):
            raise ValueError(
                f"object asset {self.name!r} has duplicate articulation joint bindings"
            )
        if len(set(articulation_indices)) != len(articulation_indices):
            raise ValueError(
                f"object asset {self.name!r} has duplicate articulation reference bindings"
            )


@dataclass(frozen=True, eq=False)
class HandReference:
    """Per-side hand trajectory: wrist pose, finger joints, contacts, frames."""

    wrist_pos_w: np.ndarray  # (T, 3) [m]
    wrist_quat_w: np.ndarray  # (T, 4) wxyz
    finger_joint_pos: np.ndarray  # (T, Nf) [rad]
    finger_joint_names: tuple[str, ...]  # Nf names matching the last axis
    contact_pos_w: np.ndarray  # (T, K, 3) [m], object-surface contact points
    contact_normal_w: np.ndarray  # (T, K, 3) unit normals on active slots
    contact_part_ids: np.ndarray  # (T, K) int32, 1-indexed object body, 0 = no contact
    frame_pos_w: np.ndarray  # (T, F, 3) named robot-frame positions [m]
    frame_names: tuple[str, ...]  # F names matching the frame axis

    def contact_active(self) -> np.ndarray:
        """``(T,)`` float32: whether any contact slot is active on this frame."""
        active = np.any(self.contact_part_ids > 0, axis=1).astype(np.float32)
        return np.ascontiguousarray(active)

    def sliced(self, frames: slice) -> HandReference:
        """Return a copy restricted to ``frames`` along the time axis."""

        def cut(values: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(values[frames])

        return replace(
            self,
            wrist_pos_w=cut(self.wrist_pos_w),
            wrist_quat_w=cut(self.wrist_quat_w),
            finger_joint_pos=cut(self.finger_joint_pos),
            contact_pos_w=cut(self.contact_pos_w),
            contact_normal_w=cut(self.contact_normal_w),
            contact_part_ids=cut(self.contact_part_ids),
            frame_pos_w=cut(self.frame_pos_w),
        )


@dataclass(frozen=True, eq=False)
class ObjectReference:
    """Object trajectory and asset metadata."""

    name: str
    body_names: tuple[str, ...]  # B unique nonempty names
    mesh_paths: tuple[str, ...]  # B paths as stored in the parquet
    urdf_paths: tuple[str, ...]  # per rigid body; empty for mesh-only articulated
    mesh_radius: np.ndarray  # (B,) float64 bounding-ball radius (torque scale rc)
    body_pos_w: np.ndarray  # (T, B, 3) [m]
    body_quat_w: np.ndarray  # (T, B, 4) wxyz
    articulation: np.ndarray  # (T, A); A == 0 if rigid
    assets: tuple[ObjectAssetSpec, ...]

    def sliced(self, frames: slice) -> ObjectReference:
        """Return a copy restricted to ``frames`` along the time axis."""

        def cut(values: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(values[frames])

        return replace(
            self,
            body_pos_w=cut(self.body_pos_w),
            body_quat_w=cut(self.body_quat_w),
            articulation=cut(self.articulation),
        )


@dataclass(frozen=True, eq=False)
class ChordReference:
    """Dual-hand + object reference trajectory, mirroring ``ManoSharpaReference``."""

    metadata: ReferenceMetadata
    playback_fps: float
    hands: dict[str, HandReference]  # side ("left"/"right") -> trajectory
    object_ref: ObjectReference

    @property
    def num_frames(self) -> int:
        return self.object_ref.body_pos_w.shape[0]

    @property
    def fps(self) -> float:
        return self.playback_fps

    @property
    def sides(self) -> tuple[str, ...]:
        return _SIDES

    # --- hand reference ---
    def wrist_pos_w(self, side: str) -> np.ndarray:
        return self.hands[side].wrist_pos_w

    def wrist_quat_w(self, side: str) -> np.ndarray:
        return self.hands[side].wrist_quat_w

    def finger_joint_pos(self, side: str) -> np.ndarray:
        return self.hands[side].finger_joint_pos

    def finger_joint_names(self, side: str) -> tuple[str, ...]:
        return self.hands[side].finger_joint_names

    def contact_pos_w(self, side: str) -> np.ndarray:
        return self.hands[side].contact_pos_w

    def contact_normal_w(self, side: str) -> np.ndarray:
        return self.hands[side].contact_normal_w

    def contact_part_ids(self, side: str) -> np.ndarray:
        return self.hands[side].contact_part_ids

    def contact_active(self, side: str) -> np.ndarray:
        return self.hands[side].contact_active()

    def frame_pos_w(self, side: str) -> np.ndarray:
        return self.hands[side].frame_pos_w

    def frame_names(self, side: str) -> tuple[str, ...]:
        return self.hands[side].frame_names

    # --- object reference ---
    def object_body_pos_w(self) -> np.ndarray:
        return self.object_ref.body_pos_w

    def object_body_quat_w(self) -> np.ndarray:
        return self.object_ref.body_quat_w

    def object_articulation(self) -> np.ndarray:
        return self.object_ref.articulation

    def object_name(self) -> str:
        return self.object_ref.name

    def object_body_names(self) -> tuple[str, ...]:
        return self.object_ref.body_names

    def object_mesh_paths(self) -> tuple[str, ...]:
        return self.object_ref.mesh_paths

    def object_urdf_paths(self) -> tuple[str, ...]:
        return self.object_ref.urdf_paths

    def object_mesh_radius(self) -> np.ndarray:
        return self.object_ref.mesh_radius

    def object_assets(self) -> tuple[ObjectAssetSpec, ...]:
        return self.object_ref.assets

    def frame_window(self, start_frame: int = 0, end_frame: int = -1) -> ChordReference:
        """Return a copy restricted to the half-open frame window.

        Matches flash_chord: the window applies to the loaded (already
        resampled) arrays, ``start_frame`` inclusive, ``end_frame``
        exclusive, negative end means sequence end.
        """
        frames = reference_frame_slice(self.num_frames, start_frame, end_frame)
        if frames.start == 0 and frames.stop == self.num_frames:
            return self
        return replace(
            self,
            hands={side: hand.sliced(frames) for side, hand in self.hands.items()},
            object_ref=self.object_ref.sliced(frames),
        )


def reference_frame_slice(
    num_frames: int, start_frame: int = 0, end_frame: int = -1
) -> slice:
    """Validate and resolve a half-open reference-frame window.

    Matches robotic_grounding's motion trimming convention: ``start_frame``
    inclusive, ``end_frame`` exclusive, any negative end selects the rest.
    """
    end = num_frames if end_frame < 0 else end_frame
    if start_frame < 0 or end > num_frames or start_frame >= end:
        raise ValueError(
            f"invalid reference frame window [{start_frame}, {end}) for "
            f"{num_frames} frames; require 0 <= start_frame < end_frame <= num_frames"
        )
    return slice(start_frame, end)
