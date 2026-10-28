"""Reference replay through UniSim's public tensor and entity-state contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import torch

from chord_unilab.assets.compose_sharpa_scene import SharpaSceneAssembly
from chord_unilab.assets.reference_state import reset_sharpa_scene
from chord_unilab.data.reference import ChordReference
from chord_unilab.replay.object_control import (
    object_root_voc,
    sharpa_ball_wrist_torque,
)
from chord_unilab.replay.quaternion import from_rotvec_wxyz, to_rotvec_wxyz

ReplayMode = Literal["kinematic", "voc0", "voc1"]


class ReplayRunner:
    """Replay one environment with kinematic writes or continuous physical control."""

    def __init__(
        self,
        backend: Any,
        reference: ChordReference,
        assembly: SharpaSceneAssembly,
        mode: ReplayMode,
        substeps: int = 4,
        *,
        record: bool = False,
    ) -> None:
        if mode not in ("kinematic", "voc0", "voc1"):
            raise ValueError(f"unknown replay mode {mode!r}")
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps <= 0:
            raise ValueError("substeps must be a positive integer")
        if backend.num_envs != 1:
            raise ValueError("ReplayRunner requires exactly one environment")
        self.backend = backend
        self.reference = reference
        self.assembly = assembly
        self.mode = mode
        self.substeps = substeps
        self.layout = backend.get_scene_layout()
        self._closed = False
        self._record = record
        self._records: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "achieved_qpos",
                "achieved_qvel",
                "achieved_body_pose",
                "reference_qpos",
                "reference_body_pose",
                "voc_wrench",
                "joint_targets",
            )
        }
        self._frames: list[int] = []

        views = backend.get_state_views(("qpos", "qvel"))
        self._device = views["qpos"].device
        self._env_ids = torch.tensor([0], dtype=torch.int64, device=self._device)
        self._ctrl = torch.zeros(
            (1, self.layout.nu), dtype=torch.float32, device=self._device
        )
        self._joint_force = torch.zeros(
            (1, self.layout.nv), dtype=torch.float32, device=self._device
        )
        body_count = np.asarray(backend.get_body_mass()).shape[-1]
        self._zero_body_force = torch.zeros(
            (1, body_count, 6), dtype=torch.float32, device=self._device
        )
        self._body_force = self._zero_body_force.clone()
        self._qpos_base = _host(views["qpos"])[0].copy()
        self._position_rows: dict[str, int] = {}
        self._velocity_rows: dict[str, int] = {}
        self._object_actuator_ids: list[int] = []
        self._hand_ball_joints: dict[str, Any] = {}
        self._object_root_ids: dict[str, int] = {}
        self._object_mass: dict[str, float] = {}
        self._object_inertia: dict[str, np.ndarray] = {}
        self._body_names = tuple(
            body for binding in assembly.bindings for body in binding.bodies.values()
        )
        self._build_cold_mappings()
        self._last_wrench = np.zeros((len(assembly.bindings), 6), dtype=np.float64)
        self._clear_forces()
        if mode == "voc0":
            self._set_object_gain_scale(0.0)
        elif mode == "voc1":
            self._set_object_gain_scale(1.0)

    def _build_cold_mappings(self) -> None:
        actuator_names = self.backend.get_actuator_names()
        actuator_joints = self.backend.get_actuator_joint_names()
        if (
            len(actuator_names) != self.layout.nu
            or len(actuator_joints) != self.layout.nu
        ):
            raise ValueError("actuator metadata does not match the public scene layout")
        for row, (actuator, joint) in enumerate(
            zip(actuator_names, actuator_joints, strict=True)
        ):
            local_actuator = actuator.rsplit("/", 1)[-1]
            local_joint = joint.rsplit("/", 1)[-1]
            if local_actuator == f"{local_joint}_pos":
                self._position_rows[joint] = row
            elif local_actuator == f"{local_joint}_vel":
                self._velocity_rows[joint] = row

        driven_joints: list[str] = []
        for hand in self.assembly.hand_bindings:
            entity = self.layout.get_entity(hand.entity_name)
            balls = [joint for joint in entity.root_joints if joint.kind == "ball"]
            if len(balls) != 1:
                raise ValueError(
                    f"{hand.entity_name} must have exactly one root ball joint"
                )
            self._hand_ball_joints[hand.side] = balls[0]
            driven_joints.extend(
                f"{hand.entity_name}/{joint.name}"
                for joint in entity.root_joints
                if joint.kind == "slide"
            )
            driven_joints.extend(
                f"{hand.entity_name}/{name}" for name in hand.joint_names
            )
        for binding in self.assembly.bindings:
            root_body_id = self.backend.get_body_id(binding.root_body)
            self._object_root_ids[binding.entity_name] = root_body_id
            body_ids = [
                self.backend.get_body_id(body) for body in binding.bodies.values()
            ]
            masses = np.asarray(self.backend.get_body_mass(), dtype=np.float64)
            if masses.ndim == 2:
                if masses.shape[0] != 1:
                    raise ValueError(
                        "ReplayRunner requires one body-mass row per environment"
                    )
                masses = masses[0]
            if masses.ndim != 1:
                raise ValueError("body masses must have a single body axis")
            self._object_mass[binding.entity_name] = float(masses[body_ids].sum())
            inertia = np.asarray(
                self.backend.get_reset_term_default("body_inertia"), dtype=np.float64
            )
            if inertia.ndim == 3:
                if inertia.shape[0] != 1:
                    raise ValueError(
                        "ReplayRunner requires one inertia row per environment"
                    )
                inertia = inertia[0]
            if inertia.ndim != 2:
                raise ValueError(
                    "body inertia defaults must have body and principal axes"
                )
            self._object_inertia[binding.entity_name] = inertia[root_body_id].copy()
            driven_joints.extend(
                f"{binding.entity_name}/{name}" for name in binding.joint_names
            )
            self._object_actuator_ids.extend(
                self._actuator_row(joint, suffix)
                for joint in (
                    f"{binding.entity_name}/{name}" for name in binding.joint_names
                )
                for suffix in ("pos", "vel")
            )
        for joint in driven_joints:
            self._actuator_row(joint, "pos")
            self._actuator_row(joint, "vel")

    def _actuator_row(self, joint: str, suffix: str) -> int:
        rows = self._position_rows if suffix == "pos" else self._velocity_rows
        try:
            return rows[joint]
        except KeyError as exc:
            raise ValueError(
                f"missing {suffix} actuator for public joint {joint!r}"
            ) from exc

    def _set_object_gain_scale(self, scale: float) -> None:
        if self._object_actuator_ids:
            self.backend.set_actuator_gain_scale(
                np.asarray(self._object_actuator_ids, dtype=np.int64),
                scale,
                env_ids=np.asarray([0], dtype=np.int64),
            )

    def _clear_forces(self) -> None:
        self._joint_force.zero_()
        self._zero_body_force.zero_()
        self._body_force.zero_()
        self.backend.set_generalized_force_tensor(
            joint_f=self._joint_force,
            body_f=self._zero_body_force,
        )

    def _validate_frame(self, frame: int) -> None:
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise TypeError("reference frame must be an integer")
        if frame < 0 or frame >= self.reference.num_frames:
            raise ValueError(
                f"reference frame {frame} is outside [0, {self.reference.num_frames})"
            )

    def reset(self, frame: int = 0) -> None:
        """Snap hands and objects to one reference frame and clear persistent forces."""
        self._require_open()
        reset_sharpa_scene(self.backend, self.reference, self.assembly, frame)
        self._ctrl.zero_()
        self._last_wrench.fill(0.0)
        self._clear_forces()
        self._frames.clear()
        for values in self._records.values():
            values.clear()
        self._capture(frame)

    def _reference_qpos(self, frame: int) -> np.ndarray:
        qpos = self._qpos_base.copy()
        for hand in self.assembly.hand_bindings:
            entity = self.layout.get_entity(hand.entity_name)
            qpos[list(entity.root_qpos_indices)] = np.concatenate(
                (
                    self.reference.wrist_pos_w(hand.side)[frame],
                    self.reference.wrist_quat_w(hand.side)[frame],
                )
            )
            reference_names = self.reference.finger_joint_names(hand.side)
            reference_positions = self.reference.finger_joint_pos(hand.side)[frame]
            joints = {joint.name: joint for joint in entity.joints}
            for name in hand.joint_names:
                joint = joints[name]
                reference_index = reference_names.index(name)
                qpos[list(joint.qpos_indices)] = reference_positions[reference_index]
        for binding in self.assembly.bindings:
            entity = self.layout.get_entity(binding.entity_name)
            root_index = self.reference.object_body_names().index(
                binding.spec.root_reference_name
            )
            qpos[list(entity.root_qpos_indices)] = np.concatenate(
                (
                    self.reference.object_body_pos_w()[frame, root_index],
                    self.reference.object_body_quat_w()[frame, root_index],
                )
            )
            joints = {joint.name: joint for joint in entity.joints}
            for articulation, name in zip(
                binding.spec.articulations, binding.joint_names, strict=True
            ):
                qpos[list(joints[name].qpos_indices)] = (
                    self.reference.object_articulation()[
                        frame, articulation.reference_index
                    ]
                )
        return qpos

    def _control_targets(self, frame: int) -> None:
        self._ctrl.zero_()
        targets: dict[str, float] = {}
        for hand in self.assembly.hand_bindings:
            entity = self.layout.get_entity(hand.entity_name)
            slide_joints = [
                joint for joint in entity.root_joints if joint.kind == "slide"
            ]
            root_columns = {joint.qpos_indices[0]: joint for joint in slide_joints}
            for index, root_column in enumerate(entity.root_qpos_indices[:3]):
                joint = root_columns.get(root_column)
                if joint is None:
                    raise ValueError(
                        f"missing wrist slide at root column {root_column}"
                    )
                targets[f"{hand.entity_name}/{joint.name}"] = float(
                    self.reference.wrist_pos_w(hand.side)[frame, index]
                )
            reference_names = self.reference.finger_joint_names(hand.side)
            positions = self.reference.finger_joint_pos(hand.side)[frame]
            for name in hand.joint_names:
                targets[f"{hand.entity_name}/{name}"] = float(
                    positions[reference_names.index(name)]
                )
        for binding in self.assembly.bindings:
            for articulation, name in zip(
                binding.spec.articulations, binding.joint_names, strict=True
            ):
                targets[f"{binding.entity_name}/{name}"] = float(
                    self.reference.object_articulation()[
                        frame, articulation.reference_index
                    ]
                )
        for joint, value in targets.items():
            self._ctrl[0, self._actuator_row(joint, "pos")] = value

    def _reference_body_pose(self, frame: int) -> np.ndarray:
        poses = []
        body_names = self.reference.object_body_names()
        for binding in self.assembly.bindings:
            for reference_name in binding.spec.bodies:
                index = body_names.index(reference_name.reference_name)
                poses.append(
                    np.concatenate(
                        (
                            self.reference.object_body_pos_w()[frame, index],
                            self.reference.object_body_quat_w()[frame, index],
                        )
                    )
                )
        return np.asarray(poses, dtype=np.float32)

    def _achieved_body_pose(self) -> np.ndarray:
        poses = []
        for body in self._body_names:
            position = _host(self.backend.get_sensor_view(f"track_pos_w_{body}"))[0]
            quaternion = _host(self.backend.get_sensor_view(f"track_quat_w_{body}"))[0]
            poses.append(np.concatenate((position, quaternion)))
        return np.asarray(poses, dtype=np.float32)

    def _capture(self, frame: int) -> None:
        if not self._record:
            return
        views = self.backend.get_state_views(("qpos", "qvel"))
        self._frames.append(frame)
        self._records["achieved_qpos"].append(_host(views["qpos"])[0].copy())
        self._records["achieved_qvel"].append(_host(views["qvel"])[0].copy())
        self._records["achieved_body_pose"].append(self._achieved_body_pose())
        self._records["reference_qpos"].append(self._reference_qpos(frame))
        self._records["reference_body_pose"].append(self._reference_body_pose(frame))
        self._records["voc_wrench"].append(self._last_wrench.copy())
        self._records["joint_targets"].append(_host(self._ctrl)[0].copy())

    def _write_kinematic_frame(self, frame: int) -> None:
        qpos = torch.as_tensor(
            self._reference_qpos(frame)[None, :],
            dtype=torch.float32,
            device=self._device,
        )
        qvel = torch.zeros(
            (1, self.layout.nv), dtype=torch.float32, device=self._device
        )
        self.backend.set_state_tensor(self._env_ids, qpos, qvel)

    def _apply_physical_substep(self, frame: int) -> None:
        self._joint_force.zero_()
        self._body_force.zero_()
        views = self.backend.get_state_views(("qpos", "qvel"))
        qpos = _host(views["qpos"])[0]
        for hand in self.assembly.hand_bindings:
            ball = self._hand_ball_joints[hand.side]
            current = qpos[list(ball.qpos_indices)]
            target_rotvec = to_rotvec_wxyz(
                self.reference.wrist_quat_w(hand.side)[frame]
            )
            torque = sharpa_ball_wrist_torque(current, from_rotvec_wxyz(target_rotvec))
            self._joint_force[0, list(ball.qvel_indices)] = torch.as_tensor(
                torque, dtype=torch.float32, device=self._device
            )

        wrenches = []
        voc_scale = 0.0 if self.mode == "voc0" else 1.0
        for binding in self.assembly.bindings:
            root = binding.root_body
            position = _host(self.backend.get_sensor_view(f"track_pos_w_{root}"))[0]
            quaternion = _host(self.backend.get_sensor_view(f"track_quat_w_{root}"))[0]
            linear = _host(self.backend.get_sensor_view(f"track_linvel_w_{root}"))[0]
            angular = _host(self.backend.get_sensor_view(f"track_angvel_w_{root}"))[0]
            root_index = self.reference.object_body_names().index(
                binding.spec.root_reference_name
            )
            force, torque = object_root_voc(
                position,
                self.reference.object_body_pos_w()[frame, root_index],
                quaternion,
                self.reference.object_body_quat_w()[frame, root_index],
                linear,
                angular,
                self._object_mass[binding.entity_name],
                self._object_inertia[binding.entity_name],
                voc_scale=voc_scale,
            )
            root_body_id = self._object_root_ids[binding.entity_name]
            self._body_force[0, root_body_id, :3] = torch.as_tensor(
                force, dtype=torch.float32, device=self._device
            )
            self._body_force[0, root_body_id, 3:] = torch.as_tensor(
                torque, dtype=torch.float32, device=self._device
            )
            wrenches.append(np.concatenate((force, torque)))
        self._last_wrench = np.asarray(wrenches, dtype=np.float64).reshape(
            len(self.assembly.bindings), 6
        )
        self.backend.set_generalized_force_tensor(
            joint_f=self._joint_force, body_f=self._body_force
        )
        self.backend.step_tensor(self._ctrl, nsteps=1)

    def step_frame(self, frame: int) -> None:
        """Write or track one reference frame without implicitly advancing the frame."""
        self._require_open()
        self._validate_frame(frame)
        if self.mode == "kinematic":
            self._last_wrench.fill(0.0)
            self._clear_forces()
            self._write_kinematic_frame(frame)
        else:
            self._control_targets(frame)
            for _ in range(self.substeps):
                self._apply_physical_substep(frame)
        self._capture(frame)

    def save_npz(self, path: str | Path) -> Path:
        """Save optional public-state samples; no native solver diagnostics are used."""
        if not self._record:
            raise RuntimeError("recording is disabled; construct with record=True")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: np.asarray(values) for name, values in self._records.items()}
        payload: dict[str, Any] = {
            "frames": np.asarray(self._frames, dtype=np.int64),
            "body_names": np.asarray(self._body_names),
            "object_root_names": np.asarray(
                [binding.root_body for binding in self.assembly.bindings]
            ),
            "mode": np.asarray(self.mode),
            **arrays,
        }
        np.savez_compressed(output, **payload)
        return output

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("ReplayRunner is closed")

    def close(self) -> None:
        """Clear persistent forces before releasing the backend-owned scene."""
        if self._closed:
            return
        self._clear_forces()
        self._closed = True
        self.backend.cleanup_scene_assets()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _host(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy()
