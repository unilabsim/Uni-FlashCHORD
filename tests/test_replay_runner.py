"""ReplayRunner tests against a fake implementing the public UniSim contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout

from chord_unilab.assets.compose_sharpa_scene import (
    HandReferenceBinding,
    ObjectComponentBinding,
    SharpaSceneAssembly,
)
from chord_unilab.data.reference import (
    ChordReference,
    ObjectArticulationSpec,
    ObjectAssetSpec,
    ObjectBodySpec,
    ObjectJointDriveSpec,
    ObjectJointPhysicsSpec,
)
from chord_unilab.replay.runner import ReplayRunner


class FakeReference:
    num_frames = 2

    def __init__(self) -> None:
        self.wrist_positions = {
            side: np.asarray(
                [[0.1, -0.2 if side == "left" else 0.2, 0.5], [0.2, 0.3, 0.6]],
                dtype=np.float32,
            )
            for side in ("left", "right")
        }
        self.wrist_quaternions = {
            side: np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1))
            for side in ("left", "right")
        }
        self.finger_positions = {
            side: np.asarray([[0.1], [0.4]], dtype=np.float32)
            for side in ("left", "right")
        }
        self.body_positions = np.asarray(
            [
                [[0.0, 0.0, 0.8], [0.0, 0.0, 0.7]],
                [[0.1, 0.0, 0.9], [0.1, 0.0, 0.8]],
            ],
            dtype=np.float32,
        )
        self.body_quaternions = np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 2, 1)
        )
        self.articulation = np.asarray([[0.2], [0.5]], dtype=np.float32)

    def wrist_pos_w(self, side: str) -> np.ndarray:
        return self.wrist_positions[side]

    def wrist_quat_w(self, side: str) -> np.ndarray:
        return self.wrist_quaternions[side]

    def finger_joint_names(self, side: str) -> tuple[str, ...]:
        return ("finger",)

    def finger_joint_pos(self, side: str) -> np.ndarray:
        return self.finger_positions[side]

    def object_body_names(self) -> tuple[str, ...]:
        return ("root", "child")

    def object_body_pos_w(self) -> np.ndarray:
        return self.body_positions

    def object_body_quat_w(self) -> np.ndarray:
        return self.body_quaternions

    def object_articulation(self) -> np.ndarray:
        return self.articulation


class FakeBackend:
    def __init__(self, layout: CompiledSceneLayout) -> None:
        self.layout = layout
        self.qpos = torch.zeros((1, layout.nq), dtype=torch.float32)
        self.qvel = torch.zeros((1, layout.nv), dtype=torch.float32)
        for entity in layout.entities:
            if entity.root_qpos_indices:
                self.qpos[0, list(entity.root_qpos_indices)[3]] = 1.0
            for joint in entity.root_joints:
                if joint.kind == "ball":
                    self.qpos[0, joint.qpos_indices[0]] = 1.0
        actuator_names = []
        actuator_joints = []
        for entity in layout.entities:
            for name, joint in zip(
                entity.actuator_names, entity.actuator_joint_names, strict=True
            ):
                actuator_names.append(f"{entity.name}/{name}")
                actuator_joints.append(f"{entity.name}/{joint}")
        self.actuator_names = tuple(actuator_names)
        self.actuator_joints = tuple(actuator_joints)
        self.body_ids = {
            f"{entity.name}/{body}": body_id
            for entity in layout.entities
            for body, body_id in zip(entity.body_names, entity.body_ids, strict=True)
        }
        self.masses = np.asarray([[1.0, 1.0, 2.0, 3.0]], dtype=np.float32)
        self.inertias = np.ones((1, 4, 3), dtype=np.float32)
        self.inertias[0, 3] = [3.0, 6.0, 9.0]
        self.step_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.state_writes: list[tuple[np.ndarray, np.ndarray]] = []
        self.reset_requests: list[Any] = []
        self.gain_calls: list[tuple[np.ndarray, float, np.ndarray]] = []
        self.force_calls: list[tuple[np.ndarray | None, np.ndarray | None]] = []
        self.joint_force = torch.zeros_like(self.qvel)
        self.body_force = torch.zeros((1, layout.nbody, 6), dtype=torch.float32)
        self.cleanup_calls = 0

    @property
    def num_envs(self) -> int:
        return 1

    def get_scene_layout(self) -> CompiledSceneLayout:
        return self.layout

    def get_state_views(self, fields: tuple[str, ...]) -> dict[str, torch.Tensor]:
        return {field: getattr(self, field) for field in fields}

    def get_actuator_names(self) -> tuple[str, ...]:
        return self.actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return self.actuator_joints

    def get_body_id(self, name: str) -> int:
        return self.body_ids[name]

    def get_body_mass(self) -> np.ndarray:
        return self.masses.copy()

    def get_reset_term_default(self, term: str) -> np.ndarray:
        assert term == "body_inertia"
        return self.inertias.copy()

    def set_actuator_gain_scale(self, actuator_ids, scale, *, env_ids=None) -> None:
        self.gain_calls.append(
            (np.asarray(actuator_ids).copy(), scale, np.asarray(env_ids).copy())
        )

    def set_generalized_force_tensor(self, *, joint_f=None, body_f=None) -> None:
        if joint_f is not None:
            self.joint_force = joint_f.clone()
        if body_f is not None:
            self.body_force = body_f.clone()
        self.force_calls.append(
            (
                None if joint_f is None else joint_f.clone().numpy(),
                None if body_f is None else body_f.clone().numpy(),
            )
        )

    def set_state_tensor(self, env_ids, qpos, qvel) -> None:
        rows = env_ids.cpu().numpy()
        self.qpos[rows] = qpos
        self.qvel[rows] = qvel
        self.state_writes.append((qpos.clone().numpy(), qvel.clone().numpy()))

    def reset_entities(self, request) -> None:
        self.reset_requests.append(request)
        row = request.env_ids[0]
        for patch in request.patches:
            entity = self.layout.get_entity(patch.entity)
            if patch.root_pose is not None:
                self.qpos[row, list(entity.root_qpos_indices)] = torch.as_tensor(
                    patch.root_pose[0].copy()
                )
            if patch.root_velocity is not None:
                self.qvel[row, list(entity.root_qvel_indices)] = torch.as_tensor(
                    patch.root_velocity[0].copy()
                )
            joints = {joint.name: joint for joint in entity.joints}
            for index, name in enumerate(patch.joint_names or ()):
                joint = joints[name]
                if patch.joint_positions is not None:
                    self.qpos[row, list(joint.qpos_indices)] = float(
                        patch.joint_positions[0, index]
                    )
                if patch.joint_velocities is not None:
                    self.qvel[row, list(joint.qvel_indices)] = float(
                        patch.joint_velocities[0, index]
                    )

    def get_sensor_view(self, name: str) -> torch.Tensor:
        for prefix, width in (
            ("track_pos_w_", 3),
            ("track_quat_w_", 4),
            ("track_linvel_w_", 3),
            ("track_angvel_w_", 3),
        ):
            if name.startswith(prefix):
                body = name[len(prefix) :]
                break
        else:  # pragma: no cover - guards test typos
            raise KeyError(name)
        entity_name, local_body = body.split("/", 1)
        entity = self.layout.get_entity(entity_name)
        if local_body != entity.root_body:
            values = {
                "track_pos_w_": [9.0, 9.0, 9.0],
                "track_quat_w_": [1.0, 0.0, 0.0, 0.0],
                "track_linvel_w_": [0.0, 0.0, 0.0],
                "track_angvel_w_": [0.0, 0.0, 0.0],
            }
            return torch.asarray([values[prefix]], dtype=torch.float32)
        if prefix == "track_pos_w_":
            return self.qpos[:, list(entity.root_qpos_indices[:3])]
        if prefix == "track_quat_w_":
            return self.qpos[:, list(entity.root_qpos_indices[3:])]
        if prefix == "track_linvel_w_":
            return self.qvel[:, list(entity.root_qvel_indices[:3])]
        return self.qvel[:, list(entity.root_qvel_indices[3:])]

    def step_tensor(self, ctrl, nsteps=1) -> None:
        assert nsteps == 1
        self.step_calls.append(
            (
                ctrl.clone().numpy(),
                self.joint_force.clone().numpy(),
                self.body_force.clone().numpy(),
            )
        )

    def cleanup_scene_assets(self) -> None:
        self.cleanup_calls += 1


def make_layout() -> CompiledSceneLayout:
    entities = []
    nq = nv = nu = 0
    for body_id, side in enumerate(("left", "right")):
        root_joints = []
        root_qpos = []
        root_qvel = []
        actuator_names: list[str] = []
        actuator_joints: list[str] = []
        actuator_indices: list[int] = []
        for axis in "xyz":
            root_qpos.append(nq)
            root_qvel.append(nv)
            root_joints.append(
                JointLayout(f"wrist_{axis}", "slide", (nq,), (nv,), "palm")
            )
            actuator_names.extend((f"wrist_{axis}_pos", f"wrist_{axis}_vel"))
            actuator_joints.extend((f"wrist_{axis}", f"wrist_{axis}"))
            actuator_indices.extend((nu, nu + 1))
            nq += 1
            nv += 1
            nu += 2
        ball_qpos = tuple(range(nq, nq + 4))
        ball_qvel = tuple(range(nv, nv + 3))
        root_qpos.extend(ball_qpos)
        root_qvel.extend(ball_qvel)
        root_joints.append(
            JointLayout("wrist_rotvec", "ball", ball_qpos, ball_qvel, "palm")
        )
        nq += 4
        nv += 3
        finger = JointLayout("finger", "hinge", (nq,), (nv,), "palm")
        actuator_names.extend(("finger_pos", "finger_vel"))
        actuator_joints.extend(("finger", "finger"))
        actuator_indices.extend((nu, nu + 1))
        nq += 1
        nv += 1
        nu += 2
        entities.append(
            EntityLayout(
                f"{side}_hand",
                "articulation",
                "floating",
                "palm",
                ("palm",),
                (body_id,),
                (None,),
                (finger,),
                tuple(actuator_names),
                tuple(actuator_joints),
                tuple(actuator_indices),
                tuple(root_qpos),
                tuple(root_qvel),
                root_joints=tuple(root_joints),
            )
        )
    object_joint = JointLayout("object_joint", "hinge", (nq + 7,), (nv + 6,), "child")
    entities.append(
        EntityLayout(
            "object",
            "articulation",
            "floating",
            "root",
            ("child", "root"),
            (2, 3),
            ("root", None),
            (object_joint,),
            ("object_joint_pos", "object_joint_vel"),
            ("object_joint", "object_joint"),
            (nu, nu + 1),
            tuple(range(nq, nq + 7)),
            tuple(range(nv, nv + 6)),
        )
    )
    nq += 8
    nv += 7
    nu += 2
    return CompiledSceneLayout(tuple(entities), nq, nv, nu, 4)


def make_assembly() -> SimpleNamespace:
    hands = tuple(
        HandReferenceBinding(side, f"{side}_hand", ("finger",), ("finger",))
        for side in ("left", "right")
    )
    spec = ObjectAssetSpec(
        "object",
        "object.urdf",
        (
            ObjectBodySpec("root", "root"),
            ObjectBodySpec("child", "child"),
        ),
        "root",
        (
            ObjectArticulationSpec(
                "joint",
                0,
                ObjectJointPhysicsSpec(0.0, 0.0),
                ObjectJointDriveSpec(10.0, 1.0, 100.0),
            ),
        ),
    )
    binding = ObjectComponentBinding(
        spec,
        "object",
        Path("object.xml"),
        {"root": "object/root", "child": "object/child"},
        {0: "object/object_joint"},
    )
    return SimpleNamespace(hand_bindings=hands, bindings=(binding,))


def make_runner(mode: str, *, substeps=4, record=False):
    layout = make_layout()
    backend = FakeBackend(layout)
    runner = ReplayRunner(
        backend,
        cast(ChordReference, FakeReference()),
        cast(SharpaSceneAssembly, make_assembly()),
        cast(Any, mode),
        substeps,
        record=record,
    )
    return runner, backend, layout


def test_kinematic_writes_reference_without_stepping() -> None:
    runner, backend, layout = make_runner("kinematic")
    runner.reset(0)
    runner.step_frame(1)

    assert backend.step_calls == []
    assert len(backend.reset_requests) == 1
    assert len(backend.state_writes) == 1
    left = layout.get_entity("left_hand")
    np.testing.assert_allclose(
        backend.qpos[0, list(left.root_qpos_indices[:3])], [0.2, 0.3, 0.6]
    )
    np.testing.assert_allclose(backend.qpos[0, left.joints[0].qpos_indices[0]], 0.4)
    obj = layout.get_entity("object")
    np.testing.assert_allclose(backend.qpos[0, obj.joints[0].qpos_indices[0]], 0.5)
    assert torch.count_nonzero(backend.qvel) == 0
    assert torch.count_nonzero(backend.joint_force) == 0
    assert torch.count_nonzero(backend.body_force) == 0
    runner.close()
    assert backend.cleanup_calls == 1


def test_voc0_steps_every_substep_and_scales_only_object_actuators() -> None:
    runner, backend, layout = make_runner("voc0", substeps=4)
    runner.reset(0)
    runner.step_frame(1)

    assert len(backend.step_calls) == 4
    obj = layout.get_entity("object")
    assert len(backend.gain_calls) == 1
    actuator_ids, scale, env_ids = backend.gain_calls[0]
    np.testing.assert_array_equal(actuator_ids, obj.actuator_indices)
    assert scale == 0.0
    np.testing.assert_array_equal(env_ids, [0])
    assert set(backend.gain_calls[0][0]).isdisjoint(
        set(layout.get_entity("left_hand").actuator_indices)
        | set(layout.get_entity("right_hand").actuator_indices)
    )
    for ctrl, _joint_force, body_force in backend.step_calls:
        assert ctrl[0, obj.actuator_indices[0]] == 0.5
        assert ctrl[0, obj.actuator_indices[1]] == 0.0
        np.testing.assert_array_equal(
            body_force[0, backend.body_ids["object/root"], :], np.zeros(6)
        )
    runner.close()


def test_voc1_uses_total_object_mass_and_root_inertia() -> None:
    runner, backend, layout = make_runner("voc1", substeps=2)
    runner.reset(0)
    obj = layout.get_entity("object")
    backend.qvel[0, obj.root_qvel_indices[3]] = 0.1
    runner.step_frame(0)

    assert len(backend.step_calls) == 2
    np.testing.assert_array_equal(backend.gain_calls[0][0], obj.actuator_indices)
    assert backend.gain_calls[0][1] == 1.0
    for _ctrl, _joint_force, body_force in backend.step_calls:
        np.testing.assert_allclose(
            body_force[0, backend.body_ids["object/root"], :],
            [0.0, 0.0, 5.0 * 9.81, -24.0, 0.0, 0.0],
            rtol=1.0e-6,
        )
    runner.reset(1)
    assert torch.count_nonzero(backend.joint_force) == 0
    assert torch.count_nonzero(backend.body_force) == 0
    runner.close()
    assert torch.count_nonzero(backend.joint_force) == 0
    assert torch.count_nonzero(backend.body_force) == 0


def test_optional_npz_contains_only_public_replay_samples(tmp_path: Path) -> None:
    runner, _backend, layout = make_runner("voc0", substeps=1, record=True)
    runner.reset(0)
    runner.step_frame(1)
    output = runner.save_npz(tmp_path / "replay.npz")
    runner.close()

    with np.load(output) as data:
        np.testing.assert_array_equal(data["frames"], [0, 1])
        assert data["achieved_qpos"].shape == (2, layout.nq)
        assert data["achieved_qvel"].shape == (2, layout.nv)
        assert data["achieved_body_pose"].shape == (2, 2, 7)
        assert data["reference_qpos"].shape == (2, layout.nq)
        assert data["reference_body_pose"].shape == (2, 2, 7)
        assert data["voc_wrench"].shape == (2, 1, 6)
        assert data["joint_targets"].shape == (2, layout.nu)
        np.testing.assert_array_equal(data["voc_wrench"], np.zeros((2, 1, 6)))
        assert not {"nacon", "qfrc_actuator", "body_q"}.intersection(data.files)
