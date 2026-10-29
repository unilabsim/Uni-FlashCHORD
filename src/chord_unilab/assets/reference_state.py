"""Shared reference-to-public-entity reset mapping for replay and task resets."""

from __future__ import annotations

from typing import Any

import numpy as np

from chord_unilab.assets.compose_sharpa_scene import SharpaSceneAssembly
from chord_unilab.data.reference import ChordReference


def reference_state_patch(
    reference: ChordReference,
    assembly: SharpaSceneAssembly,
    frame: int | np.ndarray,
    *,
    env_ids: tuple[int, ...] = (0,),
    finger_scale: float | np.ndarray = 1.0,
    finger_limits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
):
    """Return public zero-velocity reference states for the selected environments.

    The scalar form remains the replay default. Vector frames and finger scales
    support training resets without changing that exact-pose behavior. Support
    and fixed-floor state are intentionally absent, so reset preserves their
    asset defaults.
    """
    from unisim.entities import EntityStatePatch, SceneResetRequest

    if not env_ids:
        raise ValueError("env_ids must not be empty")
    row_count = len(env_ids)
    frames = np.broadcast_to(np.asarray(frame), (row_count,))
    if not np.issubdtype(frames.dtype, np.integer) or np.issubdtype(
        frames.dtype, np.bool_
    ):
        raise TypeError("reference frames must be integers")
    if np.any(frames < 0) or np.any(frames >= reference.num_frames):
        raise ValueError(f"reference frames must be inside [0, {reference.num_frames})")
    scales = np.broadcast_to(np.asarray(finger_scale, dtype=np.float32), (row_count,))
    if not np.all(np.isfinite(scales)) or np.any(scales < 0.0):
        raise ValueError("finger scales must be finite and non-negative")
    zeros6 = np.zeros((row_count, 6), dtype=np.float32)
    patches = []
    for hand_binding in assembly.hand_bindings:
        root_pose = np.concatenate(
            (
                reference.wrist_pos_w(hand_binding.side)[frames],
                reference.wrist_quat_w(hand_binding.side)[frames],
            ),
            axis=-1,
        )
        reference_names = reference.finger_joint_names(hand_binding.side)
        finger_indices = [
            reference_names.index(name) for name in hand_binding.joint_names
        ]
        fingers = reference.finger_joint_pos(hand_binding.side)[frames][
            :, finger_indices
        ]
        fingers = fingers * scales[:, None]
        if finger_limits is not None and hand_binding.side in finger_limits:
            lower, upper = finger_limits[hand_binding.side]
            fingers = np.clip(fingers, lower, upper)
        patches.append(
            EntityStatePatch(
                hand_binding.entity_name,
                root_pose=root_pose.astype(np.float32),
                root_velocity=zeros6.copy(),
                joint_positions=np.asarray(fingers, dtype=np.float32),
                joint_velocities=np.zeros(
                    (row_count, len(finger_indices)), dtype=np.float32
                ),
                joint_names=hand_binding.joint_names,
            )
        )
    for object_binding in assembly.bindings:
        root_index = reference.object_body_names().index(
            object_binding.spec.root_reference_name
        )
        root_pose = np.concatenate(
            (
                reference.object_body_pos_w()[frames, root_index],
                reference.object_body_quat_w()[frames, root_index],
            ),
            axis=-1,
        )
        articulation_indices = tuple(
            articulation.reference_index
            for articulation in object_binding.spec.articulations
        )
        positions = reference.object_articulation()[frames][
            :, list(articulation_indices)
        ]
        patches.append(
            EntityStatePatch(
                object_binding.entity_name,
                root_pose=root_pose.astype(np.float32),
                root_velocity=zeros6.copy(),
                joint_positions=np.asarray(positions, dtype=np.float32),
                joint_velocities=np.zeros(
                    (row_count, len(articulation_indices)), dtype=np.float32
                ),
                joint_names=object_binding.joint_names,
            )
        )
    return SceneResetRequest(env_ids=env_ids, patches=tuple(patches))


def reference_public_qpos(
    reference: ChordReference,
    assembly: SharpaSceneAssembly,
    layout: Any,
    base_qpos: np.ndarray,
    frame: int,
) -> np.ndarray:
    """Map one reference frame into the public scene qpos layout."""
    qpos = np.asarray(base_qpos, dtype=np.float64).copy()
    if qpos.shape != (layout.nq,):
        raise ValueError(f"base_qpos shape {qpos.shape}; expected ({layout.nq},)")
    for hand in assembly.hand_bindings:
        entity = layout.get_entity(hand.entity_name)
        qpos[list(entity.root_qpos_indices)] = np.concatenate(
            (
                reference.wrist_pos_w(hand.side)[frame],
                reference.wrist_quat_w(hand.side)[frame],
            )
        )
        reference_names = reference.finger_joint_names(hand.side)
        reference_positions = reference.finger_joint_pos(hand.side)[frame]
        joints = {joint.name: joint for joint in entity.joints}
        for name in hand.joint_names:
            joint = joints[name]
            reference_index = reference_names.index(name)
            qpos[list(joint.qpos_indices)] = reference_positions[reference_index]
    for binding in assembly.bindings:
        entity = layout.get_entity(binding.entity_name)
        root_index = reference.object_body_names().index(
            binding.spec.root_reference_name
        )
        qpos[list(entity.root_qpos_indices)] = np.concatenate(
            (
                reference.object_body_pos_w()[frame, root_index],
                reference.object_body_quat_w()[frame, root_index],
            )
        )
        joints = {joint.name: joint for joint in entity.joints}
        for articulation, name in zip(
            binding.spec.articulations, binding.joint_names, strict=True
        ):
            qpos[list(joints[name].qpos_indices)] = reference.object_articulation()[
                frame, articulation.reference_index
            ]
    return qpos


def reset_sharpa_scene(
    backend: Any,
    reference: ChordReference,
    assembly: SharpaSceneAssembly,
    frame: int,
    *,
    env_ids: tuple[int, ...] = (0,),
) -> None:
    """Apply the shared reference patch through UniSim's public reset API."""
    request = reference_state_patch(reference, assembly, frame, env_ids=env_ids)
    backend.reset_entities(request)
