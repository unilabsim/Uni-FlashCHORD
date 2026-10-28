"""Shared MJCF authoring helpers for exact flash_chord actuator pairs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence

import numpy as np


def append_position_velocity_actuators(
    parent: ET.Element,
    *,
    name: str,
    joint: str,
    kp: float,
    kd: float,
) -> None:
    """Append one exact flash_chord position + velocity actuator pair.

    The source stack emits two MuJoCo actuators per driven scalar DOF: an
    affine position actuator with ``kp`` and an affine velocity actuator with
    ``kd``. Keeping the pair explicit preserves the compiled ``nu`` count,
    actuator order, independent ctrl channels, and later VOC gain-scaling
    semantics; a combined ``<position kp kv>`` only reproduces the special
    case whose velocity target is zero.
    """
    ET.SubElement(
        parent,
        "position",
        {
            "name": f"{name}_pos",
            "joint": joint,
            "kp": f"{kp:.9g}",
        },
    )
    ET.SubElement(
        parent,
        "velocity",
        {
            "name": f"{name}_vel",
            "joint": joint,
            "kv": f"{kd:.9g}",
        },
    )


def position_velocity_control(
    actuator_names: Sequence[str],
    joint_names: Sequence[str],
    position_targets: np.ndarray,
    velocity_targets: np.ndarray | None = None,
) -> np.ndarray:
    """Expand per-joint targets into exact paired-actuator ctrl columns.

    ``joint_names`` may be either entity-qualified public names or local names
    when the actuator names are entity-qualified. The returned array follows the
    backend actuator order; position columns receive ``position_targets`` and
    velocity columns receive ``velocity_targets`` (default zero). This is the
    owner-side mapping from flash_chord's per-DOF target buffers to the
    compiled model's two independent MuJoCo ctrl channels.
    """
    position = np.asarray(position_targets, dtype=np.float32)
    expected = (position.shape[0], len(joint_names))
    if position.ndim != 2 or position.shape != expected:
        raise ValueError(
            f"position targets must have shape {expected}, got {position.shape}"
        )
    velocity = (
        np.zeros(expected, dtype=np.float32)
        if velocity_targets is None
        else np.asarray(velocity_targets, dtype=np.float32)
    )
    if velocity.shape != expected:
        raise ValueError(
            f"velocity targets must have shape {expected}, got {velocity.shape}"
        )
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("actuator targets must be finite")

    def paired_index(joint: str, suffix: str) -> int:
        exact = f"{joint}{suffix}"
        matches = [
            index
            for index, name in enumerate(actuator_names)
            if name == exact or name.rsplit("/", 1)[-1] == exact
        ]
        if len(matches) != 1:
            raise ValueError(
                f"joint {joint!r} must own exactly one {suffix.strip('_')} actuator; "
                f"got {[actuator_names[i] for i in matches]}"
            )
        return matches[0]

    output = np.zeros((expected[0], len(actuator_names)), dtype=np.float32)
    for column, joint in enumerate(joint_names):
        output[:, paired_index(joint, "_pos")] = position[:, column]
        output[:, paired_index(joint, "_vel")] = velocity[:, column]
    return output
