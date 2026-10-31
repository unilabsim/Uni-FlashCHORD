# pyright: reportAttributeAccessIssue=false
"""CHORD single-environment Viser playback with a reference-pose overlay."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
import torch
import viser
from omegaconf import DictConfig
from unilab.visualization.interactive_playback import PlaybackControls, PlaybackSession
from unilab.visualization.playback_state import (
    PhysicsStateApplier,
    assert_physics_state_playback_supported,
)
from unilab.visualization.viser_scene import MujocoViserScene

from chord_unilab.assets.compose_sharpa_scene import SharpaSceneAssembly
from chord_unilab.assets.reference_state import reference_public_qpos
from chord_unilab.data.reference import ChordReference
from chord_unilab.tasks.sharpa_chord.commands import reference_command
from chord_unilab.tasks.sharpa_chord.scene import scene_from_env


def reference_ghost_model(
    model: mujoco.MjModel, assembly: SharpaSceneAssembly
) -> mujoco.MjModel:
    """Copy a playback model with only hands and objects visible in green."""
    ghost = copy.deepcopy(model)
    object_bodies = {
        body for binding in assembly.bindings for body in binding.bodies.values()
    }
    for geom_index in range(ghost.ngeom):
        body_index = int(ghost.geom_bodyid[geom_index])
        body_name = mujoco.mj_id2name(ghost, mujoco.mjtObj.mjOBJ_BODY, body_index) or ""
        visible = (
            body_name.startswith(("left_hand/", "right_hand/"))
            or body_name in object_bodies
        )
        ghost.geom_group[geom_index] = 0 if visible else 3
        if visible:
            ghost.geom_rgba[geom_index] = (0.12, 0.95, 0.25, 0.28)
    return ghost


class ReferenceGhost:
    """Render one reference trajectory without modifying the physics owner."""

    def __init__(
        self,
        server: viser.ViserServer,
        model: mujoco.MjModel,
        reference: ChordReference,
        assembly: SharpaSceneAssembly,
        layout: Any,
        *,
        visible: bool = True,
    ) -> None:
        self._server = server
        self._reference = reference
        self._assembly = assembly
        self._layout = layout
        self._model = reference_ghost_model(model, assembly)
        self._reference_geoms = self._model.geom_group == 0
        self._visible = True
        self._data = mujoco.MjData(self._model)
        self._base_qpos = self._data.qpos.copy()
        self._scene = MujocoViserScene(
            server,
            self._model,
            name_prefix="/reference/ghost/single",
            render_plane=False,
        )
        self.set_visible(visible)

    def set_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible == self._visible:
            return
        for geom_index in np.flatnonzero(self._reference_geoms):
            handle = self._server.scene.get_handle_by_name(
                f"/reference/ghost/single/geom/{geom_index}"
            )
            if handle is not None:
                handle.visible = visible
        self._visible = visible

    def update(self, frame: int) -> None:
        self._data.qpos[:] = reference_public_qpos(
            self._reference, self._assembly, self._layout, self._base_qpos, frame
        )
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._scene.update(self._data)

    def close(self) -> None:
        self._scene.close()


def _load_playback_model(env: Any) -> mujoco.MjModel:
    model = env.get_playback_model(0)
    if isinstance(model, mujoco.MjModel):
        model = copy.deepcopy(model)
    elif isinstance(model, str):
        if model.lower().endswith(".mjb"):
            model = mujoco.MjModel.from_binary_path(model)
        else:
            model = mujoco.MjModel.from_xml_path(model)
    else:
        raise TypeError(
            f"Expected MuJoCo model or model file path, got {type(model)!r}"
        )
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_group[collision] = 3
    return model


def run_viser_playback(
    session: PlaybackSession,
    *,
    port: int = 8080,
    start_paused: bool = False,
    speed: float = 1.0,
    show_reference_ghost: bool = True,
    entrypoint: str = "CHORD Viser playback",
    log: Callable[[str], None] = print,
) -> None:
    """Serve one policy environment with green reference hands and object."""
    env = session.env
    if int(env.num_envs) != 1:
        raise ValueError(
            f"CHORD Viser eval requires exactly one environment, got {env.num_envs}"
        )
    assert_physics_state_playback_supported(env, entrypoint=entrypoint)
    scene = scene_from_env(env)
    reference = scene.reference
    assembly = scene.assembly

    server = viser.ViserServer(port=int(port))
    server.scene.add_grid(
        "/visual/floor_grid",
        width=2.0,
        height=2.0,
        plane="xy",
        cell_size=0.1,
        section_size=0.5,
        infinite_grid=False,
        plane_opacity=0.0,
    )
    with server.gui.add_folder("Controls"):
        pause_button = server.gui.add_button("Pause / Resume")
        step_button = server.gui.add_button("Step")
        speed_slider: Any = server.gui.add_slider(
            "Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0
        )
        show_reference_ghost_control = server.gui.add_checkbox(
            "Show Reference Ghost", initial_value=bool(show_reference_ghost)
        )

    controls = PlaybackControls(paused=bool(start_paused), speed=float(speed))
    speed_slider.value = controls.speed
    model = _load_playback_model(env)
    data = mujoco.MjData(model)
    applier = PhysicsStateApplier(env, model, env_index=0)
    policy_scene = MujocoViserScene(server, model, name_prefix="/mujoco/single")
    ghost = ReferenceGhost(
        server,
        model,
        reference,
        assembly,
        env.backend.get_scene_layout(),
        visible=show_reference_ghost,
    )

    @show_reference_ghost_control.on_update
    def on_reference_ghost_toggle(event: Any) -> None:
        del event
        ghost.set_visible(bool(show_reference_ghost_control.value))

    @pause_button.on_click
    def on_pause_click(event: Any) -> None:
        del event
        status = "paused" if controls.toggle_pause() else "resumed"
        log(status)

    @step_button.on_click
    def on_step_click(event: Any) -> None:
        del event
        if not controls.paused:
            controls.pause()
        controls.request_single_step()

    session.reset()
    log(f"Server running at http://localhost:{server.get_port()}")
    log("1 environment loaded with reference ghost enabled.")

    try:
        with torch.inference_mode():
            while True:
                started = time.perf_counter()
                controls.set_speed(float(speed_slider.value))
                session.advance(controls)
                applier.apply(session.physics_state()[0], data)
                policy_scene.update(data)
                ghost.update(int(reference_command(env).frames[0].item()))

                elapsed = time.perf_counter() - started
                remaining = controls.target_dt(float(env.cfg.ctrl_dt)) - elapsed
                if remaining > 0.0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        log("Viser playback shutting down.")
    finally:
        policy_scene.close()
        ghost.close()


def run_viser_playback_from_cfg(
    session: PlaybackSession,
    cfg: DictConfig,
    *,
    entrypoint: str = "CHORD Viser playback",
    start_paused: bool = False,
    speed: float = 1.0,
    log: Callable[[str], None] = print,
) -> None:
    """Run single-environment reference-ghost playback from the Viser config."""
    run_viser_playback(
        session,
        port=int(cfg.viser.port),
        start_paused=start_paused,
        speed=speed,
        show_reference_ghost=bool(cfg.viser.show_reference_ghost),
        entrypoint=entrypoint,
        log=log,
    )


__all__ = [
    "ReferenceGhost",
    "reference_ghost_model",
    "run_viser_playback",
    "run_viser_playback_from_cfg",
]
