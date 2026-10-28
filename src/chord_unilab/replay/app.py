"""Interactive Viser application for production parquet replay."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import unisim
from omegaconf import DictConfig
from unisim.backend.base import RenderClosedError

from chord_unilab.assets.compose_sharpa_scene import (
    SharpaSceneAssembly,
    assemble_sharpa_scene,
)
from chord_unilab.data.loader import load_chord_reference
from chord_unilab.data.reference import ChordReference
from chord_unilab.data.reference_assets import load_reference_manifest
from chord_unilab.replay.runner import ReplayMode, ReplayRunner

_MODES = ("kinematic", "voc0", "voc1")


class ReplayApplication:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        if cfg.viewer.backend != "viser":
            raise ValueError(f"replay viewer must be viser, got {cfg.viewer.backend}")
        if cfg.mode not in _MODES:
            raise ValueError(f"replay mode must be one of {_MODES}, got {cfg.mode}")
        if int(cfg.sim.substeps) <= 0:
            raise ValueError("sim.substeps must be positive")
        self.manifest = load_reference_manifest()
        self._playing = not bool(cfg.viewer.start_paused)
        self._requested_mode: str | None = None
        self._requested_sequence: str | None = None
        self._requested_reset = False
        self._requested_frame: int | None = None
        self.runner: ReplayRunner
        self.backend: Any
        self.reference: ChordReference
        self.assembly: SharpaSceneAssembly
        self._build_backend(str(cfg.task.parquet), str(cfg.mode))
        self._install_controls()

    @property
    def frame(self) -> int:
        return self._frame

    def _build_backend(self, parquet: str, mode: str) -> None:
        if getattr(self, "runner", None) is not None:
            self.runner.close()
        if getattr(self, "backend", None) is not None:
            self.backend.close()
        self.reference = load_chord_reference(
            Path(parquet).expanduser(),
            control_fps=float(self.cfg.sim.control_fps),
            motion_speed=float(self.cfg.task.motion_speed),
        ).frame_window(
            int(self.cfg.task.motion_start_frame), int(self.cfg.task.motion_end_frame)
        )
        self.assembly = assemble_sharpa_scene(self.reference)
        sim_dt = 1.0 / (float(self.cfg.sim.control_fps) * int(self.cfg.sim.substeps))
        self.backend = unisim.create_backend(
            "newton",
            scene=self.assembly.scene_cfg,
            num_envs=1,
            sim_dt=sim_dt,
            newton_device="cuda:0",
            newton_use_cuda_graph=False,
            njmax=16384,
            nconmax=4096,
        )
        self.backend.materialize()
        self.backend.bind_renderer(
            renderer="viser",
            viser_kwargs={
                "port": int(self.cfg.viewer.port),
                "label": f"CHORD {self.reference.metadata.sequence_id}",
                "verbose": False,
            },
        )
        self.runner = ReplayRunner(
            self.backend,
            self.reference,
            self.assembly,
            mode=cast(ReplayMode, mode),
            substeps=int(self.cfg.sim.substeps),
            record=self.cfg.record.output is not None,
        )
        self.runner.reset(0)
        self._frame = 0

    def _install_controls(self) -> None:
        server = self.backend.get_viser_server()
        gui = server.gui
        registered = tuple(entry.sequence_id for entry in self.manifest.sequences)
        current = self.reference.metadata.sequence_id
        self._dataset = gui.add_dropdown(
            "Dataset",
            options=registered + ("custom",),
            initial_value=current if current in registered else "custom",
        )
        self._mode = gui.add_dropdown(
            "Mode", options=_MODES, initial_value=self.runner.mode
        )
        self._play = gui.add_button("Pause" if self._playing else "Play")
        self._restart = gui.add_button("Restart")
        self._speed = gui.add_dropdown(
            "Speed", options=("0.25", "0.5", "1.0", "2.0"), initial_value="1.0"
        )
        self._slider = gui.add_slider(
            "Frame", min=0, max=self.reference.num_frames - 1, step=1, initial_value=0
        )
        self._status = gui.add_markdown(self._status_text())

        @self._dataset.on_update
        def _request_dataset(_event) -> None:
            if self._dataset.value != "custom":
                self._requested_sequence = self._dataset.value

        @self._mode.on_update
        def _request_mode(_event) -> None:
            self._requested_mode = self._mode.value

        @self._play.on_click
        def _toggle_play(_event) -> None:
            if self._frame >= self.reference.num_frames:
                self._requested_reset = True
            self._playing = not self._playing
            self._play.label = "Pause" if self._playing else "Play"

        @self._restart.on_click
        def _request_restart(_event) -> None:
            self._requested_reset = True

        @self._slider.on_update
        def _request_frame(_event) -> None:
            if not self._playing:
                self._requested_frame = int(self._slider.value)

    def _status_text(self) -> str:
        return (
            f"**{self.reference.metadata.sequence_id}** | `{self.runner.mode}` | "
            f"frame {self._frame}/{self.reference.num_frames} | "
            + (
                "physics"
                if self.runner.mode != "kinematic"
                else "reference only; no physics"
            )
        )

    def _reset(self) -> None:
        self.runner.reset(0)
        self._frame = 0
        self._requested_reset = False
        self._requested_frame = None
        self._playing = False
        self._play.label = "Play"

    def _change_mode(self, mode: str) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown replay mode: {mode}")
        self.runner.close()
        self.runner = ReplayRunner(
            self.backend,
            self.reference,
            self.assembly,
            mode=cast(ReplayMode, mode),
            substeps=int(self.cfg.sim.substeps),
            record=self.cfg.record.output is not None,
        )
        self._requested_mode = None
        self._reset()

    def _seek(self, target: int) -> None:
        target = int(np.clip(target, 0, self.reference.num_frames - 1))
        self.runner.reset(0)
        self._frame = 0
        while self._frame < target:
            self.runner.step_frame(self._frame)
            self._frame += 1
        self._requested_frame = None
        self._playing = False
        self._play.label = "Play"

    def _save_record(self) -> None:
        output = self.cfg.record.output
        if output is not None:
            self.runner.save_npz(str(output))

    def run(self) -> None:
        try:
            while True:
                started = time.perf_counter()
                if self._requested_sequence is not None:
                    sequence = self._requested_sequence
                    self._requested_sequence = None
                    entry = self.manifest.sequence(sequence)
                    self._build_backend(str(entry.parquet_path), self.runner.mode)
                    self._install_controls()
                if self._requested_mode is not None:
                    self._change_mode(self._requested_mode)
                if self._requested_reset:
                    self._reset()
                if self._requested_frame is not None:
                    self._seek(self._requested_frame)
                if self._playing and self._frame < self.reference.num_frames:
                    self.runner.step_frame(self._frame)
                    self._frame += 1
                    if self._frame >= self.reference.num_frames:
                        self._playing = False
                        self._play.label = "Play"
                        self._save_record()
                self._slider.value = min(self._frame, self.reference.num_frames - 1)
                self._status.content = self._status_text()
                self.backend.render()
                elapsed = time.perf_counter() - started
                delay = 1.0 / float(self.cfg.sim.control_fps) / float(self._speed.value)
                time.sleep(max(0.0, delay - elapsed))
        except (KeyboardInterrupt, RenderClosedError):
            pass
        finally:
            if self.runner is not None:
                self.runner.close()
            if self.backend is not None:
                self.backend.close()


def launch(cfg: DictConfig) -> None:
    ReplayApplication(cfg).run()
