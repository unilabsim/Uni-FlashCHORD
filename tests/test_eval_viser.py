"""CHORD eval delegates deterministic playback to UniLab's Viser path."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from chord_unilab.eval.viser import compose_viser_config, launch
from chord_unilab.eval.viser_playback import (
    ReferenceGhost,
    _load_playback_model,
    reference_ghost_model,
)


def test_viser_eval_config_loads_checkpoint_without_training() -> None:
    config = compose_viser_config(
        Path("/tmp/chord-eval-parquet"),
        Path("/tmp/chord-eval-model.pt"),
        port=8091,
    )
    assert config.training.task_name == "SharpaChordFlashSAC"
    assert config.env.scene.reference_path == "/tmp/chord-eval-parquet"
    assert config.algo.load_run == "/tmp/chord-eval-model.pt"
    assert config.training.play_only
    assert not config.training.no_play
    assert config.training.play_render_mode == "viser"
    assert config.training.play_env_num == 1
    assert not config.training.export_onnx
    assert config.env.newton_nconmax == 512
    assert config.env.auto_reset
    assert config.env.commands.reference.always_reset_to_first_frame
    assert config.env.commands.reference.initial_voc_scale == 0.0
    assert config.env.commands.reference.voc_decay_steps == 20
    assert not config.env.commands.reference.curriculum_enabled
    assert config.env.terminations.wrist_position is not None
    assert config.env.terminations.object_position is not None
    assert config.env.terminations.object_orientation is not None
    assert config.env.terminations.reference_end.time_out
    assert "pause_on_done" not in config.viser
    assert config.viser.display_mode == "single"
    assert config.viser.max_envs == 1
    assert config.viser.port == 8091
    assert config.viser.show_reference_ghost


def test_viser_eval_uses_one_visible_environment() -> None:
    config = compose_viser_config(
        Path("/tmp/chord-eval-parquet"),
        Path("/tmp/chord-eval-model.pt"),
        show_reference_ghost=False,
    )
    assert config.training.play_env_num == 1
    assert config.viser.display_mode == "single"
    assert config.viser.max_envs == 1
    assert not config.viser.show_reference_ghost


def test_viser_eval_rejects_missing_checkpoint(tmp_path: Path) -> None:
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        launch(parquet, tmp_path / "missing.pt")


def test_reference_ghost_model_only_recolors_hands_and_object() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="left_hand/palm"><geom type="box" size="0.1 0.1 0.1"/></body>
            <body name="tissue/body"><geom type="box" size="0.1 0.1 0.1"/></body>
            <body name="support/base"><geom type="box" size="0.1 0.1 0.1"/></body>
          </worldbody>
        </mujoco>
        """
    )
    assembly = SimpleNamespace(
        bindings=[SimpleNamespace(bodies={"root": "tissue/body"})]
    )
    original_groups = model.geom_group.copy()
    original_rgba = model.geom_rgba.copy()

    ghost = reference_ghost_model(model, cast(Any, assembly))

    assert ghost is not model
    assert ghost.geom_group.tolist() == [0, 0, 3]
    np.testing.assert_allclose(
        ghost.geom_rgba[:2], np.tile((0.12, 0.95, 0.25, 0.28), (2, 1))
    )
    np.testing.assert_array_equal(model.geom_group, original_groups)
    np.testing.assert_array_equal(model.geom_rgba, original_rgba)


def test_reference_ghost_visibility_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="right_hand/palm"><geom type="box" size="0.1 0.1 0.1"/></body>
            <body name="mug/body"><geom type="box" size="0.1 0.1 0.1"/></body>
            <body name="support/base"><geom type="box" size="0.1 0.1 0.1"/></body>
          </worldbody>
        </mujoco>
        """
    )
    assembly = SimpleNamespace(bindings=[SimpleNamespace(bodies={"root": "mug/body"})])
    handles = {
        f"/reference/ghost/single/geom/{index}": SimpleNamespace(visible=True)
        for index in range(2)
    }
    server = SimpleNamespace(scene=SimpleNamespace(get_handle_by_name=handles.get))
    monkeypatch.setattr(
        "chord_unilab.eval.viser_playback.MujocoViserScene",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    ghost = ReferenceGhost(
        cast(Any, server),
        model,
        cast(Any, None),
        cast(Any, assembly),
        cast(Any, None),
        visible=False,
    )

    assert [handle.visible for handle in handles.values()] == [False, False]
    assert ghost._model.geom_group.tolist() == [0, 0, 3]
    ghost.set_visible(True)
    assert [handle.visible for handle in handles.values()] == [True, True]


def test_playback_model_hides_only_collision_geoms() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="object">
              <geom name="visual" group="1" contype="0" conaffinity="0" type="box" size="0.1 0.1 0.1"/>
              <geom name="collision" group="1" contype="1" conaffinity="1" type="box" size="0.1 0.1 0.1"/>
              <geom name="support" group="2" contype="0" conaffinity="0" type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    original_groups = model.geom_group.copy()
    env = SimpleNamespace(get_playback_model=lambda _index: model)

    playback = _load_playback_model(env)

    assert playback is not model
    assert playback.geom_group.tolist() == [1, 3, 2]
    np.testing.assert_array_equal(model.geom_group, original_groups)
