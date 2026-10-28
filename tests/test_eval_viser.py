"""CHORD eval delegates deterministic playback to UniLab's Viser path."""

from pathlib import Path

import pytest

from chord_unilab.eval.viser import compose_viser_config, launch


def test_viser_eval_config_loads_checkpoint_without_training() -> None:
    config = compose_viser_config(
        Path("/tmp/chord-eval-parquet"),
        Path("/tmp/chord-eval-model.pt"),
        port=8091,
        num_envs=2,
    )
    assert config.training.task_name == "SharpaChordFlashSAC"
    assert config.env.scene.reference_path == "/tmp/chord-eval-parquet"
    assert config.algo.load_run == "/tmp/chord-eval-model.pt"
    assert config.training.play_only
    assert not config.training.no_play
    assert config.training.play_render_mode == "viser"
    assert config.training.play_env_num == 2
    assert not config.training.export_onnx
    assert config.env.auto_reset
    assert config.env.commands.reference.always_reset_to_first_frame
    assert config.env.commands.reference.initial_voc_scale == 0.0
    assert config.env.commands.reference.voc_decay_steps == 0
    assert not config.env.commands.reference.curriculum_enabled
    assert config.env.terminations.wrist_position is None
    assert config.env.terminations.object_position is None
    assert config.env.terminations.object_orientation is None
    assert config.env.terminations.reference_end.time_out
    assert "pause_on_done" not in config.viser
    assert config.viser.display_mode == "all"
    assert config.viser.max_envs == 2
    assert config.viser.port == 8091


def test_viser_eval_defaults_to_four_visible_envs() -> None:
    config = compose_viser_config(
        Path("/tmp/chord-eval-parquet"), Path("/tmp/chord-eval-model.pt")
    )
    assert config.training.play_env_num == 4
    assert config.viser.display_mode == "all"
    assert config.viser.max_envs == 4


def test_viser_eval_rejects_missing_checkpoint(tmp_path: Path) -> None:
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        launch(parquet, tmp_path / "missing.pt")
