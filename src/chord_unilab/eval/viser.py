"""Compose UniLab's deterministic FlashSAC playback as a Viser evaluation."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def compose_viser_config(
    parquet: Path,
    checkpoint: Path,
    *,
    port: int = 8080,
    show_reference_ghost: bool = True,
) -> DictConfig:
    overrides = [
        "task=sharpa_chord/newton",
        f"env.scene.reference_path={json.dumps(str(parquet))}",
        f"algo.load_run={json.dumps(str(checkpoint))}",
        "training.play_only=true",
        "training.no_play=false",
        "training.play_render_mode=viser",
        "training.play_env_num=1",
        "training.export_onnx=false",
        "env.newton_nconmax=512",
        "+env.auto_reset=true",
        "env.commands.reference.always_reset_to_first_frame=true",
        "env.commands.reference.initial_voc_scale=0.0",
        "env.commands.reference.voc_decay_steps=20",
        "env.commands.reference.curriculum_enabled=false",
        "viser.display_mode=single",
        "viser.max_envs=1",
        f"viser.port={port}",
        f"viser.show_reference_ghost={str(show_reference_ghost).lower()}",
    ]
    config_resource = resources.files("chord_unilab.conf.flashsac")
    with (
        resources.as_file(config_resource) as config_dir,
        initialize_config_dir(version_base="1.3", config_dir=str(config_dir)),
    ):
        return compose(config_name="config", overrides=overrides)


def launch(
    parquet: Path,
    checkpoint: Path,
    *,
    port: int = 8080,
    show_reference_ghost: bool = True,
) -> None:
    if not parquet.exists():
        raise FileNotFoundError(f"CHORD parquet does not exist: {parquet}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FlashSAC checkpoint does not exist: {checkpoint}")
    if not 1 <= port <= 65535:
        raise ValueError("port must be in [1, 65535]")

    from chord_unilab.training.offpolicy import main

    main(
        compose_viser_config(
            parquet,
            checkpoint,
            port=port,
            show_reference_ghost=show_reference_ghost,
        )
    )
