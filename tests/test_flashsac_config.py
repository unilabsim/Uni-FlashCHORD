"""Installed Hydra discovery and canonical Sharpa FlashSAC owner values."""

import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from chord_unilab.tasks.sharpa_chord.rewards import TERM_NAMES

ROOT = Path(__file__).resolve().parents[1]


def test_installed_owner_resolves_without_searchpath_override() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "chord_unilab.scripts.train_flashsac",
            "task=sharpa_chord/newton",
            'env.scene.reference_path="/tmp/chord-test-parquet"',
            "--cfg",
            "job",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "is not available" not in result.stderr
    config = OmegaConf.create(result.stdout)
    assert config.training.task_name == "SharpaChordFlashSAC"
    assert config.env.scene.reference_path == "/tmp/chord-test-parquet"
    assert (
        config.env.scene._target_
        == "chord_unilab.tasks.sharpa_chord.scene.SharpaSceneCfg"
    )
    assert config.env.newton_njmax == 512
    assert config.env.newton_nconmax == 256
    assert config.algo.save_interval == 2500
    assert config.algo.algo_params.use_whole_cycle_cuda_graph
    assert config.algo.algo_params.amp_dtype == "fp16"
    assert config.training.nan_guard.enabled
    assert config.algo.checkpoint == -1
    assert tuple(config.reward) == TERM_NAMES
    assert config.env.policy_observation_group == "actor"
    assert config.env.critic_observation_group == "critic"
