"""Floating dual-Sharpa FlashChord Manager-Based task."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from .env import make_sharpa_chord_env

TASK_NAME = "SharpaChordFlashSAC"

registry.register_env_config(TASK_NAME, ManagerBasedRlEnvCfg)
registry.register_env(TASK_NAME, make_sharpa_chord_env, sim_backend="newton")

__all__ = ["TASK_NAME"]
