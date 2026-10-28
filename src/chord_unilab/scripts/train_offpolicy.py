"""Compatibility imports for the CHORD-owned FlashSAC implementation."""

from chord_unilab.training.offpolicy import (
    build_failure_summary,
    build_offpolicy_env_cfg_override,
    build_offpolicy_play_env_cfg_override,
    build_runner,
    enable_faulthandler,
    main,
    play_offpolicy,
    resolve_train_resume_checkpoint,
)

__all__ = [
    "build_failure_summary",
    "build_offpolicy_env_cfg_override",
    "build_offpolicy_play_env_cfg_override",
    "build_runner",
    "enable_faulthandler",
    "main",
    "play_offpolicy",
    "resolve_train_resume_checkpoint",
]
