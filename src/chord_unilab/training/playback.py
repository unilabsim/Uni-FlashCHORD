"""CHORD-owned FlashSAC checkpoint playback and environment adaptation."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.utils.checkpoint import normalize_checkpoint_value, resolve_checkpoint_path
from unilab.utils.sim2sim import policy_load_dim_guard, resolve_sim2sim_config

from chord_unilab.rl.algos.common.actor_factory import build_actor
from chord_unilab.rl.algos.common.normalization import EmpiricalNormalization
from chord_unilab.rl.algos.flash_sac.actor_spec import (
    FlashSACActorSpec,
    build_builtin_actor_semantics,
    build_custom_actor_semantics,
    validate_actor_semantics,
)
from chord_unilab.rl.contracts.observations import get_obs_dims, split_obs_dict
from chord_unilab.training.common import assert_offpolicy_task_choice_matches_algo
from chord_unilab.training.run import get_log_root

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class PlaybackConfig:
    """Configuration needed to bootstrap an interactive playback session."""

    task: str
    load_run: str
    checkpoint: str | None
    action_mode: str
    policy_obs_mode: str
    algo_log_name: str
    log_root: str | None
    num_envs: int = 1
    speed: float = 1.0
    start_paused: bool = False


class OffPolicyPlaybackSession:
    """Direct environment stepping session for FlashSAC actors."""

    def __init__(
        self,
        *,
        env: Any,
        device: str,
        action_mode: str,
        actor: Any | None,
        actor_algo_type: str,
        normalizer: Any | None,
        num_envs: int,
        obs_extractor: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    ) -> None:
        self.env = env
        # A backend may remap a generic CUDA request to a concrete device.
        # ``env.device`` is authoritative once the environment exists.
        self.policy_device = (
            torch.device(device)
            if getattr(env, "device", None) is None
            else torch.device(env.device)
        )
        self.action_mode = action_mode
        self.actor = actor
        self.actor_algo_type = str(actor_algo_type)
        self.normalizer = normalizer
        self.num_envs = int(num_envs)
        self.obs_extractor = obs_extractor
        self.obs: torch.Tensor | None = None
        self.step_count = 0

    def reset(self) -> torch.Tensor:
        if self.env.state is None:
            self.env.init_state()
        reset_result = self.env.reset()
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            raise ValueError(
                f"Unexpected env.reset return format: {type(reset_result)!r}"
            )
        obs_out, _ = reset_result
        self.obs = _to_play_tensor(
            self.obs_extractor(obs_out), device=self.policy_device
        )
        self.step_count = 0
        return self.obs

    def step_once(self) -> torch.Tensor:
        actions = self._build_actions()
        state = self.env.step(actions)
        self.obs = _to_play_tensor(
            self.obs_extractor(state.obs), device=self.policy_device
        )
        self.step_count += 1
        return self.obs

    def advance(self, controls: Any) -> bool:
        if not controls.consume_step_permission():
            return False
        self.step_once()
        return True

    def physics_state(self) -> np.ndarray:
        return self.env.get_physics_state_snapshot()

    @property
    def info(self) -> dict[str, Any]:
        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        return info if isinstance(info, dict) else {}

    def _build_actions(self) -> torch.Tensor:
        if self.obs is None:
            raise RuntimeError("Playback session must be reset before stepping.")
        action_space = self.env.action_space
        action_dim = int(action_space.shape[0])
        if self.action_mode == "policy" and self.actor is not None:
            obs_torch = self.obs.to(device=self.policy_device, dtype=torch.float32)
            if self.normalizer is not None:
                obs_torch = self.normalizer(obs_torch, update=False)
            actions = self.actor.explore(obs_torch, deterministic=True)
            return (
                actions.detach()
                .to(device=self.env.device, dtype=torch.float32)
                .contiguous()
            )
        if self.action_mode == "random":
            low = torch.as_tensor(action_space.low, device=self.env.device).to(
                torch.float32
            )
            high = torch.as_tensor(action_space.high, device=self.env.device).to(
                torch.float32
            )
            return low + (high - low) * torch.rand(
                (self.num_envs, action_dim), device=self.env.device, dtype=torch.float32
            )
        return torch.zeros(
            (self.num_envs, action_dim), dtype=torch.float32, device=self.env.device
        )


def _to_play_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "Off-policy playback observations must be torch tensors; "
            f"received {type(value).__name__}"
        )
    return value.to(device=device, dtype=torch.float32).contiguous()


def default_device(torch_module, preferred: str | None = None) -> str:
    """Resolve runtime device with optional user override."""
    if preferred:
        return preferred
    if torch_module.cuda.is_available():
        return "cuda"
    xpu = getattr(torch_module, "xpu", None)
    xpu_is_available = getattr(xpu, "is_available", None)
    if callable(xpu_is_available) and xpu_is_available():
        return "xpu"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_play_obs_dims(obs_groups_spec: dict[str, int]) -> tuple[int, int]:

    obs_dim, critic_obs_dim = get_obs_dims(obs_groups_spec)
    return int(obs_dim), int(critic_obs_dim)


def extract_play_obs(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:

    obs_out, _ = split_obs_dict(obs_dict)
    return obs_out


def resolve_playback_device(env_device: Any, requested_device: str) -> torch.device:
    """Resolve actor placement from the authoritative TorchEnv device."""
    if env_device is None:
        return torch.device(requested_device)
    return torch.device(env_device)


def _flashsac_actor_kwargs(cfg: DictConfig) -> dict[str, Any]:
    kwargs = {
        "actor_num_blocks": cfg.algo.algo_params.actor_num_blocks,
        "actor_noise_zeta_mu": cfg.algo.algo_params.actor_noise_zeta_mu,
        "actor_noise_zeta_max": cfg.algo.algo_params.actor_noise_zeta_max,
    }
    actor_config = cfg.algo.algo_params.get("actor_config")
    if actor_config is not None:
        kwargs["actor_config"] = actor_config
    return kwargs


def flashsac_play_actor_semantics(
    cfg: Any,
    *,
    obs_dim: int,
    action_dim: int,
    actor: Any | None = None,
) -> dict[str, Any]:
    """Recompute the FlashSAC actor semantics payload expected at play time."""

    actor_config = cfg.algo.algo_params.get("actor_config")
    if actor_config is not None:
        transform_id = getattr(actor, "action_transform_id", None)
        if not isinstance(transform_id, str) or not transform_id:
            raise TypeError(
                "Custom FlashSAC play actor must declare a non-empty string 'action_transform_id'"
            )
        return build_custom_actor_semantics(
            FlashSACActorSpec.from_config(actor_config),
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_transform_id=transform_id,
        )
    return build_builtin_actor_semantics(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_blocks=int(cfg.algo.algo_params.actor_num_blocks),
        hidden_dim=int(cfg.algo.actor_hidden_dim),
        noise_zeta_mu=float(cfg.algo.algo_params.actor_noise_zeta_mu),
        noise_zeta_max=int(cfg.algo.algo_params.actor_noise_zeta_max),
    )


def load_play_actor(
    algo_name: str,
    actor: Any,
    normalizer: Any | None,
    checkpoint: dict[str, Any],
    expected_actor_semantics: dict[str, Any] | None = None,
) -> None:
    """Restore an off-policy play actor and its optional observation normalizer.

    For FlashSAC, ``expected_actor_semantics`` enables fail-closed validation
    of the checkpoint's actor construction/observation/action semantics against
    the owner config before weights are loaded.
    """
    if algo_name != "flashsac":
        raise ValueError(f"Unsupported algo: {algo_name}")
    if expected_actor_semantics is not None:
        validate_actor_semantics(
            checkpoint.get("actor_semantics"), expected_actor_semantics
        )
    actor.load_state_dict(checkpoint["actor"])
    if normalizer is not None and checkpoint.get("obs_normalizer"):
        normalizer.load_state_dict(checkpoint["obs_normalizer"])
        normalizer.eval()


def build_offpolicy_env_cfg_override(
    algo_name: str,
    cfg: DictConfig,
    *,
    root_dir: str | Path,
) -> dict[str, Any] | None:
    """Build the task env override for off-policy training."""

    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)
    return cast(
        dict[str, Any] | None,
        EnvConfigAdapter(
            cfg, root_dir=root_dir, algo_name=algo_name
        ).build_task_env_cfg_override(),
    )


def build_offpolicy_play_env_cfg_override(
    algo_name: str,
    cfg: DictConfig,
    *,
    root_dir: str | Path,
) -> dict[str, Any] | None:
    """Build the off-policy play override, including backend render intent.

    Training and playback deliberately use separate adapters: an IsaacSim
    renderer must be selected before the worker's Kit ``INIT`` handshake, but
    learner/collector environments must stay on the no-rendering experience.
    """

    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)
    return cast(
        dict[str, Any] | None,
        EnvConfigAdapter(
            cfg, root_dir=root_dir, algo_name=algo_name
        ).build_play_env_cfg_override(),
    )


def create_flashsac_playback_session(
    *,
    playback_cfg: PlaybackConfig,
    cfg: Any,
    env_factory: Callable[[int], Any],
    root_dir: str | Path,
    device: str | None,
    algo_name: str = "flashsac",
    log: LogFn = print,
) -> tuple[OffPolicyPlaybackSession, str, str | None]:
    """Create a deterministic FlashSAC playback session."""
    if algo_name != "flashsac":
        raise ValueError(f"Unsupported algo: {algo_name}")

    device_name = default_device(torch, str(device) if device is not None else None)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    obs_dim, _ = resolve_play_obs_dims(env.obs_groups_spec)
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    session_device = resolve_playback_device(getattr(env, "device", None), device_name)
    actor_algo_type = "flashsac"
    actor_kwargs = _flashsac_actor_kwargs(cfg)

    actor = None
    checkpoint_path: str | None = None
    normalizer = None
    if bool(getattr(cfg.algo, "obs_normalization", False)):
        from chord_unilab.rl.algos.common.normalization import EmpiricalNormalization

        normalizer = EmpiricalNormalization(shape=obs_dim, device=session_device)
    if playback_cfg.action_mode == "policy":
        actor = build_actor(
            actor_algo_type,
            obs_dim,
            action_dim,
            cfg.algo.actor_hidden_dim,
            cfg.algo.use_layer_norm,
            str(session_device),
            **actor_kwargs,
        )
        actor.eval()
        algo_log_name = str(cfg.algo.algo_log_name)
        if playback_cfg.log_root is not None:
            task_log_root = (
                get_log_root(
                    Path(root_dir),
                    algo_log_name=algo_log_name,
                    log_root=playback_cfg.log_root,
                )
                / playback_cfg.task
            )
        else:
            # Without an explicit log_root the lookup stays on the repo-rooted
            # ``<root_dir>/logs/<algo>`` tree; the test-only
            # ``UNILAB_TEST_LOG_ROOT`` override is reserved for cfg-driven
            # resolution (``training.resolve_configured_checkpoint_path``).
            task_log_root = (
                Path(root_dir) / "logs" / algo_log_name
            ).absolute() / playback_cfg.task
        resolved_path, resolved_dir = resolve_checkpoint_path(
            task_log_root,
            playback_cfg.load_run,
            checkpoint=normalize_checkpoint_value(playback_cfg.checkpoint),
        )
        checkpoint_path = str(resolved_path) if resolved_path is not None else None
        checkpoint_dir = str(resolved_dir) if resolved_dir is not None else None
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            log(
                f"WARNING: no {algo_name} checkpoint found for "
                f"load_run={playback_cfg.load_run} - falling back to zero actions."
            )
            actor = None
        else:
            resolve_sim2sim_config(
                checkpoint_dir,
                cfg,
                algo_name=algo_name,
                strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=session_device, weights_only=True
            )
            expected_actor_semantics = (
                flashsac_play_actor_semantics(
                    cfg,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    actor=actor,
                )
                if algo_name == "flashsac"
                else None
            )
            with policy_load_dim_guard(
                env_obs_dim=obs_dim,
                env_action_dim=action_dim,
                algo_name=algo_name,
            ):
                load_play_actor(
                    algo_name,
                    actor,
                    normalizer,
                    checkpoint,
                    expected_actor_semantics=expected_actor_semantics,
                )
            log(f"Loading {algo_name} checkpoint: {checkpoint_path}")

    log(f"Action mode: {playback_cfg.action_mode}")
    return (
        OffPolicyPlaybackSession(
            env=env,
            device=device_name,
            action_mode=playback_cfg.action_mode,
            actor=actor,
            actor_algo_type=actor_algo_type,
            normalizer=normalizer,
            num_envs=playback_cfg.num_envs,
            obs_extractor=extract_play_obs,
        ),
        "actor",
        checkpoint_path,
    )
