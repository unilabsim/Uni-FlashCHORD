"""Sharpa CHORD environment lifecycle and reset ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.cpu_runtime import apply_cpu_confinement
from unilab.base.entity import EntityCfg
from unilab.base.torch_env import TorchEnvState
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from unilab.managers import ResetOwner, ResetOwnerCfg

from chord_unilab.rl.contracts.env import EnvAlgoCapabilities

from .commands import reference_command
from .scene import SharpaSceneCfg


class SharpaChordEnv(ManagerBasedRlEnv):
    """Publish CHORD replay signals without changing UniLab reward evaluation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = cast(ManagerBasedRlEnvCfg, self.cfg)
        self._reward_term_names = self.reward_manager.step_reward_names
        weights = []
        for name in self._reward_term_names:
            term_cfg = cfg.rewards[name]
            if term_cfg is None:
                raise ValueError(f"Active Sharpa CHORD reward term {name!r} has no cfg")
            weights.append(float(term_cfg.weight))
        if any(weight == 0.0 for weight in weights):
            raise ValueError(
                "Sharpa CHORD raw reward recovery requires non-zero weights"
            )
        self._reward_term_weights = torch.tensor(
            weights, dtype=torch.float32, device=self.device
        )
        self._reward_dt = self.step_dt if cfg.scale_rewards_by_dt else 1.0
        self._raw_reward_terms_enabled = False
        self._reward_metric_names = (*self._reward_term_names, "total", "dense")

    def enable_raw_reward_terms(self) -> None:
        """Enable CHORD-owned raw replay-term publication."""
        self._raw_reward_terms_enabled = True

    def step(self, actions: torch.Tensor) -> TorchEnvState:
        state = super().step(actions)
        components = state.info.get("reward_components")
        if isinstance(components, torch.Tensor):
            components = components * self._reward_dt
            total = components.sum(dim=1, keepdim=True)
            termination = self._reward_term_names.index("termination")
            dense = total - components[:, termination : termination + 1]
            state.info["reward_components"] = torch.cat(
                (components, total, dense), dim=1
            )
        if self._raw_reward_terms_enabled:
            state.info["raw_reward_terms"] = (
                self.reward_manager.step_rewards / self._reward_term_weights
            )
            state.info["reward_term_weights"] = self._reward_term_weights
            state.info["reward_dt"] = self._reward_dt
            state.info["curriculum_stage"] = {
                "manager": {},
                "commands": {
                    "reference": reference_command(cast(Any, self)).curriculum_stage,
                },
            }
        return state

    def publish_reward_log(self) -> dict[str, float]:
        entries = {
            name: value * self._reward_dt
            for name, value in super().publish_reward_log().items()
        }
        total = sum(entries.values())
        entries["reward/total"] = total
        entries["reward/dense"] = total - entries["reward/termination"]
        return entries

    @property
    def algo_capabilities(self) -> Any:
        capabilities = super().algo_capabilities
        cfg = cast(ManagerBasedRlEnvCfg, self.cfg)
        obs_dim = self._mapped_obs_dims["obs"]
        critic_dim = self._mapped_obs_dims.get("critic", obs_dim)
        actor_terms = tuple(
            self.observation_manager.active_terms[cfg.policy_observation_group]
        )
        critic_group = cfg.critic_observation_group
        critic_terms = (
            tuple(self.observation_manager.active_terms[critic_group])
            if critic_group is not None
            else ()
        )
        has_actor_prefix = (
            len(critic_terms) > len(actor_terms)
            and critic_terms[: len(actor_terms)] == actor_terms
        )
        context_dim = critic_dim - obs_dim if has_actor_prefix else 0
        return EnvAlgoCapabilities(
            action_low=capabilities.action_low,
            action_high=capabilities.action_high,
            joint_names=capabilities.joint_names,
            reward_component_names=self._reward_metric_names,
            raw_reward_term_names=self._reward_term_names,
            critic_context_dim=context_dim if context_dim > 0 else None,
        )


def make_sharpa_chord_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "newton",
) -> SharpaChordEnv:
    """Construct the CHORD task through UniLab's public backend factory."""
    if backend_type != "newton":
        raise ValueError(
            f"Sharpa CHORD requires the newton backend, got {backend_type!r}"
        )
    if not isinstance(cfg, ManagerBasedRlEnvCfg) or not isinstance(
        cfg.scene, SharpaSceneCfg
    ):
        raise TypeError(
            "Sharpa CHORD requires ManagerBasedRlEnvCfg with SharpaSceneCfg"
        )
    cfg.validate()
    apply_cpu_confinement(cfg.cpu_ids)
    left_hand = cfg.scene.entities["left_hand"]
    if not isinstance(left_hand, EntityCfg) or left_hand.root_body_name is None:
        raise ValueError("Sharpa CHORD requires a left_hand root body")
    base_name = left_hand.root_body_name
    backend_kwargs = env_backend_kwargs(cfg)
    backend_kwargs["newton_deferred_contact_publication"] = (
        cfg.scene.deferred_contact_publication
    )
    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=True,
        base_name=base_name,
        **backend_kwargs,
    )
    try:
        return SharpaChordEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise


@dataclass(kw_only=True)
class ReferenceResetOwnerCfg(ResetOwnerCfg):
    command_name: str = "reference"
    owns_observation_reset: bool = False


class ReferenceResetOwner(ResetOwner):
    """Reset physics before commit and persistent action/metric state after it.

    SimBackend capabilities: none directly; ReferenceCommand performs the public
    selected entity reset before action and episode metric state are cleared.
    """

    def reset_transaction(self, env_ids: torch.Tensor) -> dict[str, float]:
        reference_command(self._env, self.cfg.command_name).reset(env_ids)
        return {}

    def reset_committed(self, env_ids: torch.Tensor) -> dict[str, float]:
        self._env.action_manager.clear_action_state(env_ids)
        self._env.metrics_manager.clear_episode_state(env_ids)
        return {}
