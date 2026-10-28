"""Probe env dims through an injected env factory (issue #1479)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from chord_unilab.rl.contracts.env import (
    EnvFactory,
    get_critic_context_dim,
    get_produces_completion_receipts,
    get_raw_reward_term_names,
)
from chord_unilab.rl.contracts.observations import get_obs_dims as _get_obs_dims


@dataclass(frozen=True)
class EnvContractProbe:
    """Cold-path env contract probe result."""

    obs_dim: int
    action_dim: int
    critic_obs_dim: int
    produces_completion_receipts: bool
    raw_reward_term_names: tuple[str, ...] = ()
    critic_context_dim: int = 0


def probe_env_contract(
    env_factory: EnvFactory,
    env_cfg_override: Mapping[str, Any] | None = None,
) -> EnvContractProbe:
    """Probe one env instance for dims and completion-receipt semantics.

    Builds a one-env probe through the injected factory, reads the env
    contract, and closes the probe again. ``env_cfg_override`` is an opaque
    mapping forwarded to the factory. ``produces_completion_receipts``
    defaults to ``True`` (fail closed) when the env does not declare it.
    """
    env = env_factory(1, env_cfg_override)
    try:
        obs_dim, critic_dim = _get_obs_dims(dict(env.obs_groups_spec))
        action_shape = env.action_space.shape
        assert action_shape is not None
        action_dim = action_shape[0]
        produces_completion_receipts = get_produces_completion_receipts(env)
        raw_reward_term_names = get_raw_reward_term_names(env)
        critic_context_dim = get_critic_context_dim(env)
        if critic_context_dim:
            if critic_dim != obs_dim + critic_context_dim:
                raise ValueError(
                    "critic concat declaration requires critic_dim=obs_dim+context_dim, "
                    f"got {critic_dim} != {obs_dim}+{critic_context_dim}"
                )
            state = env.init_state()
            actor_obs = state.obs.get("obs")
            critic_obs = state.obs.get("critic", actor_obs)
            if not isinstance(actor_obs, torch.Tensor) or not isinstance(
                critic_obs, torch.Tensor
            ):
                raise TypeError("critic context probe requires tensor observations")
            if critic_obs.shape != actor_obs.shape[:1] + (critic_dim,):
                raise ValueError(
                    "critic context probe received an invalid critic shape"
                )
            if not torch.equal(critic_obs[:, :obs_dim], actor_obs):
                raise ValueError(
                    "critic observation prefix must equal actor observation"
                )
    finally:
        env.close()
    return EnvContractProbe(
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_obs_dim=critic_dim,
        produces_completion_receipts=produces_completion_receipts,
        raw_reward_term_names=tuple(raw_reward_term_names),
        critic_context_dim=critic_context_dim,
    )


__all__ = ["EnvContractProbe", "probe_env_contract"]
