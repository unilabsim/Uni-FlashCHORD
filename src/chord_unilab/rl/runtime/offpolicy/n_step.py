"""Vectorized n-step pending windows at the collector replay boundary."""

from __future__ import annotations

from typing import Any

import torch

_TRANSITION_FIELDS = (
    "obs",
    "actions",
    "rewards",
    "next_obs",
    "dones",
    "truncated",
    "bootstrap_discount",
    "critic",
    "next_critic",
    "reward_terms",
)


def _clone_transition(transition: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(transition)
    for name in _TRANSITION_FIELDS:
        value = cloned.get(name)
        if isinstance(value, torch.Tensor):
            cloned[name] = value.clone()
    return cloned


def _subset_transition(
    transition: dict[str, Any],
    mask: torch.Tensor,
    *,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    reward_terms: torch.Tensor | None,
) -> dict[str, Any]:
    output = dict(transition)
    indices = mask.nonzero(as_tuple=False).flatten()
    for name in _TRANSITION_FIELDS:
        value = transition.get(name)
        if isinstance(value, torch.Tensor):
            output[name] = value.index_select(0, indices)
    output["obs"] = obs.index_select(0, indices)
    output["actions"] = actions.index_select(0, indices)
    output["rewards"] = rewards.index_select(0, indices)
    if reward_terms is not None:
        output["reward_terms"] = reward_terms.index_select(0, indices)
    output.pop("terminal_mask", None)
    output.pop("terminal_next_obs", None)
    output.pop("terminal_next_critic", None)
    return output


def validate_n_step(value: Any) -> int:
    """Require a non-bool positive integer at every public n-step boundary."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"n_step must be a non-bool integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"n_step must be positive, got {value}")
    return int(value)


def _aggregate_window(pending: list[dict[str, Any]], gamma: float) -> dict[str, Any]:
    first = pending[0]
    latest = pending[-1]
    aggregated = dict(latest)
    rewards = latest["rewards"]
    reward_terms = latest.get("reward_terms")
    dones = latest["dones"]
    truncated = latest["truncated"]
    next_obs = latest["next_obs"]
    next_critic = latest["next_critic"]
    window_length = torch.ones_like(dones)
    for prior in reversed(pending[:-1]):
        done = prior["dones"] > 0.5
        continuing = (~done).to(dtype=rewards.dtype)
        window_length = window_length + continuing
        rewards = prior["rewards"] + gamma * rewards * continuing
        if reward_terms is not None:
            reward_terms = (
                prior["reward_terms"] + gamma * reward_terms * continuing[:, None]
            )
        dones = torch.where(done, prior["dones"], dones)
        truncated = torch.where(done, prior["truncated"], truncated)
        next_obs = torch.where(done[:, None], prior["next_obs"], next_obs)
        next_critic = torch.where(done[:, None], prior["next_critic"], next_critic)
    aggregated["obs"] = first["obs"]
    aggregated["actions"] = first["actions"]
    aggregated["critic"] = first["critic"]
    aggregated["rewards"] = rewards
    aggregated["dones"] = dones
    aggregated["truncated"] = truncated
    aggregated["next_obs"] = next_obs
    aggregated["next_critic"] = next_critic
    aggregated["bootstrap_discount"] = gamma**window_length
    if reward_terms is not None:
        aggregated["reward_terms"] = reward_terms
    return aggregated


class NStepAccumulator:
    """Commit only the oldest complete window with per-row bootstrap discounts."""

    def __init__(self, n_step: int, gamma: float) -> None:
        self.n_step = validate_n_step(n_step)
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.gamma = float(gamma)
        self._pending: list[dict[str, Any]] = []
        self._stage: Any = None

    def flush_pending(self) -> None:
        """Discard in-flight starts without touching committed replay entries."""
        self._pending.clear()

    def add(
        self, transition: dict[str, Any], *, stage: Any = None
    ) -> list[dict[str, Any]]:
        """Add one vector interaction and return committed rows in start order."""
        if self.n_step == 1:
            return [transition]
        if self._stage is not None and stage != self._stage:
            self.flush_pending()
        self._stage = stage

        current = _clone_transition(transition)
        self._pending.append(current)
        if len(self._pending) < self.n_step:
            return []

        aggregated = _aggregate_window(self._pending, self.gamma)
        self._pending.pop(0)
        commit = torch.ones_like(current["dones"], dtype=torch.bool)
        return [
            _subset_transition(
                aggregated,
                commit,
                obs=aggregated["obs"],
                actions=aggregated["actions"],
                rewards=aggregated["rewards"],
                reward_terms=aggregated.get("reward_terms"),
            )
        ]


__all__ = ["NStepAccumulator", "validate_n_step"]
