"""Full and compact critic replay equivalence on the real device pipeline."""

from __future__ import annotations

import pytest
import torch

from chord_unilab.rl.runtime.ipc.replay_buffer import ReplayBuffer
from chord_unilab.rl.runtime.ipc.replay_pipelines.device import DeviceReplayPipeline
from chord_unilab.rl.runtime.offpolicy.n_step import NStepAccumulator

_OBS_DIM = 2
_CONTEXT_DIM = 2


def _transition(
    step: int, *, compact: bool, done: bool = False
) -> dict[str, torch.Tensor]:
    obs = torch.tensor([[float(step), step + 0.25]])
    next_obs = obs if done else obs + 1.0
    context = torch.tensor([[100.0 + step, 200.0 + step]])
    next_context = context if done else context + 1.0
    critic = context if compact else torch.cat((obs, context), dim=1)
    next_critic = (
        next_context if compact else torch.cat((next_obs, next_context), dim=1)
    )
    return {
        "obs": obs,
        "actions": torch.tensor([[step + 0.5]]),
        "rewards": torch.tensor([step + 1.0]),
        "next_obs": next_obs,
        "dones": torch.tensor([float(done)]),
        "truncated": torch.tensor([float(done)]),
        "critic": critic,
        "next_critic": next_critic,
        "reward_terms": torch.tensor([[step + 1.0, 2.0 * (step + 1)]]),
    }


def _sample(n_step: int, compact: bool) -> dict[str, torch.Tensor]:
    replay = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=1,
        critic_dim=_CONTEXT_DIM if compact else _OBS_DIM + _CONTEXT_DIM,
        reward_term_dim=2,
        compact_critic=compact,
        device="cuda:0",
        ingress_slot_rows=1,
        ingress_device="cpu",
        require_completion_receipt=False,
    )
    pipeline = DeviceReplayPipeline(
        replay, device="cuda:0", sample_count=4, base_seed=17
    )
    accumulator = NStepAccumulator(n_step=n_step, gamma=0.5)
    try:
        for step in range(6):
            transition = _transition(step, compact=compact, done=step == 1)
            stage = int(step >= 3)
            committed = accumulator.add(transition, stage=stage)
            for window in committed:
                assert replay.add_batch(**window)
        pipeline.progress(wait=True)
        pipeline.read_committed_fields(("obs",), start_ptr=0)
        pipeline.start_prepare(tick_id=1, sample_count=4)
        batch = pipeline.sample_large_batch(tick_id=1, sample_count=4)
        return {name: value.clone() for name, value in batch.items()}
    finally:
        pipeline.close()
        replay.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA DeviceReplayPipeline required"
)
@pytest.mark.parametrize("n_step", [1, 3])
def test_full_and_compact_replay_sample_identical_values(n_step: int) -> None:
    full = _sample(n_step, compact=False)
    compact = _sample(n_step, compact=True)

    assert full.keys() == compact.keys()
    assert set(full) == {
        "obs",
        "next_obs",
        "actions",
        "rewards",
        "dones",
        "truncated",
        "critic",
        "next_critic",
        "reward_terms",
        "bootstrap_discount",
    }
    for name in full:
        torch.testing.assert_close(
            compact[name],
            full[name],
            msg=f"replay field {name}: {compact[name]} != {full[name]}",
        )
