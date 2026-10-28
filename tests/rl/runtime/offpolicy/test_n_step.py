"""N-step pending-window and current-weight replay relabeling contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from chord_unilab.rl.runtime.offpolicy.double_buffer_runner import (
    DoubleBufferOffPolicyRunner,
)
from chord_unilab.rl.runtime.offpolicy.n_step import NStepAccumulator, validate_n_step
from chord_unilab.rl.runtime.offpolicy.runner import environment_step_budget
from chord_unilab.rl.runtime.offpolicy.worker import curriculum_stage_key


def _transition(
    step: int,
    *,
    done: bool = False,
    timeout: bool = False,
    final_obs: float | None = None,
) -> dict[str, Any]:
    next_obs = torch.tensor([[100.0 + step]])
    if done:
        assert final_obs is not None
        next_obs = torch.tensor([[final_obs]])
    return {
        "obs": torch.tensor([[float(step)]]),
        "actions": torch.tensor([[step + 0.25]]),
        "rewards": torch.tensor([float(step + 1)]),
        "next_obs": next_obs,
        "dones": torch.tensor([float(done)]),
        "truncated": torch.tensor([float(timeout)]),
        "critic": torch.tensor([[step + 0.5]]),
        "next_critic": next_obs + 0.5,
        "reward_terms": torch.tensor([[float(step + 1), 10.0 * (step + 1)]]),
    }


def test_early_done_commits_complete_windows_without_shortening() -> None:
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)
    assert accumulator.add(_transition(0), stage="stage-0") == []
    assert (
        accumulator.add(
            _transition(1, done=True, timeout=True, final_obs=900.0),
            stage="stage-0",
        )
        == []
    )

    first = accumulator.add(_transition(2), stage="stage-0")[0]
    torch.testing.assert_close(first["obs"], torch.tensor([[0.0]]))
    torch.testing.assert_close(first["critic"], torch.tensor([[0.5]]))
    torch.testing.assert_close(first["rewards"], torch.tensor([2.0]))
    torch.testing.assert_close(first["next_obs"], torch.tensor([[900.0]]))
    torch.testing.assert_close(first["next_critic"], torch.tensor([[900.5]]))
    torch.testing.assert_close(first["dones"], torch.ones(1))
    torch.testing.assert_close(first["truncated"], torch.ones(1))
    torch.testing.assert_close(first["bootstrap_discount"], torch.tensor([0.25]))

    second = accumulator.add(_transition(3), stage="stage-0")[0]
    torch.testing.assert_close(second["obs"], torch.tensor([[1.0]]))
    torch.testing.assert_close(second["critic"], torch.tensor([[1.5]]))
    torch.testing.assert_close(second["rewards"], torch.tensor([2.0]))
    torch.testing.assert_close(second["next_obs"], torch.tensor([[900.0]]))
    torch.testing.assert_close(second["next_critic"], torch.tensor([[900.5]]))
    torch.testing.assert_close(second["bootstrap_discount"], torch.tensor([0.25]))

    new_episode = accumulator.add(_transition(4), stage="stage-0")[0]
    torch.testing.assert_close(new_episode["obs"], torch.tensor([[2.0]]))
    torch.testing.assert_close(new_episode["critic"], torch.tensor([[2.5]]))
    torch.testing.assert_close(new_episode["rewards"], torch.tensor([6.25]))
    torch.testing.assert_close(new_episode["next_obs"], torch.tensor([[104.0]]))
    torch.testing.assert_close(new_episode["next_critic"], torch.tensor([[104.5]]))
    torch.testing.assert_close(new_episode["dones"], torch.zeros(1))
    torch.testing.assert_close(new_episode["bootstrap_discount"], torch.tensor([0.125]))


def test_consecutive_done_windows_keep_earliest_terminal_boundary() -> None:
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)
    accumulator.add(_transition(0), stage=0)
    accumulator.add(_transition(1, done=True, timeout=True, final_obs=900.0), stage=0)

    first = accumulator.add(
        _transition(2, done=True, timeout=True, final_obs=901.0), stage=0
    )[0]
    second = accumulator.add(_transition(3), stage=0)[0]
    third = accumulator.add(_transition(4), stage=0)[0]

    for output, start in zip((first, second, third), (0.0, 1.0, 2.0), strict=True):
        torch.testing.assert_close(output["obs"], torch.tensor([[start]]))
        torch.testing.assert_close(output["critic"], torch.tensor([[start + 0.5]]))
        torch.testing.assert_close(output["dones"], torch.ones(1))
        torch.testing.assert_close(output["truncated"], torch.ones(1))
    torch.testing.assert_close(first["rewards"], torch.tensor([2.0]))
    torch.testing.assert_close(second["rewards"], torch.tensor([2.0]))
    torch.testing.assert_close(third["rewards"], torch.tensor([3.0]))
    torch.testing.assert_close(first["next_obs"], torch.tensor([[900.0]]))
    torch.testing.assert_close(second["next_obs"], torch.tensor([[900.0]]))
    torch.testing.assert_close(third["next_obs"], torch.tensor([[901.0]]))
    torch.testing.assert_close(first["next_critic"], torch.tensor([[900.5]]))
    torch.testing.assert_close(second["next_critic"], torch.tensor([[900.5]]))
    torch.testing.assert_close(third["next_critic"], torch.tensor([[901.5]]))
    torch.testing.assert_close(first["bootstrap_discount"], torch.tensor([0.25]))
    torch.testing.assert_close(second["bootstrap_discount"], torch.tensor([0.5]))
    torch.testing.assert_close(third["bootstrap_discount"], torch.tensor([0.25]))


def test_full_window_ending_done_preserves_final_obs_timeout_and_raw_discounts() -> (
    None
):
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)
    accumulator.add(_transition(0), stage="stage-0")
    accumulator.add(_transition(1), stage="stage-0")

    outputs = accumulator.add(
        _transition(2, done=True, timeout=True, final_obs=900.0),
        stage="stage-0",
    )

    assert len(outputs) == 1
    output = outputs[0]
    torch.testing.assert_close(output["obs"], torch.tensor([[0.0]]))
    torch.testing.assert_close(output["critic"], torch.tensor([[0.5]]))
    torch.testing.assert_close(output["rewards"], torch.tensor([2.75]))
    torch.testing.assert_close(output["reward_terms"], torch.tensor([[2.75, 27.5]]))
    torch.testing.assert_close(output["next_obs"], torch.tensor([[900.0]]))
    torch.testing.assert_close(output["next_critic"], torch.tensor([[900.5]]))
    torch.testing.assert_close(output["dones"], torch.ones(1))
    torch.testing.assert_close(output["truncated"], torch.ones(1))
    torch.testing.assert_close(output["bootstrap_discount"], torch.tensor([0.125]))


def test_full_window_uses_configured_discount_series() -> None:
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)
    accumulator.add(_transition(0), stage=0)
    accumulator.add(_transition(1), stage=0)

    output = accumulator.add(_transition(2), stage=0)[0]

    torch.testing.assert_close(output["obs"], torch.tensor([[0.0]]))
    torch.testing.assert_close(output["critic"], torch.tensor([[0.5]]))
    torch.testing.assert_close(output["rewards"], torch.tensor([2.75]))
    torch.testing.assert_close(output["next_obs"], torch.tensor([[102.0]]))
    torch.testing.assert_close(output["dones"], torch.zeros(1))
    torch.testing.assert_close(output["bootstrap_discount"], torch.tensor([0.125]))


def test_stage_change_drops_only_pending_windows() -> None:
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)
    accumulator.add(_transition(0), stage=0)
    committed_before_change = accumulator.add(_transition(1), stage=0)

    assert accumulator.add(_transition(2), stage=1) == []
    assert accumulator.add(_transition(3), stage=1) == []
    output = accumulator.add(_transition(4), stage=1)[0]

    assert committed_before_change == []
    torch.testing.assert_close(output["obs"], torch.tensor([[2.0]]))
    torch.testing.assert_close(
        output["rewards"], torch.tensor([3.0 + 0.5 * 4.0 + 0.25 * 5.0])
    )


def test_only_explicit_stage_change_flushes_pending() -> None:
    term = SimpleNamespace(
        curriculum_stage=0,
        dynamic_scalar=torch.tensor(1.0),
        update_curriculum=lambda: None,
    )

    def stage():
        return curriculum_stage_key({"reference": term.curriculum_stage})

    accumulator = NStepAccumulator(n_step=3, gamma=0.5)
    accumulator.add(_transition(0), stage=stage())

    term.dynamic_scalar.fill_(2.0)
    stage_value = stage()
    assert accumulator.add(_transition(1), stage=stage_value) == []
    term.curriculum_stage = 1
    stage_value = stage()
    assert accumulator.add(_transition(2), stage=stage_value) == []
    assert accumulator.add(_transition(3), stage=stage_value) == []

    output = accumulator.add(_transition(4), stage=stage_value)[0]
    torch.testing.assert_close(output["obs"], torch.tensor([[2.0]]))


def test_default_one_step_transition_is_unchanged() -> None:
    transition = _transition(4)
    assert NStepAccumulator(n_step=1, gamma=0.99).add(transition, stage=0) == [
        transition
    ]


def test_raw_relabel_uses_current_weights_dt_and_keeps_zero_weight_columns() -> None:
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.reward_relabeling = True
    runner.raw_reward_term_names = ("active", "currently_zero")
    runner._curriculum_signal = {"weights": [2.0, 0.0], "dt": 0.05, "stage": 4}
    batch = {
        "rewards": torch.full((2,), 999.0),
        "reward_terms": torch.tensor([[3.0, 100.0], [4.0, 200.0]]),
    }

    runner._relabel_rewards(batch)

    torch.testing.assert_close(batch["rewards"], torch.tensor([0.3, 0.4]))
    torch.testing.assert_close(
        batch["reward_terms"][:, 1], torch.tensor([100.0, 200.0])
    )


def test_scalar_relabel_opt_out_leaves_rewards_unchanged() -> None:
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.reward_relabeling = False
    batch = {"rewards": torch.tensor([1.0, 2.0])}

    runner._relabel_rewards(batch)

    torch.testing.assert_close(batch["rewards"], torch.tensor([1.0, 2.0]))


def test_reward_normalizer_signal_is_one_step_not_n_step_return() -> None:
    calls = []
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.learner = SimpleNamespace(
        update_reward_stats=lambda rewards, dones: calls.append((rewards, dones))
    )
    rewards = torch.tensor([2.0, 3.0])
    dones = torch.tensor([0.0, 1.0])

    runner._handle_collector_signal(
        {"one_step_reward": {"rewards": rewards, "dones": dones}}
    )

    assert calls == [(rewards, dones)]


def test_fresh_run_budget_includes_warmup_collections() -> None:
    total = environment_step_budget(61_036, num_envs=4_096)

    assert total == 250_003_456
    assert environment_step_budget(100, num_envs=4_096) == 409_600


@pytest.mark.parametrize("value", [2.5, True, "3"])
def test_n_step_validation_rejects_non_integer_and_bool(value: Any) -> None:
    with pytest.raises(TypeError, match="non-bool integer"):
        validate_n_step(value)


@pytest.mark.parametrize("value", [0, -1])
def test_n_step_validation_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        validate_n_step(value)
