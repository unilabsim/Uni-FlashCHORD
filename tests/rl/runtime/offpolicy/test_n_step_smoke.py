"""Formal off-policy runner smoke for raw n-step replay and real updates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from chord_unilab.rl.algos.flash_sac.learner import FlashSACLearner
from chord_unilab.rl.contracts.env import EnvAlgoCapabilities
from chord_unilab.rl.runtime.offpolicy.double_buffer_runner import (
    DoubleBufferOffPolicyRunner,
)
from chord_unilab.rl.runtime.placement import InferencePlacement, InferenceTransport

_OBS_DIM = 4
_CRITIC_DIM = 6
_ACTION_DIM = 2
_REWARD_NAMES = ("active", "zero")


def _observations(value: float, num_envs: int) -> dict[str, torch.Tensor]:
    obs = torch.full((num_envs, _OBS_DIM), value)
    context = torch.full((num_envs, _CRITIC_DIM - _OBS_DIM), value + 1.0)
    return {"obs": obs, "critic": torch.cat((obs, context), dim=1)}


class _FakeNStepEnv:
    produces_completion_receipts = False

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.obs_groups_spec = {"obs": _OBS_DIM, "critic": _CRITIC_DIM}
        self.action_space = SimpleNamespace(shape=(_ACTION_DIM,))
        self.play_capabilities = SimpleNamespace(supports_physics_state_playback=False)
        self.cfg = SimpleNamespace(ctrl_dt=0.5, max_episode_seconds=100.0)
        self.state = None
        self.step_count = 0
        self.raw_enabled = False

    @property
    def algo_capabilities(self) -> EnvAlgoCapabilities:
        return EnvAlgoCapabilities(
            reward_component_names=_REWARD_NAMES,
            raw_reward_term_names=_REWARD_NAMES,
            critic_context_dim=_CRITIC_DIM - _OBS_DIM,
        )

    def enable_raw_reward_terms(self) -> None:
        self.raw_enabled = True

    def init_state(self):
        self.step_count = 0
        self.state = SimpleNamespace(obs=_observations(0.0, self.num_envs))
        return self.state

    def step(self, actions: torch.Tensor):
        assert actions.shape == (self.num_envs, _ACTION_DIM)
        assert self.raw_enabled
        self.step_count += 1
        stage = int(self.step_count >= 4) + int(self.step_count >= 7)
        raw = torch.stack(
            (
                torch.full((self.num_envs,), float(self.step_count)),
                torch.full((self.num_envs,), 10.0 * self.step_count),
            ),
            dim=1,
        )
        weights = torch.tensor([2.0 + stage, 0.0])
        done = self.step_count == 7
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.full((self.num_envs,), done, dtype=torch.bool)
        self.state = SimpleNamespace(
            obs=_observations(0.0 if done else float(self.step_count), self.num_envs),
            reward=0.5 * (raw * weights).sum(dim=1),
            terminated=terminated,
            truncated=truncated,
            final_observation=(_observations(900.0, self.num_envs) if done else None),
            info={
                "timing": {},
                "raw_reward_terms": raw,
                "reward_components": raw * weights,
                "reward_term_weights": weights,
                "reward_dt": 0.5,
                "curriculum_stage": {"stage": stage},
            },
        )
        return self.state

    def close(self) -> None:
        return None


def _fake_env_factory(num_envs: int, env_cfg_override=None) -> _FakeNStepEnv:
    del env_cfg_override
    return _FakeNStepEnv(num_envs)


class _RecordingFlashSACLearner(FlashSACLearner):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seen_rewards: list[torch.Tensor] = []
        self.seen_obs: list[torch.Tensor] = []
        self.seen_next_obs: list[torch.Tensor] = []
        self.seen_critics: list[torch.Tensor] = []
        self.seen_next_critics: list[torch.Tensor] = []

    def update_critic(self, batch, *, read_metrics: bool = True):
        self.seen_rewards.append(batch["rewards"].detach().clone())
        self.seen_obs.append(batch["obs"].detach().clone())
        self.seen_next_obs.append(batch["next_obs"].detach().clone())
        self.seen_critics.append(batch["critic"].detach().clone())
        self.seen_next_critics.append(batch["next_critic"].detach().clone())
        return super().update_critic(batch, read_metrics=read_metrics)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA DeviceReplayPipeline required"
)
@pytest.mark.parametrize("compact_critic_replay", [False, True])
def test_formal_runner_raw_n_step_completes_multiple_real_updates(
    tmp_path, compact_critic_replay: bool
) -> None:
    learner = _RecordingFlashSACLearner(
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_obs_dim=_CRITIC_DIM,
        device="cuda:0",
        gamma=0.9,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        use_compile=False,
        use_whole_cycle_cuda_graph=False,
        normalize_reward=False,
        n_step=3,
        actor_config={
            "builder": "tests.rl.algos.flash_sac_actor_fixtures:build_scaled_tanh_actor",
            "kwargs": {"hidden_dim": 8},
        },
    )
    runner = DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name="FakeNStep",
        algo_type="flashsac",
        env_factory=_fake_env_factory,
        num_envs=4,
        replay_buffer_n=16,
        batch_size=2,
        learning_starts=2,
        updates_per_step=2,
        policy_frequency=1,
        env_steps_per_sync=1,
        device="cuda:0",
        sim_backend="mujoco",
        n_step=3,
        gamma=0.9,
        reward_relabeling=True,
        compact_critic_replay=compact_critic_replay,
        inference_placement=InferencePlacement(
            mode=InferenceTransport.HOST,
            env_device="cpu",
            ring_device="cpu",
            learner_device="cuda:0",
            collector_tensor_native=False,
            staging_policy="cpu_explicit_staging",
        ),
    )

    try:
        runner.learn(
            max_iterations=12,
            save_interval=0,
            log_dir=str(tmp_path),
            logger_type="no_print",
        )
    finally:
        runner.close()

    assert runner.last_run_summary["status"] == "completed"
    assert runner.last_run_summary["completed_iterations"] == 12
    assert runner.last_run_summary["total_env_steps"] == 48
    assert learner.update_count >= 2
    real_rewards = [
        reward for reward in learner.seen_rewards if bool(reward.abs().sum() > 0)
    ]
    assert len(real_rewards) == 2 * learner.update_count
    assert all(reward.shape == (2,) for reward in real_rewards)
    assert all(bool(torch.isfinite(reward).all()) for reward in real_rewards)
    assert learner.actor_optimizer.state
    assert learner.critic_optimizer.state
    for obs, next_obs, critic, next_critic in zip(
        learner.seen_obs,
        learner.seen_next_obs,
        learner.seen_critics,
        learner.seen_next_critics,
        strict=True,
    ):
        assert critic.shape == next_critic.shape == (2, _CRITIC_DIM)
        torch.testing.assert_close(critic[:, :_OBS_DIM], obs)
        torch.testing.assert_close(next_critic[:, :_OBS_DIM], next_obs)
