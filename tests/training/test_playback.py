"""End-to-end CHORD FlashSAC checkpoint playback without UniLab RL imports."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from chord_unilab.rl.algos.common.actor_factory import build_actor
from chord_unilab.rl.algos.flash_sac.actor_spec import build_builtin_actor_semantics
from chord_unilab.training.playback import (
    PlaybackConfig,
    create_flashsac_playback_session,
)


class _Env:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.obs_groups_spec = {"obs": 3, "critic": 5}
        self.action_space = SimpleNamespace(
            shape=(2,), low=[-1.0, -1.0], high=[1.0, 1.0]
        )
        self.state = None
        self.last_action = None

    def init_state(self):
        self.state = SimpleNamespace(obs=self._obs(0.0))
        return self.state

    def _obs(self, value: float) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.full((1, 3), value),
            "critic": torch.full((1, 5), value),
        }

    def reset(self):
        self.state = SimpleNamespace(obs=self._obs(0.0))
        return self.state.obs, {}

    def step(self, actions: torch.Tensor):
        self.last_action = actions.detach().clone()
        self.state = SimpleNamespace(obs=self._obs(1.0))
        return self.state


def _config():
    return OmegaConf.create(
        {
            "algo": {
                "algo_log_name": "flash_sac",
                "actor_hidden_dim": 8,
                "use_layer_norm": False,
                "obs_normalization": False,
                "algo_params": {
                    "actor_num_blocks": 1,
                    "actor_noise_zeta_mu": 2.0,
                    "actor_noise_zeta_max": 4,
                    "actor_config": None,
                },
            },
            "training": {"sim2sim_strict": False},
        }
    )


def test_checkpoint_session_loads_and_steps_through_chord(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = _config()
    actor = build_actor(
        "flashsac", 3, 2, 8, False, "cpu", actor_num_blocks=1, actor_noise_zeta_max=4
    )
    semantics = build_builtin_actor_semantics(
        obs_dim=3,
        action_dim=2,
        num_blocks=1,
        hidden_dim=8,
        noise_zeta_mu=2.0,
        noise_zeta_max=4,
    )
    checkpoint = tmp_path / "model_1.pt"
    torch.save({"actor": actor.state_dict(), "actor_semantics": semantics}, checkpoint)
    monkeypatch.setattr(
        "chord_unilab.training.playback.resolve_sim2sim_config",
        lambda *args, **kwargs: None,
    )
    env = _Env()
    playback_cfg = PlaybackConfig(
        task="SharpaChordFlashSAC",
        load_run=str(checkpoint),
        checkpoint=None,
        action_mode="policy",
        policy_obs_mode="actor",
        algo_log_name="flash_sac",
        log_root=None,
    )

    session, obs_mode, resolved = create_flashsac_playback_session(
        playback_cfg=playback_cfg,
        cfg=cfg,
        env_factory=lambda num_envs: env,
        root_dir=tmp_path,
        device="cpu",
    )
    session.reset()
    session.step_once()

    assert obs_mode == "actor"
    assert resolved == str(checkpoint)
    assert session.step_count == 1
    assert env.last_action is not None
    assert env.last_action.shape == (1, 2)
    assert torch.all(env.state.obs["obs"] == 1.0)
