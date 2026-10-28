"""Real Newton Manager lifecycle and owner optimizer contract."""

import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf
from unilab.base.config_adapter import create_env
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg

from chord_unilab.assets.reference_state import reference_state_patch
from chord_unilab.eval.viser import compose_viser_config
from chord_unilab.rl.algos.flash_sac.actor_spec import FlashSACActorSpec
from chord_unilab.rl.contracts.probe import probe_env_contract
from chord_unilab.rl.flash_sac_actor import build_actor
from chord_unilab.tasks.sharpa_chord.commands import reference_command
from chord_unilab.tasks.sharpa_chord.env import make_sharpa_chord_env
from chord_unilab.training import ensure_registries
from chord_unilab.training.playback import build_offpolicy_play_env_cfg_override

pytestmark = pytest.mark.newton

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data/assets"
TISSUE = Path(
    os.environ.get(
        "CHORD_TISSUE_PARQUET",
        ASSETS
        / "human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave",
    )
)


def _require_reference(path: Path) -> None:
    parquet_files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    unavailable = not path.exists() or not parquet_files
    if not unavailable:
        unavailable = (
            parquet_files[0]
            .read_bytes()
            .startswith(b"version https://git-lfs.github.com/spec/v1")
        )
    if unavailable:
        message = f"CHORD test data unavailable; run git lfs pull: {path}"
        if os.environ.get("CHORD_TEST_REQUIRE_LFS") == "1":
            pytest.fail(message)
        pytest.skip(message)


def _env(path: Path, num_envs: int):
    _require_reference(path)
    if not torch.cuda.is_available():
        pytest.skip("real Newton smoke requires CUDA")
    owner = OmegaConf.load(
        ROOT / "src/chord_unilab/conf/flashsac/task/sharpa_chord/base.yaml"
    )
    overrides = cast(dict[str, Any], OmegaConf.to_container(owner, resolve=False))
    overrides["env"]["scene"]["reference_path"] = str(path)
    overrides["env"]["rewards"] = overrides["reward"]
    cfg = ManagerBasedRlEnvCfg()
    apply_cfg_overrides(cfg, overrides["env"])
    cfg.validate()
    return make_sharpa_chord_env(cfg, num_envs=num_envs, backend_type="newton")


def test_tissue_multi_env_reset_step_raw_rewards_and_selected_reset() -> None:
    env = _env(TISSUE, 2)
    try:
        env.enable_raw_reward_terms()
        assert env.algo_capabilities.reward_component_names == (
            *env.reward_manager.step_reward_names,
            "total",
            "dense",
        )
        obs, info = env.reset()
        assert obs["obs"].shape == (2, 504)
        assert obs["critic"].shape == (2, 514)
        assert info["log"]["Curriculum/voc/stage"] == 0.0
        assert info["log"]["Curriculum/voc/target_voc_scale"] == 1.0
        assert info["log"]["Curriculum/voc/first_frame_probability"] == 0.1
        state = env.step(torch.zeros(2, 56, device=env.device))
        assert torch.isfinite(state.obs["obs"]).all()
        assert torch.isfinite(state.obs["critic"]).all()
        assert torch.isfinite(state.reward).all()
        assert state.info["raw_reward_terms"].shape == (2, 9)
        expected_components = 0.05 * (
            state.info["raw_reward_terms"] * state.info["reward_term_weights"]
        )
        expected = expected_components.sum(dim=-1)
        torch.testing.assert_close(state.reward, expected)
        components = state.info["reward_components"]
        assert components.shape == (2, 11)
        torch.testing.assert_close(components[:, :9], expected_components)
        torch.testing.assert_close(components[:, 9], expected)
        torch.testing.assert_close(
            components[:, 10], expected - expected_components[:, 6]
        )
        logs = env.publish_reward_log()
        assert logs["reward/total"] == pytest.approx(
            sum(logs[f"reward/{name}"] for name in env.reward_manager.step_reward_names)
        )
        assert logs["reward/dense"] == pytest.approx(
            logs["reward/total"] - logs["reward/termination"]
        )
        curriculum = state.info["curriculum_stage"]["commands"]["reference"]
        assert curriculum == 0
        assert reference_command(cast(Any, env)).target_voc_scale.item() == 1.0

        before = env.backend.get_state_views(("qpos",))["qpos"].clone()
        env.reset(torch.tensor([1], device=env.device))
        after = env.backend.get_state_views(("qpos",))["qpos"]
        torch.testing.assert_close(after[0], before[0])
        assert not torch.equal(after[1], before[1])
    finally:
        env.close()


def test_tissue_selected_reset_clears_only_selected_contacts() -> None:
    env = _env(TISSUE, 2)
    try:
        env.reset()
        command = reference_command(cast(Any, env))
        env.backend.reset_entities(
            reference_state_patch(
                command.reference,
                command.assembly,
                68,
                env_ids=(0, 1),
            )
        )
        command.timestep.fill_(68)
        command.episode_start_frame.fill_(68)
        command.steps_since_reset.fill_(20)
        env.action_manager.clear_action_state()
        env.step(torch.zeros(2, 56, device=env.device))
        before = command.contact_state()
        assert before.active.any()
        env.reset(torch.tensor([1], device=env.device))
        after = command.contact_state()
        torch.testing.assert_close(after.active[0], before.active[0])
        torch.testing.assert_close(after.position_w[0], before.position_w[0])
        torch.testing.assert_close(after.force_w[0], before.force_w[0])
        assert not after.active[1].any()
        torch.testing.assert_close(
            after.position_w[1], torch.zeros_like(after.position_w[1])
        )
        torch.testing.assert_close(after.force_w[1], torch.zeros_like(after.force_w[1]))
    finally:
        env.close()


def test_tissue_single_body_observation_and_physical_step() -> None:
    env = _env(TISSUE, 1)
    try:
        env.enable_raw_reward_terms()
        obs, _ = env.reset()
        assert obs["obs"].shape == (1, 504)
        assert obs["critic"].shape == (1, 514)
        torch.testing.assert_close(
            obs["obs"][:, 300:], torch.zeros(1, 204, device=env.device)
        )
        before = env.backend.get_state_views(("qpos",))["qpos"].clone()
        state = env.step(torch.zeros(1, 56, device=env.device))
        after = env.backend.get_state_views(("qpos",))["qpos"]
        assert torch.isfinite(after).all()
        assert not torch.equal(after, before)
        assert torch.isfinite(state.reward).all()
        assert not state.terminated.any()
        assert not state.truncated.any()
        action = cast(Any, env.action_manager.get_term("residual_hand_pose"))
        assert action.joint_force[:, action.ball_qvel].abs().max() < 0.1
        contact = reference_command(cast(Any, env)).contact_state()
        assert not contact.position_o[~contact.active].any()
        assert not contact.position_b[~contact.active].any()
    finally:
        env.close()


def test_owner_residual_mean_optimizer_group_uses_exact_actor_parameters() -> None:
    owner = OmegaConf.load(
        ROOT / "src/chord_unilab/conf/flashsac/task/sharpa_chord/newton.yaml"
    )
    spec = FlashSACActorSpec.from_config(owner.algo.algo_params.actor_config)
    assert len(spec.optimizer_param_groups) == 1
    group = spec.optimizer_param_groups[0]
    assert group.lr_multiplier == 0.1
    actor = build_actor(obs_dim=504, action_dim=56)
    names = {name for name, _ in actor.named_parameters()}
    assert set(group.parameters) <= names
    assert set(group.parameters) == {"predictor.mean_w.gain", "predictor.mean_bias"}
    assert owner.algo.algo_params.relabel_rewards
    assert owner.algo.algo_params.n_step == 3


def test_tissue_compact_critic_probe_declares_ten_context_columns() -> None:
    _require_reference(TISSUE)
    if not torch.cuda.is_available():
        pytest.skip("real Newton probe requires CUDA")
    owner = OmegaConf.load(
        ROOT / "src/chord_unilab/conf/flashsac/task/sharpa_chord/base.yaml"
    )
    overrides = cast(dict[str, Any], OmegaConf.to_container(owner, resolve=False))
    overrides["env"]["scene"]["reference_path"] = str(TISSUE)
    overrides["env"]["rewards"] = overrides["reward"]

    def factory(num_envs, env_cfg_override):
        cfg = ManagerBasedRlEnvCfg()
        apply_cfg_overrides(cfg, env_cfg_override)
        cfg.validate()
        return make_sharpa_chord_env(cfg, num_envs=num_envs, backend_type="newton")

    probe = probe_env_contract(cast(Any, factory), overrides["env"])
    assert probe.obs_dim == 504
    assert probe.critic_obs_dim == 514
    assert probe.critic_context_dim == 10
    assert probe.action_dim == 56
    assert len(probe.raw_reward_term_names) == 9

    compact_owner = OmegaConf.load(
        ROOT / "src/chord_unilab/conf/flashsac/task/sharpa_chord/newton.yaml"
    )
    assert compact_owner.algo.algo_params.compact_critic_replay


def test_explicit_eval_starts_unassisted_at_frame_zero() -> None:
    _require_reference(TISSUE)
    if not torch.cuda.is_available():
        pytest.skip("real Newton eval requires CUDA")
    cfg = compose_viser_config(TISSUE, ROOT / "unused-checkpoint.pt")
    ensure_registries()
    override = build_offpolicy_play_env_cfg_override("flashsac", cfg, root_dir=ROOT)
    env = create_env(cfg, num_envs=1, env_cfg_override=override)
    try:
        env.reset()
        command = reference_command(cast(Any, env))
        assert command.timestep.item() == 0
        assert command.target_voc_scale.item() == 0.0
        assert command.applied_voc_scale.item() == 0.0
        state = env.step(torch.zeros(1, 56, device=env.device))
        assert command.timestep.item() == 1
        assert command.applied_voc_scale.item() == 0.0
        action = cast(Any, env.action_manager.get_term("residual_hand_pose"))
        assert action.body_force.abs().max().item() == 0.0
        assert not state.terminated.any()
        assert not state.truncated.any()
    finally:
        env.close()
