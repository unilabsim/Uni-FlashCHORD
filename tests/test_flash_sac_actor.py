"""Contract and FlashChord-semantics tests for the custom FlashSAC actor."""

from __future__ import annotations

import json
import math
import multiprocessing
from typing import Any, cast

import pytest
import torch

from chord_unilab.rl.algos.flash_sac.actor_spec import (
    FlashSACActorSpec,
    build_custom_actor,
)
from chord_unilab.rl.algos.flash_sac.layers import (
    UnitBatchNorm,
    UnitLinear,
    UnitRMSNorm,
)
from chord_unilab.rl.flash_sac_actor import (
    FlashChordActor,
    GainedUnitLinear,
    ResidualTanhPolicy,
)

BUILDER = "chord_unilab.rl.flash_sac_actor:build_actor"
OBS_DIM = 4
ACTION_DIM = 3


def make_actor(**kwargs: Any) -> FlashChordActor:
    options: dict[str, Any] = {
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "hidden_dim": 8,
        "num_blocks": 2,
        "expansion": 3,
        "device": "cpu",
    }
    options.update(kwargs)
    return FlashChordActor(**options)


def build_actor(**kwargs: Any) -> FlashChordActor:
    spec = FlashSACActorSpec(builder=BUILDER, kwargs=kwargs)
    actor = build_custom_actor(
        spec, obs_dim=OBS_DIM, action_dim=ACTION_DIM, device="cpu"
    )
    assert isinstance(actor, FlashChordActor)
    return actor


def spawn_build_child(
    kwargs_json: str,
    obs_dim: int,
    action_dim: int,
    result_queue: Any,
) -> None:
    spec = FlashSACActorSpec(builder=BUILDER, kwargs=json.loads(kwargs_json))
    actor = cast(
        FlashChordActor,
        build_custom_actor(spec, obs_dim=obs_dim, action_dim=action_dim, device="cpu"),
    )
    observations = torch.zeros(3, obs_dim)
    deterministic = actor.explore(observations, deterministic=True)
    exported = actor.as_export_module()(observations)
    result_queue.put(
        {
            "export_matches": bool(torch.allclose(exported, deterministic)),
            "repeat_count": int(actor._repeat_count),
            "action_shape": tuple(deterministic.shape),
        }
    )


def test_official_builder_and_initial_distribution() -> None:
    actor = build_actor(hidden_dim=8, num_blocks=2, expansion=3)
    observations = torch.randn(5, OBS_DIM)

    mean, std = actor.get_mean_and_std(observations, training=False)

    torch.testing.assert_close(mean, torch.zeros_like(mean), rtol=0.0, atol=0.0)
    torch.testing.assert_close(std, torch.full_like(std, 0.1))
    assert len(actor.encoder) == 2
    first_block = cast(Any, actor.encoder[0])
    assert first_block.w1.w.weight.shape == (24, 8)
    assert first_block.w2.w.weight.shape == (8, 24)
    assert actor.action_transform_id == "tanh"
    assert actor._repeat_count.item() == 0
    assert actor._repeat_target.item() == 0
    assert actor.inference_startup_memory_categories(5) == {
        "persistent_exploration_scratch": 5 * ACTION_DIM * 4 + 8
    }


def test_initial_log_std_bias_matches_gt_projection_formula() -> None:
    initial_std = 0.1
    log_std_min = -10.0
    log_std_max = -1.8971199848858813
    policy = ResidualTanhPolicy(8, ACTION_DIM, initial_std, log_std_min, log_std_max)
    normalized = (
        2.0 * (math.log(initial_std) - log_std_min) / (log_std_max - log_std_min) - 1.0
    )

    torch.testing.assert_close(
        policy.std_bias,
        torch.full((ACTION_DIM,), math.atanh(normalized)),
    )
    _mean, std = policy.get_mean_and_std(torch.randn(4, 8))
    torch.testing.assert_close(std, torch.full_like(std, initial_std))


def test_gained_unit_linear_matches_controlled_projection() -> None:
    layer = GainedUnitLinear(3, 2)
    weight = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    gain = torch.tensor([0.25, -0.75])
    value = torch.tensor([[2.0, 1.0, -4.0], [-1.0, 0.5, 3.0]])
    with torch.no_grad():
        layer.projection.w.weight.copy_(weight)
        layer.gain.copy_(gain)

    torch.testing.assert_close(layer(value), (value @ weight.T) * gain)


def test_forward_action_and_log_prob_match_tanh_gaussian() -> None:
    actor = make_actor()
    observations = torch.randn(6, OBS_DIM)

    actions, info = actor(observations, training=True)
    mean = info["mean"]
    std = info["std"]
    raw_action = torch.atanh(actions)
    noise = (raw_action - mean) / std
    normal_log_prob = -0.5 * (
        noise.square() + 2.0 * std.log() + math.log(2.0 * math.pi)
    )
    tanh_log_det = 2.0 * (
        math.log(2.0) - raw_action - torch.nn.functional.softplus(-2.0 * raw_action)
    )
    expected = (normal_log_prob - tanh_log_det).sum(dim=-1)

    assert actions.shape == (6, ACTION_DIM)
    assert actions.abs().max() < 1.0
    assert "aux_losses" not in info
    torch.testing.assert_close(info["log_prob"], expected)


def test_deterministic_export_does_not_advance_exploration() -> None:
    actor = make_actor()
    observations = torch.randn(4, OBS_DIM)
    actor.explore(observations)
    count = actor._repeat_count.clone()
    target = actor._repeat_target.clone()
    noise = actor._noise.clone()

    mean, _std = actor.get_mean_and_std(observations, training=False)
    deterministic = actor.explore(observations, dones=torch.ones(4), deterministic=True)
    exported = actor.as_export_module()(observations)

    torch.testing.assert_close(deterministic, torch.tanh(mean))
    torch.testing.assert_close(exported, deterministic)
    torch.testing.assert_close(actor._repeat_count, count)
    torch.testing.assert_close(actor._repeat_target, target)
    torch.testing.assert_close(actor._noise, noise)


def test_global_repeat_ignores_dones_and_recomputes_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = make_actor()

    def controlled_distribution(
        observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not training
        scale = observations[:, :1]
        return observations[:, :ACTION_DIM], 0.1 + scale

    monkeypatch.setattr(actor, "get_mean_and_std", controlled_distribution)
    candidates = iter(torch.full((2, ACTION_DIM), value) for value in (0.5, 1.5, 2.5))
    monkeypatch.setattr(torch, "randn_like", lambda _value: next(candidates))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.7))

    first_obs = torch.zeros(2, OBS_DIM)
    second_obs = torch.full((2, OBS_DIM), 0.2)
    third_obs = torch.full((2, OBS_DIM), 0.4)
    first = actor.explore(first_obs, dones=torch.zeros(2))
    first_noise = actor._noise.clone()
    second = actor.explore(second_obs, dones=torch.ones(2))
    third = actor.explore(third_obs, dones=torch.tensor([1.0, 0.0]))

    torch.testing.assert_close(first, torch.tanh(0.1 * 0.5 + first_obs[:, :ACTION_DIM]))
    torch.testing.assert_close(
        second, torch.tanh(second_obs[:, :ACTION_DIM] + 0.3 * first_noise)
    )
    torch.testing.assert_close(third, torch.tanh(third_obs[:, :ACTION_DIM] + 0.5 * 2.5))
    assert actor._repeat_count.shape == ()
    assert actor._repeat_target.shape == ()
    assert actor._repeat_count.item() == 1
    assert actor._repeat_target.item() == 2


def test_zeta_cdf_is_normalized_inverse_square_distribution() -> None:
    actor = make_actor(noise_zeta_max=16, noise_zeta_mu=2.0)
    lengths = torch.arange(1, 17, dtype=torch.float32)
    probabilities = lengths.pow(-2.0)
    expected_cdf = torch.cumsum(probabilities / probabilities.sum(), dim=0)

    torch.testing.assert_close(actor.zeta_cdf, expected_cdf)


def test_normalize_parameters_projects_all_unilab_layers() -> None:
    actor = make_actor()
    with torch.no_grad():
        for module in actor.modules():
            if isinstance(module, UnitLinear):
                module.w.weight.mul_(3.0)
            elif isinstance(module, UnitBatchNorm):
                module.weight.mul_(2.0)
                module.bias.add_(0.5)
            elif isinstance(module, UnitRMSNorm):
                module.weight.mul_(4.0)

    actor.normalize_parameters()

    for module in actor.modules():
        if isinstance(module, UnitLinear):
            torch.testing.assert_close(
                module.w.weight.norm(dim=-1), torch.ones(module.w.weight.shape[0])
            )
        elif isinstance(module, UnitBatchNorm):
            squared_norm = (module.weight.square() + module.bias.square()).sum(dim=-1)
            torch.testing.assert_close(
                squared_norm,
                torch.tensor(float(module.weight.shape[-1])),
            )
        elif isinstance(module, UnitRMSNorm):
            torch.testing.assert_close(
                module.weight.square().sum(),
                torch.tensor(float(module.weight.shape[-1])),
            )


def test_checkpoint_excludes_batch_noise_and_restores_safely() -> None:
    actor = make_actor()
    actor.explore(torch.randn(3, OBS_DIM))
    state = actor.state_dict()

    assert "_repeat_count" in state
    assert "_repeat_target" in state
    assert "_noise" not in state

    restored = make_actor()
    restored.load_state_dict(state)
    assert restored._repeat_count.item() == 0
    assert restored._repeat_target.item() == 0
    assert restored._noise.numel() == 0
    assert restored.explore(torch.randn(5, OBS_DIM)).shape == (5, ACTION_DIM)


def test_spawn_subprocess_builds_official_actor() -> None:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=spawn_build_child,
        args=(
            json.dumps({"hidden_dim": 8, "num_blocks": 1}),
            OBS_DIM,
            ACTION_DIM,
            result_queue,
        ),
    )
    process.start()
    process.join(timeout=120)

    assert process.exitcode == 0
    result = result_queue.get(timeout=10)
    assert result == {
        "export_matches": True,
        "repeat_count": 0,
        "action_shape": (3, ACTION_DIM),
    }
