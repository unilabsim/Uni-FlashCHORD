"""Off-policy learner-owned cold-path preparation contracts."""

from __future__ import annotations

import copy
import random
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from chord_unilab.rl.algos.flash_sac.learner import FlashSACLearner
from chord_unilab.rl.runtime.offpolicy.warmup import (
    CudaRngStateApi,
    OffPolicyWarmupContext,
    capture_rng_state,
    restore_rng_state,
)


def _context(
    *,
    obs_dim: int = 4,
    action_dim: int = 2,
    batch_size: int = 3,
) -> OffPolicyWarmupContext:
    return OffPolicyWarmupContext(
        inference_observations=torch.zeros(batch_size, obs_dim + action_dim),
        inference_dones=torch.zeros(batch_size),
        batch_size=batch_size,
        updates_per_step=2,
        policy_frequency=1,
        target_frequency=1,
        policy_before_critic=False,
    )


def _assert_equivalent(left: Any, right: Any, path: str = "state") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and torch.equal(left, right), path
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            _assert_equivalent(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list | tuple):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, values in enumerate(zip(left, right, strict=True)):
            _assert_equivalent(values[0], values[1], f"{path}[{index}]")
        return
    assert left == right, path


def test_rng_state_capture_restores_mps_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = torch.tensor([1, 2, 3])

    class FakeMPS:
        def get_rng_state(self) -> torch.Tensor:
            return current.clone()

        def set_rng_state(self, state: torch.Tensor) -> None:
            nonlocal current
            current = state.clone()

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch, "mps", FakeMPS())
    state = capture_rng_state("mps")
    current = torch.tensor([9, 9, 9])

    restore_rng_state(state)

    torch.testing.assert_close(current, torch.tensor([1, 2, 3]))


def test_rng_state_api_table_skips_cuda_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda *_args, **_kwargs: calls.append("get") or torch.empty(0),
    )

    assert CudaRngStateApi().state_available(torch.device("cuda:0")) is False
    assert calls == []


def test_rng_state_api_table_dispatches_initialized_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.device]] = []
    state = torch.tensor([4, 5])

    def fake_get(device: torch.device) -> torch.Tensor:
        calls.append(("get", device))
        return state

    def fake_set(value: torch.Tensor, device: torch.device) -> None:
        assert value is state
        calls.append(("set", device))

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", fake_get)
    monkeypatch.setattr(torch.cuda, "set_rng_state", fake_set)
    api = CudaRngStateApi()
    device = torch.device("cuda:3")

    captured = api.get(device)
    api.set(captured, device)

    assert calls == [("get", device), ("set", device)]


def test_flashsac_whole_cycle_warmup_captures_representative_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = FlashSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        use_compile=False,
    )
    calls: list[dict[str, Any]] = []
    original_state = copy.deepcopy(learner.get_state_dict())

    def capture_graph(large_batch: dict[str, torch.Tensor], **kwargs: Any) -> None:
        calls.append({"batch": large_batch, "kwargs": kwargs})

    monkeypatch.setattr(learner, "_compile_full_update_cycle", True)
    monkeypatch.setattr(learner, "_ensure_update_cycle_graph", capture_graph)
    learner.prepare_for_collection(_context(batch_size=4))

    assert len(calls) == 1
    assert calls[0]["kwargs"] == {
        "updates_per_step": 2,
        "policy_frequency": 1,
        "target_frequency": 1,
        "policy_before_critic": False,
    }
    assert {key: tuple(value.shape) for key, value in calls[0]["batch"].items()} == {
        "obs": (8, 4),
        "actions": (8, 2),
        "rewards": (8,),
        "next_obs": (8, 4),
        "dones": (8,),
        "truncated": (8,),
        "critic": (8, 6),
        "next_critic": (8, 6),
    }
    _assert_equivalent(original_state, learner.get_state_dict())


def test_runner_prepare_uses_custom_learner_and_runtime_hooks() -> None:
    from chord_unilab.rl.runtime.offpolicy.double_buffer_runner import (
        DoubleBufferOffPolicyRunner,
    )

    calls: list[str] = []
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.device = "cpu"
    runner.algo_type = "custom_runtime_sac"
    runner.obs_normalization = False
    runner.obs_dim = 4
    runner.batch_size = 2
    runner.updates_per_step = 1
    runner.policy_frequency = 1
    runner.target_frequency = 1
    runner.policy_before_critic = False
    runner.learner = type(
        "Learner",
        (),
        {
            "actor": object(),
            "prepare_for_collection": staticmethod(
                lambda context: calls.append("learner")
            ),
        },
    )()
    runner.learner_prepare_hook = lambda learner, context: calls.append("runtime")

    class Pipeline:
        def warmup(self) -> None:
            calls.append("replay_pipeline")

    runner._prepare_learner(
        inference_observations=torch.zeros(2, 6),
        inference_dones=torch.zeros(2),
        replay_pipeline=Pipeline(),
    )

    assert calls == ["replay_pipeline", "learner", "runtime"]


def test_actor_adapter_warmup_hook_is_preferred_and_rng_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chord_unilab.rl.runtime.offpolicy.actor_adapter as actor_adapter_module
    from chord_unilab.rl.runtime.offpolicy.double_buffer_runner import (
        DoubleBufferOffPolicyRunner,
    )

    calls: list[int] = []

    def warmup(actor, observations, dones, context):
        del actor, dones, context
        calls.append(1)
        return torch.rand(observations.shape[0], 3)

    monkeypatch.setitem(
        actor_adapter_module._ADAPTERS,
        "warm_custom_sac",
        actor_adapter_module.OffPolicyActorAdapter(
            algo_type="warm_custom_sac",
            sample_actions=lambda *_: (_ for _ in ()).throw(AssertionError()),
            warmup_actions=warmup,
        ),
    )

    class Actor:
        pass

    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.device = "cpu"
    runner.obs_dim = 4
    runner.obs_normalization = False
    runner.algo_type = "warm_custom_sac"
    runner.learner = SimpleNamespace(actor=Actor())
    before_cpu = torch.random.get_rng_state()
    before_python = random.getstate()

    runner._warm_representative_actor(_context())

    assert calls == [1]
    torch.testing.assert_close(torch.random.get_rng_state(), before_cpu)
    assert random.getstate() == before_python


def test_device_replay_pipeline_warmup_uses_scratch_gather_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chord_unilab.rl.runtime.ipc.replay_pipelines.device import DeviceReplayPipeline

    pipeline = object.__new__(DeviceReplayPipeline)
    pipeline._closed = False
    pipeline._main_thread_submission = True
    pipeline._learner_thread_id = threading.get_ident()
    pipeline._device = torch.device("cpu")
    pipeline._capacity = 32
    pipeline._sample_count = 8
    pipeline._base_seed = 99
    pipeline._cold = 1
    pipeline._gpu_packed = {
        0: torch.zeros(8, 12),
        1: torch.empty(8, 12),
    }
    gathered: list[tuple[int, int]] = []

    def gather_rows(*, visible_size: int, slot: int, gen: torch.Generator) -> None:
        gathered.append((visible_size, slot))
        assert isinstance(gen, torch.Generator)

    monkeypatch.setattr(pipeline, "_gather_rows", gather_rows)
    monkeypatch.setattr(
        pipeline, "_large_batch_view", lambda slot: {"obs": pipeline._gpu_packed[slot]}
    )
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)

    pipeline.warmup_result = pipeline.warmup()

    assert gathered == [(8, 1)]
    warmup_batch = pipeline.warmup_result
    assert warmup_batch is not None
    assert warmup_batch["obs"].shape == (8, 12)
    assert torch.count_nonzero(warmup_batch["obs"]) == 0
