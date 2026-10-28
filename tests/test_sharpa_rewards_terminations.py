"""Independent wrench, shaped-reward, and termination equations."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from unilab.envs import ManagerBasedRlEnv

from chord_unilab.tasks.sharpa_chord.commands import ReferenceCommand
from chord_unilab.tasks.sharpa_chord.env import SharpaChordEnv
from chord_unilab.tasks.sharpa_chord.math import from_euler_xyz
from chord_unilab.tasks.sharpa_chord.rewards import (
    RewardRuntime,
    WrenchWorkspace,
    sample_wrench_basis,
    wrench_support,
)
from chord_unilab.tasks.sharpa_chord.terminations import (
    TrackingTermination,
    reference_end,
)


def test_raw_reward_recovery_rejects_zero_weight() -> None:
    env = cast(Any, object.__new__(SharpaChordEnv))
    env._cfg = SimpleNamespace(
        rewards={"active": SimpleNamespace(weight=0.0)},
    )
    env._device = torch.device("cpu")
    env.reward_manager = SimpleNamespace(step_reward_names=("active",))
    with (
        patch.object(ManagerBasedRlEnv, "__init__", return_value=None),
        pytest.raises(ValueError, match="non-zero weights"),
    ):
        SharpaChordEnv.__init__(env)


def test_wrench_support_has_normal_direction_and_friction_cone() -> None:
    basis = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    support = wrench_support(
        torch.zeros(1, 1, 1, 1, 3),
        torch.tensor([[[[[0.0, 0.0, 2.0]]]]]),
        basis,
        torch.tensor([1.0]),
    )
    torch.testing.assert_close(
        support[0, 0, 0], torch.tensor([1.0, 0.0, 0.1 / (1.01**0.5)])
    )
    assert 0.099 < support[0, 0, 0, 2].item() < 0.1
    assert sample_wrench_basis(512, 1.0, 0).shape == (512, 6)


def test_reference_basis_scales_torque_by_inverse_body_radius() -> None:
    expected = np.random.default_rng(7).standard_normal((16, 6))
    expected[:, 3:] /= 0.25
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    actual = sample_wrench_basis(16, 0.25, 7)
    np.testing.assert_allclose(
        actual, expected.astype(np.float32), rtol=1e-6, atol=1e-7
    )


def test_runtime_keeps_reference_and_live_bases_separate() -> None:
    contact_position = torch.zeros(1, 1, 3)
    contact_normal = torch.tensor([[[0.0, 0.0, 1.0]]])
    reference = SimpleNamespace(
        contact_pos_w=lambda side: contact_position.numpy(),
        contact_normal_w=lambda side: contact_normal.numpy(),
        contact_part_ids=lambda side: np.ones((1, 1), dtype=np.int32),
    )
    command = SimpleNamespace(
        reference=reference,
        num_bodies=1,
        _tensor=lambda values: torch.as_tensor(values, dtype=torch.float32),
        ref_object_pos=torch.zeros(1, 1, 3),
        ref_object_quat=torch.ones(1, 1, 4),
        ref_object_radius=torch.tensor([0.2]),
        applied_voc_scale=torch.ones(1),
    )
    runtime = RewardRuntime(cast(Any, command))
    torch.testing.assert_close(
        runtime.reference_basis,
        torch.as_tensor(sample_wrench_basis(512, 0.2, 0))[None],
    )
    torch.testing.assert_close(
        runtime.live_basis,
        torch.as_tensor(sample_wrench_basis(512, 1.0, 0))[None],
    )
    assert runtime.axes.device == command.ref_object_pos.device
    assert not torch.equal(runtime.reference_basis, runtime.live_basis)


@pytest.mark.newton
def test_wrench_support_cuda_matches_torch_for_mixed_contacts() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA wrench parity requires a CUDA device")
    torch.manual_seed(17)
    position_o = torch.randn(2, 2, 2, 17, 3)
    direction_o = torch.randn(2, 2, 2, 17, 3)
    direction_o[:, :, :, 0] = 0.0
    direction_o[:, :, :, 7] = 0.0
    basis = torch.randn(2, 512, 6)
    basis /= basis.norm(dim=-1, keepdim=True)
    radius = torch.tensor([0.2, 0.35])
    expected = wrench_support(position_o, direction_o, basis, radius)
    output = torch.empty((2, 2, 2, 512), device="cuda:0")
    actual = wrench_support(
        position_o.cuda(),
        direction_o.cuda(),
        basis.cuda(),
        radius.cuda(),
        output=output,
    )
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected.cuda(), rtol=1e-5, atol=2e-6)

    actual = wrench_support(
        position_o.cuda(),
        torch.zeros_like(direction_o).cuda(),
        basis.cuda(),
        radius.cuda(),
        output=output,
    )
    torch.testing.assert_close(actual, torch.zeros_like(expected).cuda())


def test_wrench_workspace_preserves_exact_values() -> None:
    torch.manual_seed(17)
    position_o = torch.randn(2, 2, 2, 17, 3)
    direction_o = torch.randn(2, 2, 2, 17, 3)
    direction_o[..., 0] = 0.0
    basis = torch.randn(2, 512, 6)
    basis /= basis.norm(dim=-1, keepdim=True)
    radius = torch.tensor([0.2, 0.35])
    shape = (2, 2, 2, 17, 9, 16)
    workspace = WrenchWorkspace(
        product=torch.empty((*shape, 3)),
        force_values=torch.empty(shape),
        torque_values=torch.empty(shape),
        chunk_output=torch.empty((2, 2, 2, 16)),
        zero=torch.zeros((1, 1, 1, 1, 1, 1)),
        chunk_size=16,
        cone_edges=8,
    )
    for value in (
        workspace.product,
        workspace.force_values,
        workspace.torque_values,
        workspace.chunk_output,
    ):
        value.fill_(torch.nan)
    actual = torch.full((2, 2, 2, 512), torch.nan)
    wrench_support(
        position_o,
        direction_o,
        basis,
        radius,
        output=actual,
        workspace=workspace,
    )
    assert torch.equal(actual, wrench_support(position_o, direction_o, basis, radius))


def _reward_runtime():
    contact = SimpleNamespace(
        position_o=torch.zeros(1, 2, 1, 17, 3),
        direction_o=torch.tensor([0.0, 0.0, 1.0]).expand(1, 2, 1, 17, 3),
    )
    reference = SimpleNamespace(
        contact_pos_w=lambda side: contact.position_o[:, 0, 0].numpy(),
        contact_normal_w=lambda side: contact.direction_o[:, 0, 0].numpy(),
        contact_part_ids=lambda side: np.ones((1, 17), dtype=np.int32),
    )
    object_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    hand_kp = torch.zeros(1, 2, 6, 3)
    command = SimpleNamespace(
        reference=reference,
        num_bodies=1,
        _tensor=lambda values: torch.as_tensor(values, dtype=torch.float32),
        ref_object_pos=torch.zeros(1, 1, 3),
        ref_object_quat=object_quat,
        applied_voc_scale=torch.ones(1),
        current_reference=lambda: (
            torch.zeros(1, 2, 3),
            torch.ones(1, 2, 4),
            torch.zeros(1, 2, 22),
            torch.full((1, 1, 3), 0.02),
            object_quat,
            torch.zeros(1, 0),
            hand_kp + 0.1,
        ),
        object_state=lambda: (
            torch.zeros(1, 1, 3),
            object_quat,
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 3),
        ),
        hand_keypoint_state=lambda: hand_kp,
        contact_state=lambda: contact,
        ref_object_radius=torch.tensor([0.2]),
        frames=torch.zeros(1, dtype=torch.int64),
        _policy_observation=SimpleNamespace(finger_qpos=torch.arange(44)),
    )
    runtime = RewardRuntime(cast(Any, command))
    runtime.ref_support = wrench_support(
        contact.position_o,
        contact.direction_o,
        runtime.live_basis,
        command.ref_object_radius,
    )
    return runtime


def test_runtime_creates_and_reuses_workspace_with_exact_values() -> None:
    torch.manual_seed(23)
    runtime = _reward_runtime()
    assert runtime.wrench_workspace is None
    contact = runtime.command.contact_state()
    contact.position_o = torch.randn_like(contact.position_o)
    contact.direction_o = torch.randn_like(contact.direction_o)
    contact.direction_o[..., 0] = 0.0
    action = SimpleNamespace(action_rate_l2=torch.zeros(1), action_l2=torch.zeros(1))
    env = SimpleNamespace(
        common_step_counter=1,
        num_envs=1,
        backend=SimpleNamespace(
            get_state_views=lambda names: {"qpos": torch.zeros(1, 44)}
        ),
        action_manager=SimpleNamespace(get_term=lambda name: action),
        termination_manager=SimpleNamespace(
            terminated=torch.zeros(1, dtype=torch.bool)
        ),
    )
    with patch(
        "chord_unilab.tasks.sharpa_chord.rewards.wrench_support",
        wraps=wrench_support,
    ) as support_call:
        runtime.evaluate(cast(Any, env))
    workspace = runtime.wrench_workspace
    assert workspace is not None
    assert support_call.call_args.kwargs["workspace"] is workspace
    assert workspace.product.shape == (1, 2, 1, 17, 9, 16, 3)
    assert workspace.force_values.shape == (1, 2, 1, 17, 9, 16)
    assert workspace.torque_values.shape == (1, 2, 1, 17, 9, 16)
    assert workspace.chunk_output.shape == (1, 2, 1, 16)
    for value in (
        workspace.product,
        workspace.force_values,
        workspace.torque_values,
        workspace.chunk_output,
        workspace.zero,
    ):
        assert value.device == contact.position_o.device
        assert value.dtype == contact.position_o.dtype
    expected = wrench_support(
        contact.position_o,
        contact.direction_o,
        runtime.live_basis,
        runtime.command.ref_object_radius,
    )
    assert torch.equal(runtime.live_support, expected)

    contact.position_o = torch.randn_like(contact.position_o)
    contact.direction_o = torch.randn_like(contact.direction_o)
    contact.direction_o[..., 0] = 0.0
    runtime.live_support.fill_(torch.nan)
    for value in (
        workspace.product,
        workspace.force_values,
        workspace.torque_values,
        workspace.chunk_output,
    ):
        value.fill_(torch.nan)
    env.common_step_counter = 2
    runtime.evaluate(cast(Any, env))
    assert runtime.wrench_workspace is workspace
    expected = wrench_support(
        contact.position_o,
        contact.direction_o,
        runtime.live_basis,
        runtime.command.ref_object_radius,
    )
    assert torch.equal(runtime.live_support, expected)


def test_reward_terms_use_laplacian_l2_and_exact_contact_support() -> None:
    runtime = _reward_runtime()
    action = SimpleNamespace(
        action_rate_l2=torch.tensor([2.0]), action_l2=torch.tensor([3.0])
    )
    qpos = torch.zeros(1, 44)
    qpos[0, 0] = 0.3
    qpos[0, 22] = 0.3
    env = SimpleNamespace(
        common_step_counter=1,
        num_envs=1,
        backend=SimpleNamespace(get_state_views=lambda names: {"qpos": qpos}),
        action_manager=SimpleNamespace(get_term=lambda name: action),
        termination_manager=SimpleNamespace(terminated=torch.tensor([True])),
    )
    runtime.evaluate(cast(Any, env))
    torch.testing.assert_close(
        runtime.terms["object_keypoints"],
        torch.tensor([torch.exp(torch.tensor(-0.02 * 3.0**0.5 / 0.1))]),
    )
    torch.testing.assert_close(
        runtime.terms["hand_keypoints"],
        torch.tensor([torch.exp(torch.tensor(-(3.0**0.5)))]),
    )
    torch.testing.assert_close(
        runtime.terms["hand_joint_pos"], torch.tensor([torch.exp(torch.tensor(-0.3))])
    )
    torch.testing.assert_close(runtime.terms["contact_wrench_support"], torch.ones(1))
    torch.testing.assert_close(runtime.terms["missed_contact"], torch.zeros(1))
    torch.testing.assert_close(runtime.terms["unintended_contact"], torch.zeros(1))
    torch.testing.assert_close(runtime.terms["termination"], torch.ones(1))
    torch.testing.assert_close(runtime.terms["action_rate_l2"], torch.tensor([2.0]))
    torch.testing.assert_close(runtime.terms["action_l2"], torch.tensor([3.0]))


def test_tracking_thresholds_and_reference_end() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    object_quat = from_euler_xyz(torch.tensor([0.0, 0.0, 0.6]))[None, None]
    command = cast(Any, object.__new__(ReferenceCommand))
    vars(command).update(
        current_reference=lambda: (
            torch.zeros(1, 2, 3),
            identity.expand(1, 2, 4),
            torch.zeros(1, 2, 22),
            torch.zeros(1, 1, 3),
            identity.expand(1, 1, 4),
            torch.zeros(1, 0),
            torch.zeros(1, 2, 6, 3),
        ),
        wrist_state=lambda: (
            torch.tensor([[[0.16, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            identity.expand(1, 2, 4),
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 3),
        ),
        object_state=lambda: (
            torch.zeros(1, 1, 3),
            object_quat,
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 3),
        ),
        timestep=torch.tensor([9]),
        num_frames=10,
    )
    env = cast(
        Any,
        SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command)),
    )
    for kind in ("wrist_position", "object_orientation"):
        term = cast(Any, object.__new__(TrackingTermination))
        term.command = command
        term.kind = kind
        assert term(env, kind=kind).item()
    command.object_state = lambda: (
        torch.full((1, 1, 3), 0.11),
        identity.expand(1, 1, 4),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
    )
    term = cast(Any, object.__new__(TrackingTermination))
    term.command = command
    assert term(env, kind="object_position").item()
    assert reference_end(env).item()
    command.timestep[:] = 8
    assert not reference_end(env).item()
