"""FlashChord residual actor and globally repeated exploration noise."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn

from chord_unilab.rl.algos.flash_sac.layers import (
    FlashSACBlock,
    FlashSACEmbedder,
    UnitLinear,
    UnitRMSNorm,
    safe_tanh_log_det_jacobian,
)


class GainedUnitLinear(nn.Module):
    """Unit-direction projection with an independently trained output gain."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = UnitLinear(input_dim, output_dim)
        self.gain = nn.Parameter(torch.zeros(output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value) * self.gain


class ResidualTanhPolicy(nn.Module):
    """Residual Gaussian heads with bounded log standard deviation."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        initial_normalized_std: float,
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        if not math.isfinite(initial_normalized_std) or initial_normalized_std <= 0.0:
            raise ValueError("initial_normalized_std must be finite and positive")
        if not math.isfinite(log_std_min) or not math.isfinite(log_std_max):
            raise ValueError("log standard-deviation bounds must be finite")
        initial_log_std = math.log(initial_normalized_std)
        if not log_std_min < initial_log_std < log_std_max:
            raise ValueError(
                "initial log standard deviation must lie strictly inside its bounds"
            )

        self.mean_w = GainedUnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.std_w = GainedUnitLinear(hidden_dim, action_dim)
        normalized_log_std = (
            2.0 * (initial_log_std - log_std_min) / (log_std_max - log_std_min) - 1.0
        )
        self.std_bias = nn.Parameter(
            torch.full((action_dim,), math.atanh(normalized_log_std))
        )
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def get_mean_and_std(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_w(value) + self.mean_bias
        raw_log_std = self.std_w(value) + self.std_bias
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1.0 + torch.tanh(raw_log_std)
        )
        return mean, torch.exp(log_std)

    def forward(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean, std = self.get_mean_and_std(value)
        distribution = torch.distributions.Normal(mean, std, validate_args=False)
        raw_action = distribution.rsample()
        actions = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action) - safe_tanh_log_det_jacobian(
            raw_action
        )
        return actions, {"log_prob": log_prob.sum(dim=-1), "mean": mean, "std": std}


class FlashChordActor(nn.Module):
    """FlashSAC actor with FlashChord residual heads and global noise repeats."""

    action_transform_id = "tanh"
    zeta_cdf: torch.Tensor
    _noise: torch.Tensor
    _repeat_count: torch.Tensor
    _repeat_target: torch.Tensor

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        expansion: int = 4,
        initial_normalized_std: float = 0.1,
        log_std_min: float = -10.0,
        log_std_max: float = -1.8971199848858813,
        noise_zeta_mu: float = 2.0,
        noise_zeta_max: int = 16,
        rms_norm_eps: float = 1.0e-6,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        if min(obs_dim, action_dim, hidden_dim, num_blocks, expansion) <= 0:
            raise ValueError("actor dimensions and block count must be positive")
        if not math.isfinite(noise_zeta_mu) or noise_zeta_mu <= 0.0:
            raise ValueError("noise_zeta_mu must be finite and positive")
        if noise_zeta_max <= 0:
            raise ValueError("noise_zeta_max must be positive")

        self.action_dim = int(action_dim)
        self.embedder = FlashSACEmbedder(input_dim=obs_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList(
            [FlashSACBlock(hidden_dim, expansion=expansion) for _ in range(num_blocks)]
        )
        self.post_norm = UnitRMSNorm(hidden_dim, eps=rms_norm_eps)
        self.predictor = ResidualTanhPolicy(
            hidden_dim,
            action_dim,
            initial_normalized_std,
            log_std_min,
            log_std_max,
        )

        lengths = torch.arange(1, noise_zeta_max + 1, dtype=torch.float32)
        probabilities = lengths.pow(-noise_zeta_mu)
        self.register_buffer(
            "zeta_cdf", torch.cumsum(probabilities / probabilities.sum(), dim=0)
        )
        self.register_buffer("_noise", torch.zeros(0), persistent=False)
        self.register_buffer("_repeat_count", torch.zeros((), dtype=torch.int32))
        self.register_buffer("_repeat_target", torch.zeros((), dtype=torch.int32))
        self.register_load_state_dict_post_hook(
            FlashChordActor._reset_exploration_on_load
        )
        self.to(device)
        self.normalize_parameters()

    @staticmethod
    def _reset_exploration_on_load(module: nn.Module, incompatible_keys: Any) -> None:
        del incompatible_keys
        actor = cast(FlashChordActor, module)
        # Legacy checkpoints exclude batch-shaped noise; resume with a fresh repeat segment.
        actor._repeat_count.zero_()
        actor._repeat_target.zero_()

    def normalize_parameters(self) -> None:
        with torch.no_grad():
            for child in self.modules():
                if child is self:
                    continue
                normalize = getattr(child, "normalize_parameters", None)
                if callable(normalize):
                    normalize()

    def inference_startup_memory_categories(self, batch_size: int) -> dict[str, int]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("FlashChord inference batch size must be an integer")
        if batch_size <= 0:
            raise ValueError("FlashChord inference batch size must be positive")
        return {
            "persistent_exploration_scratch": batch_size * self.action_dim * 4 + 2 * 4
        }

    def _encode(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        value = self.embedder(observations, training=training)
        for block in self.encoder:
            value = block(value, training=training)
        return cast(torch.Tensor, self.post_norm(value))

    def get_mean_and_std(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predictor.get_mean_and_std(self._encode(observations, training))

    def forward(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.predictor(self._encode(observations, training))

    def as_export_module(self) -> nn.Module:
        actor = self

        class ExportModule(nn.Module):
            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                mean, _std = actor.get_mean_and_std(observations, training=False)
                return torch.tanh(mean)

        return ExportModule()

    def _ensure_exploration_state(self, mean: torch.Tensor) -> None:
        if (
            self._noise.shape == mean.shape
            and self._noise.device == mean.device
            and self._noise.dtype == mean.dtype
        ):
            return
        self._noise = torch.zeros_like(mean)
        self._repeat_count = torch.zeros((), device=mean.device, dtype=torch.int32)
        self._repeat_target = torch.zeros((), device=mean.device, dtype=torch.int32)

    def _sample_repeat_target(self, device: torch.device) -> torch.Tensor:
        uniform = torch.rand((), device=device)
        cdf = self.zeta_cdf.to(device=device)
        return torch.argmax((uniform < cdf).to(torch.int32)) + 1

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        del dones
        mean, std = self.get_mean_and_std(obs, training=False)
        if deterministic:
            return torch.tanh(mean)

        self._ensure_exploration_state(mean)
        candidate_noise = torch.randn_like(mean)
        candidate_target = self._sample_repeat_target(mean.device)
        renew = (self._repeat_count == 0) | (self._repeat_count >= self._repeat_target)
        self._noise = torch.where(renew, candidate_noise, self._noise)
        self._repeat_target = torch.where(renew, candidate_target, self._repeat_target)
        self._repeat_count = torch.where(
            renew, torch.zeros_like(self._repeat_count), self._repeat_count
        )
        actions = torch.tanh(mean + std * self._noise)
        self._repeat_count = self._repeat_count + 1
        return actions


def build_actor(
    *,
    obs_dim: int,
    action_dim: int,
    device: str | torch.device = "cpu",
    hidden_dim: int = 256,
    num_blocks: int = 2,
    expansion: int = 4,
    initial_normalized_std: float = 0.1,
    log_std_min: float = -10.0,
    log_std_max: float = -1.8971199848858813,
    noise_zeta_mu: float = 2.0,
    noise_zeta_max: int = 16,
    rms_norm_eps: float = 1.0e-6,
) -> FlashChordActor:
    """Build the actor through UniLab's declarative ``actor_config`` contract."""
    return FlashChordActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        expansion=expansion,
        initial_normalized_std=initial_normalized_std,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        noise_zeta_mu=noise_zeta_mu,
        noise_zeta_max=noise_zeta_max,
        rms_norm_eps=rms_norm_eps,
    )
