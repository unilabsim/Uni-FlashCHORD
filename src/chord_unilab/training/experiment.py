"""Shared experiment tracking utilities for local files and W&B."""

from __future__ import annotations

import dataclasses
import getpass
import importlib
import json
import os
import socket
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from unilab.utils.device import get_device_info_dict
from unilab.utils.sim2sim import extract_contract_snapshot


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _plain_dict(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _load_wandb() -> Any | None:
    try:
        return importlib.import_module("wandb")
    except ImportError:
        return None


def _tb_proto_modules() -> tuple[Any, Any]:
    """Import tensorboard's protobuf modules as ``Any``.

    The generated modules define their messages through reflection, which
    static checkers cannot see; returning them as ``Any`` keeps attribute
    access clean without per-line suppressions.
    """
    from tensorboard.compat.proto import event_pb2, summary_pb2

    return event_pb2, summary_pb2


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _copy_plain_training_state(state: Any) -> dict[str, Any]:
    """Validate and detach versioned training progress at a checkpoint boundary.

    This intentionally accepts only plain JSON-shaped values. It rejects tensors
    and other owner objects before they can enter a checkpoint payload.
    """

    def copy_value(value: Any) -> Any:
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("Training state numbers must be finite")
            return value
        if type(value) is not dict and type(value) is not list:
            raise TypeError(
                "Training state must contain only plain JSON dictionaries/lists/scalars"
            )
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError("Training state dictionary keys must be strings")
            return {key: copy_value(item) for key, item in value.items()}
        return [copy_value(item) for item in value]

    if not isinstance(state, dict):
        raise TypeError("Training state export must return a dictionary")
    result = copy_value(state)
    if not isinstance(result, dict):
        raise TypeError("Training state export must return a dictionary")
    return result


def get_git_info(root_dir: str | Path) -> dict[str, Any]:
    root = Path(root_dir)

    def _run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git("status", "--short")

    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def _write_json(path: Path, payload: Any, *, trailing_newline: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    if trailing_newline:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def build_run_config_payload(
    *,
    run_metadata: dict[str, Any],
    full_cfg: Any,
    contract_snapshot: Any | None = None,
) -> dict[str, Any]:
    """Assemble the canonical ``{"run", "config", ...}`` run-config payload."""
    payload: dict[str, Any] = {
        "run": _json_safe(run_metadata),
        "config": _json_safe(_plain_dict(full_cfg)),
    }
    if contract_snapshot is not None:
        payload["contract_snapshot"] = _json_safe(contract_snapshot)
    return payload


def write_run_config_snapshot(
    log_dir: str | Path,
    *,
    run_metadata: dict[str, Any],
    full_cfg: Any,
    filename: str = "run_config.json",
    contract_snapshot: Any | None = None,
    trailing_newline: bool = False,
) -> dict[str, Any]:
    """Write the run-config snapshot into ``log_dir`` and return the payload."""
    payload = build_run_config_payload(
        run_metadata=run_metadata,
        full_cfg=full_cfg,
        contract_snapshot=contract_snapshot,
    )
    _write_json(Path(log_dir) / filename, payload, trailing_newline=trailing_newline)
    return payload


def build_wandb_run_name(
    algo_name: str, task_name: str, log_dir: str | Path | None
) -> str:
    if log_dir is None:
        return f"{algo_name}__{task_name}"
    run_dir = Path(log_dir)
    return f"{algo_name}__{task_name}__{run_dir.name}"


def build_wandb_settings(
    training_cfg: Any,
    *,
    algo_name: str,
    task_name: str,
    sim_backend: str,
    log_dir: str | Path | None,
) -> dict[str, Any]:
    name = _cfg_get(training_cfg, "wandb_name")
    if not name:
        name = build_wandb_run_name(algo_name, task_name, log_dir)

    group = _cfg_get(training_cfg, "wandb_group")
    if not group:
        group = task_name

    job_type = _cfg_get(training_cfg, "wandb_job_type")
    if not job_type:
        job_type = algo_name

    tags = [str(tag) for tag in (_cfg_get(training_cfg, "wandb_tags", []) or [])]
    auto_tags = [algo_name, task_name, sim_backend, f"user-{getpass.getuser()}"]
    for tag in auto_tags:
        if tag not in tags:
            tags.append(tag)

    return {
        "project": _cfg_get(training_cfg, "wandb_project", "unilab"),
        "entity": _cfg_get(training_cfg, "wandb_entity"),
        "name": name,
        "group": group,
        "job_type": job_type,
        "tags": tags,
        "notes": _cfg_get(training_cfg, "wandb_notes"),
        "mode": _cfg_get(training_cfg, "wandb_mode"),
    }


class ExperimentTracker:
    """Tracks experiment metadata locally and optionally in Weights & Biases."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        log_dir: str | Path,
        algo_name: str,
        task_name: str,
        sim_backend: str,
        training_cfg: Any,
        full_cfg: Any,
        device: str | None = None,
        configured_device: str | None = None,
        collector_device: str | None = None,
        device_resolution: Mapping[str, Mapping[str, str]] | None = None,
        seed_info: Any | None = None,
        tf32_matmul_policy: Any | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.log_dir = Path(log_dir)
        self.algo_name = algo_name
        self.task_name = task_name
        self.sim_backend = sim_backend
        self.training_cfg = training_cfg
        self.full_cfg = full_cfg
        self.device = device
        self.configured_device = configured_device
        self.collector_device = collector_device
        self.device_resolution = (
            dict(device_resolution) if device_resolution is not None else None
        )
        self.seed_info = seed_info
        self.tf32_matmul_policy = tf32_matmul_policy
        self.enabled = (
            str(_cfg_get(training_cfg, "logger", "tensorboard")).lower() == "wandb"
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._wandb = None
        self._run = None
        self._owns_run = False
        self._started = False
        self._start_monotonic = 0.0
        self._start_utc = ""
        self._summary: dict[str, Any] = {}

    @property
    def run(self) -> Any | None:
        return self._run

    @property
    def run_url(self) -> str | None:
        return getattr(self._run, "url", None) if self._run is not None else None

    @property
    def wandb_settings(self) -> dict[str, Any]:
        return build_wandb_settings(
            self.training_cfg,
            algo_name=self.algo_name,
            task_name=self.task_name,
            sim_backend=self.sim_backend,
            log_dir=self.log_dir,
        )

    def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._start_monotonic = time.perf_counter()
        self._start_utc = datetime.now(UTC).isoformat()

        metadata = {
            "algo": self.algo_name,
            "task": self.task_name,
            "sim_backend": self.sim_backend,
            "device": self.device,
            "configured_device": self.configured_device,
            "collector_device": self.collector_device,
            "resolved_devices": self.device_resolution,
            "log_dir": str(self.log_dir),
            "start_time_utc": self._start_utc,
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "git": get_git_info(self.root_dir),
            "hardware": get_device_info_dict(),
            "wandb": self.wandb_settings,
        }
        if self.seed_info is not None:
            if hasattr(self.seed_info, "to_dict"):
                seed_payload = self.seed_info.to_dict()
            elif isinstance(self.seed_info, dict):
                seed_payload = dict(self.seed_info)
            else:
                seed_payload = {"effective_seed": self.seed_info}
            metadata.update(seed_payload)
        if self.tf32_matmul_policy is not None and hasattr(
            self.tf32_matmul_policy, "manifest"
        ):
            metadata["tf32_matmul"] = self.tf32_matmul_policy.manifest()

        payload = write_run_config_snapshot(
            self.log_dir,
            run_metadata=metadata,
            full_cfg=self.full_cfg,
            contract_snapshot=extract_contract_snapshot(self.full_cfg),
        )
        if self.device_resolution:
            mapping = ", ".join(
                f"{name} {entry['configured']} -> {entry['resolved']}"
                for name, entry in self.device_resolution.items()
            )
            print(f"[experiment_tracking] Device resolution: {mapping}")

        if not self.enabled:
            return

        self._wandb = _load_wandb()
        if self._wandb is None:
            print(
                "[experiment_tracking] wandb not installed, skipping W&B experiment tracking."
            )
            return

        self._run = self._wandb.run
        if self._run is None:
            kwargs = {
                "project": self.wandb_settings["project"],
                "name": self.wandb_settings["name"],
                "config": payload,
                "dir": str(self.log_dir),
                "reinit": True,
            }
            for key in ("entity", "group", "job_type", "tags", "notes", "mode"):
                value = self.wandb_settings.get(key)
                if value not in (None, "", []):
                    kwargs[key] = value
            self._run = self._wandb.init(**kwargs)
            self._owns_run = True
        else:
            self._run.config.update(payload, allow_val_change=True)

        if self._run is not None:
            self._run.summary["algo"] = self.algo_name
            self._run.summary["task"] = self.task_name
            self._run.summary["sim_backend"] = self.sim_backend
            if self.device:
                self._run.summary["device"] = self.device
            if self.collector_device:
                self._run.summary["collector_device"] = self.collector_device
            self._run.summary["log_dir"] = str(self.log_dir)

    def update_summary(self, summary: dict[str, Any] | None = None) -> None:
        if summary:
            self._summary.update(summary)

        if not self._started:
            return

        wall_time_sec = time.perf_counter() - self._start_monotonic
        payload = {
            **self._summary,
            "algo": self.algo_name,
            "task": self.task_name,
            "sim_backend": self.sim_backend,
            "log_dir": str(self.log_dir),
            "start_time_utc": self._start_utc,
            "end_time_utc": datetime.now(UTC).isoformat(),
            "wall_time_sec": wall_time_sec,
            "wandb_run_url": self.run_url,
        }
        if self.seed_info is not None:
            if hasattr(self.seed_info, "to_dict"):
                payload.update(self.seed_info.to_dict())
            elif isinstance(self.seed_info, dict):
                payload.update(self.seed_info)
            else:
                payload["effective_seed"] = self.seed_info
        if self.tf32_matmul_policy is not None and hasattr(
            self.tf32_matmul_policy, "manifest"
        ):
            payload["tf32_matmul"] = self.tf32_matmul_policy.manifest()
        _write_json(self.log_dir / "run_summary.json", _json_safe(payload))

        if self._run is not None:
            for key, value in payload.items():
                self._run.summary[key] = _json_safe(value)

    def log_video(
        self, video_path: str | Path | None, key: str = "media/play_video"
    ) -> None:
        if video_path is None:
            return

        video = Path(video_path)
        if not video.exists():
            return

        self._summary["play_video_path"] = str(video)
        if self._run is not None and self._wandb is not None:
            self._wandb.log({key: self._wandb.Video(str(video), format="mp4")})

    def finish(self) -> None:
        if not self._started:
            return

        self.update_summary()
        if self._run is not None and self._wandb is not None and self._owns_run:
            self._wandb.finish()
        self._run = None
        self._wandb = None


class _BatchedScalarWriter:
    """Buffer rsl-rl's per-tag ``add_scalar`` calls and emit one event per step.

    TensorBoard's writer thread performs one open/write/close per record, which
    saturates its async queue (depth 10) and blocks the training loop when the
    log directory lives on a network filesystem. Batching one iteration's
    scalars into a single ``Event`` keeps it to one record per step group.
    """

    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self._pending: list[tuple[str, float]] = []
        self._pending_step: int | None = None

    def add_scalar(
        self, tag: str, value: Any, step: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            self._flush()
            self._writer.add_scalar(tag, value, step, *args, **kwargs)
            return
        step_key = int(step) if step is not None else 0
        if self._pending and step_key != self._pending_step:
            self._flush()
        self._pending_step = step_key
        self._pending.append((tag, scalar))

    def _flush(self) -> None:
        if not self._pending:
            return
        event_pb2, summary_pb2 = _tb_proto_modules()
        event = event_pb2.Event(
            wall_time=time.time(),
            step=self._pending_step or 0,
            summary=summary_pb2.Summary(
                value=[
                    summary_pb2.Summary.Value(tag=tag, simple_value=value)
                    for tag, value in self._pending
                ]
            ),
        )
        self._writer.file_writer.add_event(event)
        self._pending = []
        self._pending_step = None

    def flush(self) -> None:
        self._flush()
        self._writer.flush()

    def close(self) -> None:
        self._flush()
        self._writer.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)


class _SwallowScalarWriter:
    """Drop ``add_scalar`` writes while staying truthy for rsl-rl's console log."""

    def __init__(self, writer: Any) -> None:
        self._writer = writer

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)
