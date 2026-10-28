"""IPC primitives for multi-process RL training."""

from typing import TYPE_CHECKING, Any

from chord_unilab.rl.runtime.ipc.replay_buffer import ReplayBuffer
from chord_unilab.rl.runtime.ipc.weight_sync import SharedWeightSync

if TYPE_CHECKING:
    from chord_unilab.rl.runtime.async_runner import AsyncRunner

__all__ = [
    "AsyncRunner",
    "ReplayBuffer",
    "SharedWeightSync",
]


def __getattr__(name: str) -> Any:
    # AsyncRunner lives one level up (chord_unilab.rl.runtime.async_runner) and
    # imports back into this package; resolve it lazily so either import
    # order works.
    if name == "AsyncRunner":
        from chord_unilab.rl.runtime.async_runner import AsyncRunner

        return AsyncRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
