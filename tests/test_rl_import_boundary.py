"""Import boundaries for the CHORD-owned FlashSAC and training stack."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "chord_unilab"
RL_PACKAGE = PACKAGE / "rl"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _violations(root: Path, forbidden: tuple[str, ...]) -> list[tuple[str, str]]:
    return [
        (path.relative_to(ROOT).as_posix(), module)
        for path in sorted(root.rglob("*.py"))
        for module in sorted(_imports(path))
        if any(module == layer or module.startswith(f"{layer}.") for layer in forbidden)
    ]


def test_chord_does_not_import_unilab_rl_training_or_scripts() -> None:
    violations = _violations(
        PACKAGE, ("unilab.rl", "unilab.training", "unilab.scripts")
    )
    assert [
        item for item in violations if item[1] != "unilab.training.tensor_diagnostics"
    ] == []


def test_only_flashsac_algorithm_was_migrated() -> None:
    assert (
        _violations(
            PACKAGE,
            (
                "chord_unilab.rl.algos.sac",
                "chord_unilab.rl.algos.appo",
                "chord_unilab.rl.adapters",
                "chord_unilab.rl.logging.onpolicy",
            ),
        )
        == []
    )
    appo_rollout_ipc = RL_PACKAGE / "runtime" / "ipc" / "tensor_rollout_ring_buffer.py"
    assert not appo_rollout_ipc.exists()


@pytest.mark.parametrize(
    ("layer", "forbidden"),
    [
        (
            "contracts",
            (
                "chord_unilab.training",
                "chord_unilab.scripts",
                "chord_unilab.rl.algos",
                "chord_unilab.rl.runtime",
                "unilab",
                "unisim",
            ),
        ),
        (
            "runtime",
            (
                "chord_unilab.training",
                "chord_unilab.scripts",
                "chord_unilab.rl.algos",
                "unilab.base",
                "unilab.envs",
                "unilab.tasks",
                "unilab.managers",
                "unisim",
            ),
        ),
        (
            "algos",
            (
                "chord_unilab.training",
                "chord_unilab.scripts",
                "unilab",
                "unisim",
            ),
        ),
    ],
)
def test_rl_internal_dag(layer: str, forbidden: tuple[str, ...]) -> None:
    assert _violations(RL_PACKAGE / layer, forbidden) == []


def test_logging_does_not_import_higher_layers() -> None:
    assert (
        _violations(
            RL_PACKAGE / "logging",
            ("chord_unilab.training", "chord_unilab.scripts", "unilab", "unisim"),
        )
        == []
    )


def test_rl_package_import_stays_lightweight() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; import chord_unilab.rl; "
            "assert 'torch' not in sys.modules and 'tensordict' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_training_playback_closure_excludes_unilab_rl() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; import chord_unilab.training.offpolicy; "
            "import chord_unilab.training.playback; "
            "assert not any(m.startswith(('unilab.rl', 'unilab.scripts')) "
            "or (m.startswith('unilab.training') and m != 'unilab.training.tensor_diagnostics') "
            "for m in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
