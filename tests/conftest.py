"""Shared host-only fixtures for migrated runtime tests."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def short_unix_socket_root() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="chord-socket-", dir="/tmp") as directory:
        yield Path(directory)
