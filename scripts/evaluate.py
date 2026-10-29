"""Evaluate a FlashSAC checkpoint interactively through Viser."""

from __future__ import annotations

import argparse
from pathlib import Path

from chord_unilab.eval.viser import launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    launch(
        args.parquet.expanduser().resolve(),
        args.checkpoint.expanduser().resolve(),
        port=args.port,
    )


if __name__ == "__main__":
    main()
