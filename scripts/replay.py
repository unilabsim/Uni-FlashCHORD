"""Replay one parquet through the production CHORD/UniSim scene assembly."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base="1.3", config_path="../src/chord_unilab/conf", config_name="replay"
)
def main(cfg: DictConfig) -> None:
    from chord_unilab.replay.app import launch

    launch(cfg)


if __name__ == "__main__":
    main()
