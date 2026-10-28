"""FlashSAC training and playback Hydra entrypoint."""

import hydra
from omegaconf import DictConfig
from unilab.interruption import graceful_keyboard_interrupt

from chord_unilab.training.offpolicy import main as _offpolicy_main


@graceful_keyboard_interrupt
@hydra.main(version_base="1.3", config_path="../conf/flashsac", config_name="config")
def main(cfg: DictConfig) -> None:
    _offpolicy_main(cfg)


if __name__ == "__main__":
    main()
