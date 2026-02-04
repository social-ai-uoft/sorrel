"""Entry point for TreasureHunt + CleanRL PPO training."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from sorrel.examples.treasurehunt_cleanrl.train_cleanrl_ppo import train_cleanrl_ppo


def _load_base_config() -> DictConfig:
    config_path = Path(__file__).parent / "config" / "config.json"
    with config_path.open("r") as file:
        return OmegaConf.create(json.load(file))


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cli_config: DictConfig) -> None:
    base_config = _load_base_config()
    merged_config = OmegaConf.merge(base_config, cli_config)
    outputs = train_cleanrl_ppo(merged_config)

    print("Training complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
