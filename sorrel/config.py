# --------------- #
# region: Imports #
# --------------- #

import argparse
import os
from typing import Any, Optional, Self, Sequence

# Import base packages
import yaml

# --------------- #
# endregion       #
# --------------- #

# --------------------------- #
# region: config functions    #
# --------------------------- #


class Cfg:
    """
    Configuration class for parsing the YAML configuration file.

    Nested config parameters are parsed as nested Cfg objects.
    """

    experiment: Self
    name: str
    epochs: int
    max_turns: int

    env: Self
    height: int
    width: int
    layers: int
    default_object: Any

    model: Self

    agent: Self
    num: int
    model: Any
    appearance: Any

    root: str
    log: bool

    def __init__(self, in_dict: dict):
        """Recursively turn the dictionary into attributes for this class."""
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Cfg(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Cfg(val) if isinstance(val, dict) else val)

    def to_dict(self):
        return self.__dict__


def init_log(cfg: Cfg) -> None:
    print(f"╔═════════════════════════════════════════════════════╗")
    print(f"║                                                     ║")
    print(
        f"║        Starting experiment: {str(cfg.experiment.name).ljust(16)}        ║"
    )
    print(
        f'║        Epochs: {str(cfg.experiment.epochs).ljust(5)}  Max turns: {str(cfg.experiment.max_turns).ljust(3)}  Log: {"Yes" if cfg.log else "No "}      ║'
    )
    print(f"║        Saving to: {str(cfg.save_dir).ljust(26)}        ║")
    print(f"║                                                     ║")
    print(f"╚═════════════════════════════════════════════════════╝")
    print(f"")


def parse_args(
    command_line: bool = True, args: Optional[Sequence[str]] = None
) -> argparse.Namespace:
    """Helper function for preparsing the arguments.

    Args:
        command_line (bool): whether arguments are passed in through the command line. Defaults to True.
        args (Optional[Sequence[str]]): A sequence of command line-like arguments, if not reading from command line. Defaults to None.

            .. Note::
                By default, the function uses the default `--config` argument to get the file from `./config.yaml`. \n
                Unknown arguments (i.e., not `--config`) are ignored.

    Returns:
        argpase.Namespace: the argparse.Namespace object containing the args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml", help="path to config file")
    # By default, parse the arguments
    if command_line:
        args = parser.parse_args()
    # Otherwise, args can be passed in. Default value is specified for config.
    else:
        if args is None:
            args = argparse.Namespace(config="./config.yaml")
        else:
            args, _ = parser.parse_known_args(args)
    return args


def load_config(args: argparse.Namespace) -> Cfg:
    """Load the parsed arguments into the Cfg class.

    Parameters:
        args (argparse.Namespace): the parsed arguments, where args.config must contain the path to the config file.

    Returns:
        A Cfg class object with the configurations for the experiment.
    """
    if args.config is None or not os.path.isfile(args.config):
        raise ValueError(
            "Config file not found, please make sure you've included a path to the config file."
        )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = Cfg(config)

    return config


# --------------------------- #
# endregion: config functions #
# --------------------------- #
