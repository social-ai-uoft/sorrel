# --------------------------- #
# region: Game data storage   #
# --------------------------- #

import csv
import os
from typing import Mapping

import numpy as np
from IPython.display import clear_output
from torch.utils.tensorboard.writer import SummaryWriter


class Logger:
    """Abstract class for logging.

    Attributes:
        max_epochs: The number of epochs.
        losses: A list of the loss values for each epoch.
        rewards: A list of the reward values for each epoch.
        epsilons: A list of the epsilon values for each epoch.
        additional_values: A dictionary of optional values to be stored.
    """

    max_epochs: int
    losses: list[float | np.ndarray]
    rewards: list[float | np.ndarray]
    epsilons: list[float | np.ndarray]
    additional_values: Mapping[str, list[int | float | np.ndarray]]

    def __init__(self, max_epochs: int, *args: str):
        """Initialize a log.

        Args:
            max_epochs (int): The length of the lists.
            *args: Additional optional values to be stored in a dictionary.
        """
        self.max_epochs = max_epochs
        self.losses = []
        self.rewards = []
        self.epsilons = []
        self.additional_values = {}
        for additional_value in args:
            self.additional_values[additional_value] = []

    def record_turn(
        self,
        epoch: int,
        loss: float | np.ndarray,
        reward: float | np.ndarray,
        epsilon: float = 0,
        **kwargs,
    ) -> None:
        """Record a turn.

        Args:
            epoch (int): The number of the epoch.
            loss (float | torch.Tensor): The loss value.
            reward (float | torch.Tensor): The reward value.
            epsilon (float): The epsilon value.
            kwargs: Additional values to store.
        """
        self.epsilons.append(epsilon)
        self.losses.append(loss)
        self.rewards.append(reward)
        for key, value in kwargs.items():
            assert (
                key in self.additional_values.keys()
            ), "Can only store existing values."
            self.additional_values[key].append(value)

    def to_csv(self, file_path: str | os.PathLike) -> None:
        """Write the logged data to a CSV file.

        Args:
            file_path: The path to the file to write the data to.
        """
        records = {
            "epochs": list(range(len(self.losses))),
            "losses": self.losses,
            "rewards": self.rewards,
            "epsilons": self.epsilons,
            **self.additional_values,
        }

        with open(file_path, "a") as f:
            writer = csv.writer(f)
            if os.stat(file_path).st_size == 0:
                writer.writerow(list(records.keys()))
            for epoch in range(len(self.losses)):
                writer.writerow([value[epoch] for value in records.values()])


class ConsoleLogger(Logger):
    """Logs elements to the console.

    Attributes:
        max_epochs: The number of epochs.
        losses: A list of the loss values for each epoch.
        rewards: A list of the reward values for each epoch.
        epsilons: A list of the epsilon values for each epoch.
        additional_values: A dictionary of optional values to be stored.
    """

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        loss = np.round(loss, 2)
        # Print beginning of the frame
        if epoch == 0:
            print(f"╔══════════════╦══════════════╦══════════════╗")
        else:
            print(f"╠══════════════╬══════════════╬══════════════╣")
        # Print turn
        print(
            f"║ Epoch:{str(epoch).rjust(6)} ║ Loss:{str(loss).rjust(7)} ║ Reward:{str(reward).rjust(5)} ║"
        )
        print(f"╚══════════════╩══════════════╩══════════════╝", end="\r")
        if epoch == self.max_epochs - 1:
            print(f"╚══════════════╩══════════════╩══════════════╝")
        super().record_turn(epoch, loss, reward, epsilon, **kwargs)


class JupyterLogger(Logger):
    """Logs elements to a Jupyter notebook.

    Attributes:
        max_epochs: The number of epochs.
        losses: A list of the loss values for each epoch.
        rewards: A list of the reward values for each epoch.
        epsilons: A list of the epsilon values for each epoch.
        additional_values: A dictionary of optional values to be stored.
    """

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        loss = np.round(loss, 2)
        clear_output(wait=True)
        print(f"╔══════════════╦══════════════╦══════════════╗")
        print(
            f"║ Epoch:{str(epoch).rjust(6)} ║ Loss:{str(loss).rjust(7)} ║ Reward:{str(reward).rjust(5)} ║"
        )
        print(f"╚══════════════╩══════════════╩══════════════╝")
        super().record_turn(epoch, loss, reward, epsilon, **kwargs)


class TensorboardLogger(Logger):
    """Logs elements to a Tensorboard file.

    Attributes:
        max_epochs: The number of epochs.
        losses: A list of the loss values for each epoch.
        rewards: A list of the reward values for each epoch.
        epsilons: A list of the epsilon values for each epoch.
        additional_values: A dictionary of optional values to be stored.
    """

    def __init__(self, max_epochs: int, log_dir: str | os.PathLike, *args):
        """Initialize a Tensorboard log.

        Args:
            max_epochs (int): The length of the lists.
            log_dir (str | PathLike): Where the Tensorboard log files should be stored.
            *args: Additional optional values to be stored in a dictionary.
        """
        super().__init__(max_epochs=max_epochs, *args)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        self.writer.add_scalar("loss", loss, epoch)
        self.writer.add_scalar("score", reward, epoch)
        self.writer.add_scalar("epsilon", epsilon, epoch)
        for key, value in kwargs.items():
            self.writer.add_scalar(key, value, epoch)


# --------------------------- #
# endregion                   #
# --------------------------- #
