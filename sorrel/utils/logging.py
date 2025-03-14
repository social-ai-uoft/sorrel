# --------------------------- #
# region: Game data storage   #
# --------------------------- #

import numpy as np
import torch
from IPython.display import clear_output


class GameLogger:
    """
    Container for storing game variables.
    """

    def __init__(self, max_epochs):
        self.epochs = []
        self.turns = []
        self.losses = []
        self.rewards = []
        self.max_epochs = max_epochs

    def clear(self):
        """
        Clear the game variables.
        """
        del self.epochs[:]
        del self.turns[:]
        del self.losses[:]
        del self.rewards[:]

    def record_turn(
        self,
        epoch: int,
        turn: int,
        loss: float | torch.Tensor,
        reward: int | float | torch.Tensor,
    ):
        """
        Record a game turn.
        """
        self.epochs.append(epoch)
        self.turns.append(turn)
        self.losses.append(np.round(loss, 2))
        self.rewards.append(reward)

    def pretty_print(self, *flags, **kwargs) -> None:
        """
        Take the results from a given epoch (epoch #, turn #, loss, and reward)
        and return a formatted string that can be printed to the command line.

        If `jupyter-mode` is passed in as a flag, variables need to be passed
        in with the `kwargs`.
        """

        if "jupyter-mode" in flags:
            assert all(
                key in kwargs.keys() for key in ("epoch", "turn", "reward")
            ), "Jupyter mode requires the current epoch, turn, and reward to be passed in as kwargs."
            clear_output(wait=True)
            print(f"╔═════════════╦═══════════╦═════════════╦═════════════╗")
            print(
                f'║ Epoch: {str(kwargs["epoch"]).rjust(4)} ║ Turn: {str(kwargs["turn"]).rjust(3)} ║ Loss: {str("None").rjust(5)} ║ Reward: {str(kwargs["reward"]).rjust(3)} ║'
            )
            print(f"╚═════════════╩═══════════╩═════════════╩═════════════╝")
        else:
            if self.epochs[-1] == 0:
                print(f"╔═════════════╦═══════════╦═════════════╦═════════════╗")
            else:
                print(f"╠═════════════╬═══════════╬═════════════╬═════════════╣")
            if True:
                print(
                    f"║ Epoch: {str(self.epochs[-1]).rjust(4)} ║ Turn: {str(self.turns[-1]).rjust(3)} ║ Loss: {str(self.losses[-1]).rjust(5)} ║ Reward: {str(self.rewards[-1]).rjust(3)} ║"
                )
                print(
                    f"╚═════════════╩═══════════╩═════════════╩═════════════╝", end="\r"
                )
            if self.epochs[-1] == self.max_epochs - 1:
                print(f"╚═════════════╩═══════════╩═════════════╩═════════════╝")

    def __repr__(self):
        return f"{self.__class__.__name__}(n_games={len(self.epochs)})"
    
# --------------------------- #
# endregion                   #
# --------------------------- #