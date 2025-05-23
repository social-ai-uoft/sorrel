from typing import Sequence

import numpy as np
from IPython.display import clear_output

from sorrel.buffers import Buffer
from sorrel.models import BaseModel
from sorrel.utils.visualization import plot


class HumanPlayer(BaseModel):
    """Model subclass for a human player."""

    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        memory_size: int,
        show: bool = True,
    ):
        self.name = ""
        self.action_list = np.arange(action_space)
        self.input_size = input_size
        # TODO: add way to review/revisit previous memories using buffer?
        self.memory = Buffer(capacity=memory_size, obs_shape=input_size)
        self.num_frames = memory_size
        self.show = show

    def take_action(self, state: np.ndarray | list[np.ndarray]):
        """Observe a visual field sprite output."""

        if self.show:
            clear_output(wait=True)
            plot(state)

        action = None
        while not isinstance(action, int):
            action_ = input("Select Action: ")
            if action_ in ["w", "a", "s", "d"]:
                if action_ == "w":
                    action = 0
                elif action_ == "s":
                    action = 1
                elif action_ == "a":
                    action = 2
                elif action_ == "d":
                    action = 3
            elif action_ in [str(act) for act in self.action_list]:
                action = int(action_)
            else:
                print("Please try again. Possible actions are below.")
                print(self.action_list)

        return action
