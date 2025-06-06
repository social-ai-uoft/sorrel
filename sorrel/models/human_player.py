from typing import Sequence

import numpy as np
from IPython.display import clear_output

from sorrel.buffers import Buffer
from sorrel.models import BaseModel
from sorrel.observation.observation_spec import ObservationSpec
from sorrel.utils.visualization import plot, render_sprite
from sorrel.worlds import Gridworld


class HumanObservation(ObservationSpec[str]):

    def observe(self, world: Gridworld, location: tuple | None = None) -> np.ndarray:

        sprite = render_sprite(
            world=world,
            location=location,
        )

        return np.array(sprite)


class HumanMemory(Buffer):
    def current_state(self) -> np.ndarray:
        return self.states[-1]


class HumanPlayer(BaseModel):
    """Model subclass for a human player."""

    def __init__(
        self,
        input_size: int | Sequence[int],
        action_space: int,
        memory_size: int,
        show: bool = True,
    ):

        if isinstance(input_size, int):
            raise ValueError(
                "Input size must be a sequence of integers for the human player."
            )

        self.action_list = np.arange(action_space)
        self.show = show
        self.tile_size = 16
        self.num_channels = 4
        # Shape the input for use with the dummy memory function and the observation function.
        self.input_size = input_size
        _input_size = (
            1,
            self.input_size[0]
            * self.input_size[1]
            * self.input_size[2]
            * (self.tile_size**2)
            * self.num_channels,
        )
        # We will slice off the human memory zero input.
        self.SLICE = np.prod(_input_size)
        self.memory = HumanMemory(1, _input_size, 1)

    def take_action(self, state: np.ndarray):
        """Observe a visual field sprite output."""

        if isinstance(self.input_size, int):
            raise ValueError(
                "Input size must be a sequence of integers for the human player."
            )

        if self.show:
            clear_output(wait=True)

            # Reshape the input to return to the original image (isn't python fun?)
            state = state[:, self.SLICE :]
            state = state.reshape(
                (
                    -1,
                    self.input_size[0] * self.tile_size,
                    self.input_size[1] * self.tile_size,
                    self.num_channels,
                )
            )
            state = np.array(state, dtype=int)
            state_ = []
            for i in range(state.shape[0]):
                state_.append(state[i, :, :, :])
            plot(state_)

        action = None
        num_retries = 0
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
            elif action_ == "quit":
                raise KeyboardInterrupt("Quitting...")
            else:
                num_retries += 1
                if num_retries > 5:
                    raise KeyboardInterrupt("Too many invalid inputs. Quitting...")
                print("Please try again. Possible actions are below.")
                print(np.concatenate((self.action_list, ["quit"])))

        return action
