from typing import Sequence

import numpy as np

from sorrel.observation.observation_spec import ObservationSpec
from sorrel.worlds.gridworld import Gridworld


class TaxiObservationSpec(ObservationSpec[np.ndarray, Gridworld]):
    """Observation specification for the taxi environment.

    For taxi agents whose observations are represented as one-hot encodings.
    """

    def __init__(
        self,
        passenger_loc: int,
        destination_loc: int,
        env_dims: Sequence[int],
        entity_list: list[str] = [""],
        full_view: bool = True,
        vision_radius: int | None = None,
    ):
        super().__init__(entity_list, full_view, vision_radius, env_dims)

        self.map_size = env_dims[0] - 2

        self.passenger_loc = passenger_loc
        self.destination_loc = destination_loc

    def encode_state(self, row: int, col: int, passenger: int, destination: int) -> int:
        return ((row * self.map_size + col) * 5 + passenger) * 4 + destination

    def to_one_hot(self, state: int, n_states: int) -> np.ndarray:
        one_hot = np.zeros(n_states, dtype=np.float32)
        one_hot[state] = 1.0
        return one_hot

    def observe(
        self,
        world: Gridworld,
        location: tuple | None = None,
    ) -> np.ndarray:
        """Returns an observation in the form of a one-hot encoding including the taxi's
        location, passenger location, and destination location."""

        if location is None:
            raise ValueError("Location must be provided for TaxiObservationSpec.")

        enc_state = self.encode_state(
            location[0] - 1,
            location[1] - 1,
            self.passenger_loc,
            self.destination_loc,
        )
        vec = [self.to_one_hot(enc_state, n_states=self.input_size[0])]
        vec = np.array(vec).reshape(1, -1)
        return vec
