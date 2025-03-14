from abc import abstractmethod

import numpy as np

from sorrel.action.action_spec import ActionSpec
from sorrel.entities import Entity
from sorrel.environments import GridworldEnv
from sorrel.models import SorrelModel
from sorrel.observation.observation_spec import ObservationSpec


class Agent(Entity):
    """An abstract class for agents, a special type of entities.

    Note that this is a subclass of :py:class:`agentarium.entities.Entity`.

    Attributes:
        observation_spec: The observation specification to use for this agent.
        model: The model that this agent uses.
        action_space: The range of actions that the agent is able to take, represented by a list of integers.

            .. warning::
                Currently, each element in :attr:`action_space` should be the index of that element.
                In other words, it should be a list of neighbouring integers in increasing order starting at 0.

                For example, if the agent has 4 possible actions, it should have :attr:`action_space = [0, 1, 2, 3]`.

    Attributes that override parent (Entity)'s default values:
        - :attr:`has_transitions` - Defaults to True instead of False.
    """

    observation_spec: ObservationSpec
    action_spec: ActionSpec
    model: SorrelModel

    def __init__(
        self,
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        model: SorrelModel,
        location=None,
    ):
        # initializations based on parameters
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.model = model
        self.location = location

        super().__init__()

        # overriding parent default attributes
        self.has_transitions = True

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent (and its memory)."""
        pass

    @abstractmethod
    def pov(self, env: GridworldEnv) -> np.ndarray:
        """Defines the agent's observation function.

        Args:
            env (GridworldEnv): the environment that this agent is observing.

        Returns:
            torch.Tensor: the observed state.
        """
        pass

    @abstractmethod
    def get_action(self, state: np.ndarray) -> int:
        """Gets the action to take based on the current state from the agent's model.

        Args:
            state (torch.Tensor): the current state observed by the agent.

        Returns:
            int: the action chosen by the agent's model given the state.
        """
        pass

    @abstractmethod
    def act(self, env: GridworldEnv, action: int) -> float:
        """Act on the environment.

        Args:
            action: an element from this agent's action space indicating the action to take.

        Returns:
            float: the reward associated with the action taken.
        """
        pass

    @abstractmethod
    def is_done(self, env: GridworldEnv) -> bool:
        """Determines if the agent is done acting given the environment.

        This might be based on the experiment's maximum number of turns from the agent's cfg file.

        Args:
            env (GridworldEnv): the environment that the agent is in.

        Returns:
            bool: whether the agent is done acting. False by default.
        """
        pass

    def add_memory(
        self, state: np.ndarray, action: int, reward: float, done: bool
    ) -> None:
        """Add an experience to the memory.

        Args:
            state (np.ndarray): the state to be added.
            action (int): the action taken by the agent.
            reward (float): the reward received by the agent.
            done (bool): whether the episode terminated after this experience.
        """
        self.model.memory.add(state, action, reward, done)

    def transition(self, env: GridworldEnv) -> None:
        """Processes a full transition step for the agent.

        This function does the following:
        - Get the current state from the environment through :meth:`pov()`
        - Get the action based on the current state through :meth:`get_action()`
        - Changes the environment based on the action and obtains the reward through :meth:`act()`
        - Determines if the agent is done through :meth:`is_done()`

        Args:
            env (GridworldEnv): the environment that this agent is acting in.
        """
        state = self.pov(env)
        action = self.get_action(state)
        reward = self.act(env, action)
        done = self.is_done(env)
        self.add_memory(state, action, reward, done)
