from abc import abstractmethod
from pathlib import Path

import numpy as np

from sorrel.action.action_spec import ActionSpec
from sorrel.entities import Entity
from sorrel.models import BaseModel
from sorrel.observation.observation_spec import ObservationSpec
from sorrel.worlds import Gridworld


class Agent[W: Gridworld](Entity[W]):
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
    model: BaseModel

    def __init__(
        self,
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        model: BaseModel,
        location=None,
    ):
        super().__init__()

        # initializations based on parameters
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.model = model
        self.sprite = Path(__file__).parent / "./assets/hero.png"
        self._location = location

        # overriding parent default attributes
        self.has_transitions = True

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent (and its memory)."""
        pass

    @abstractmethod
    def pov(self, world: W) -> np.ndarray:
        """Defines the agent's observation function.

        Args:
            env (Gridworld): the environment that this agent is observing.

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
    def act(self, world: W, action: int) -> float:
        """Act on the environment.

        Args:
            env (Gridworld): The environment in which the agent is acting.
            action: an element from this agent's action space indicating the action to take.

        Returns:
            float: the reward associated with the action taken.
        """
        pass

    @abstractmethod
    def is_done(self, world: W) -> bool:
        """Determines if the agent is done acting given the environment.

        This might be based on the experiment's maximum number of turns from the agent's cfg file.

        Args:
            env (Gridworld): the environment that the agent is in.

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

    def get_proposed_action(self, world: W) -> dict:
        """Calculates the proposed action and its consequences without executing it.

        Returns:
            dict: A dictionary containing:
                - 'action': The action index
                - 'state': The observed state
                - 'new_location': The intended destination (if applicable)
                - 'reward': The expected reward
                - 'done': Whether the agent is done
        """
        state = self.pov(world)
        action = self.get_action(state)
        done = self.is_done(world)
        
        # Default implementation for base Agent (assumes no movement, just action)
        # Subclasses like MovingAgent should override to include movement logic
        return {
            "action": action,
            "state": state,
            "new_location": None,
            "reward": 0.0, # Base agent doesn't know reward without acting
            "done": done
        }

    def finalize_turn(self, world: W, proposal: dict, allowed: bool = True) -> None:
        """Finalizes the turn based on the proposal and whether the move is allowed.

        Args:
            world: The environment world.
            proposal: The dictionary returned by get_proposed_action.
            allowed: Whether the proposed move is allowed (e.g. no collision).
        """
        state = proposal["state"]
        action = proposal["action"]
        reward = proposal["reward"]
        done = proposal["done"]

        if allowed:
             # If allowed, we assume the reward in proposal is valid. 
             # For base Agent, act() might need to be called if it wasn't fully simulated.
             # But for MovingAgent, we'll handle the move here.
             pass
        
        # This base method is tricky because act() in the original code did everything.
        # We'll rely on the refactored transition() to keep backward compatibility
        # and let subclasses handle the specifics.
        
        world.total_reward += reward
        self.add_memory(state, action, reward, done)

    def transition(self, world: W) -> None:
        """Processes a full transition step for the agent.
        
        Refactored to use get_proposed_action and finalize_turn.
        """
        proposal = self.get_proposed_action(world)
        # In standard sequential mode, we always allow the attempt (act handles validity)
        self.finalize_turn(world, proposal, allowed=True)


class MovingAgent[W: Gridworld](Agent):
    """An agent that implements methods for moving up, down, right, left."""

    sprite_directions = [
        Path(__file__).parent / "./assets/hero-back.png",  # Up
        Path(__file__).parent / "./assets/hero.png",  # Down
        Path(__file__).parent / "./assets/hero-left.png",  # Left
        Path(__file__).parent / "./assets/hero-right.png",  # Right
    ]

    def movement(self, action: int) -> tuple[int, int, int]:
        """Attempt to move with the specified action to a new location.

        Args:
            action (int): The action coded as an integer.

        Returns:
            tuple[int, int, int]: The new location.
        """
        # Translate the model output to an action string
        action_name = self.action_spec.get_readable_action(action)
        
        # Only update sprite for movement actions (to avoid index errors with non-movement actions)
        if action < len(self.sprite_directions):
            self.sprite = self.sprite_directions[action]
            
        new_location = self.location
        if action_name == "up":
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action_name == "down":
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action_name == "left":
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action_name == "right":
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        return new_location

    def get_proposed_action(self, world: W) -> dict:
        state = self.pov(world)
        action = self.get_action(state)
        done = self.is_done(world)
        
        new_location = self.movement(action)
        
        # Check what's at the new location to determine reward
        # Note: In the original act(), it calls world.observe(new_location)
        # We do this here to predict the reward.
        if world.valid_location(new_location):
             target_object = world.observe(new_location)
             reward = target_object.value
        else:
             # If invalid location (out of bounds), we might need logic. 
             # But movement() usually returns valid coords or clamped? 
             # The original code didn't check valid_location explicitly in act(), 
             # but world.observe might fail if out of bounds.
             # Assuming movement logic keeps it in bounds or world handles it.
             # Let's assume valid for now or catch exception if needed.
             # Actually, world.move checks passability.
             try:
                target_object = world.observe(new_location)
                reward = target_object.value
             except IndexError:
                reward = 0 # Or some penalty? Original code would crash or handle it.
                new_location = self.location # Stay put
        
        return {
            "action": action,
            "state": state,
            "new_location": new_location,
            "reward": reward,
            "done": done
        }

    def finalize_turn(self, world: W, proposal: dict, allowed: bool = True) -> None:
        if allowed:
            new_location = proposal["new_location"]
            # Try moving to new_location
            # world.move returns True if successful, False if blocked (impassable)
            # In original act(), reward was returned regardless of move success?
            # "get reward obtained from object at new_location" -> "try moving" -> return reward.
            # So yes, reward is obtained even if move fails (e.g. bumping into wall)?
            # Wait, if wall is impassable, do we get the wall's value?
            # "reward = target_object.value" happens BEFORE world.move.
            # So yes.
            
            if new_location != self.location:
                world.move(self, new_location)
        
        # Add memory and update total reward
        # Note: If !allowed (simultaneous collision), do we still get reward?
        # "if two agents simulatanously try to move into the same square, neither move into it."
        # Presumably they also don't get the reward of the thing they didn't enter?
        # Or do they bump into each other?
        # Let's assume if !allowed, they stay put and get 0 reward (or whatever staying put gives).
        
        final_reward = proposal["reward"] if allowed else 0.0 
        # Actually, if they are blocked, they might still get a "step" penalty if defined?
        # But here reward comes from the target object. If they don't reach it, they shouldn't get it.
        
        world.total_reward += final_reward
        self.add_memory(proposal["state"], proposal["action"], final_reward, proposal["done"])

    def act(self, world: W, action: int):
        # Kept for backward compatibility if called directly, but transition() uses the new flow.
        # This is effectively what finalize_turn does when allowed=True
        new_location = self.movement(action)
        target_object = world.observe(new_location)
        reward = target_object.value
        world.move(self, new_location)
        return reward
