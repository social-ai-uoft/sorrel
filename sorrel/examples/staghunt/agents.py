"""Agent implementation for the stag hunt environment.

This module defines :class:`StagHuntAgent`, a subclass of
``sorrel.agents.Agent`` that encapsulates the behaviour of players in the
stag hunt arena.  Each agent maintains an orientation and a small
inventory of collected resources (stag and hare).  The agent can move
forward or backward relative to its facing direction, turn left or
right, and fire an interaction beam to engage in a stag‑hunt game with
another ready agent.  The agent obtains small taste rewards upon
collecting resources and larger payoffs via the interaction matrix when
engaging another player.

The logic for resolving the interaction, computing payoffs and
respawning players is delegated to the environment via the
``handle_interaction`` method; this agent simply decides when to fire
the beam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from sorrel.agents import Agent
from sorrel.action.action_spec import ActionSpec
from sorrel.observation.observation_spec import ObservationSpec
from sorrel.models.pytorch import PyTorchIQN
from sorrel.worlds import Gridworld

from .entities import Empty, StagResource, HareResource
from .world import StagHuntWorld


class StagHuntAgent(Agent[StagHuntWorld]):
    """An agent that plays the stag hunt.

    Parameters
    ----------
    observation_spec : ObservationSpec
        Specification of the agent's observation space.
    action_spec : ActionSpec
        Specification of the agent's discrete action space.  The
        ``actions`` list passed into the spec should define readable
        action names in the order expected by the model.
    model : PyTorchIQN
        A quantile regression DQN model used to select actions.  Any
        ``sorrel`` compatible model may be passed.
    """

    # Mapping from orientation to vector offset (dy, dx)
    ORIENTATION_VECTORS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),  # north
        1: (0, 1),   # east
        2: (1, 0),   # south
        3: (0, -1),  # west
    }

    def __init__(self, observation_spec: ObservationSpec, action_spec: ActionSpec, model: PyTorchIQN):
        super().__init__(observation_spec, action_spec, model)
        # assign a default sprite; can be overridden externally
        self.sprite = Path(__file__).parent / "./assets/agent.png"

        # orientation encoded as 0: north, 1: east, 2: south, 3: west
        self.orientation: int = 0
        # inventory counts for resources; keys are "stag" and "hare"
        self.inventory: Dict[str, int] = {"stag": 0, "hare": 0}
        # whether the agent is ready to interact (has at least one resource)
        self.ready: bool = False

    # ------------------------------------------------------------------ #
    # Agent lifecycle methods                                             #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset the agent state at the start of an episode.

        Clears the inventory, resets the orientation and notifies the model
        that a new episode has begun.
        """
        self.orientation = 0
        self.inventory = {"stag": 0, "hare": 0}
        self.ready = False
        # reset the underlying model (e.g., clear memory of past states)
        self.model.reset()

    def pov(self, world: StagHuntWorld) -> np.ndarray:
        """Return the agent's observation vector.

        The base observation is a one‑hot encoding of the cells within the
        agent's vision radius, provided by the observation specification.  We
        augment this flattened visual field with three additional values:

        1. The number of stag resources currently held in the agent's
           inventory.
        2. The number of hare resources currently held in the agent's
           inventory.
        3. A binary flag indicating whether the agent is ready to shoot
           (i.e. has at least one resource in inventory).

        These extra features allow the model to condition its behaviour on
        internal state, reproducing the ``InventoryObserver`` and
        ``ReadyToShootObservation`` channels described in the design spec.
        """
        # observe returns an array of shape (channels, H, W)
        obs = self.observation_spec.observe(world, self.location)
        # flatten to a vector for the model input
        flat = obs.reshape(-1)
        # append inventory counts and ready flag
        inv_stag = self.inventory.get("stag", 0)
        inv_hare = self.inventory.get("hare", 0)
        ready_flag = 1 if self.ready else 0
        extra = np.array([inv_stag, inv_hare, ready_flag], dtype=flat.dtype)
        return np.concatenate([flat, extra]).reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Select an action using the underlying model.

        A stack of previous states is concatenated internally by the model's
        memory.  The model returns an integer index into the action
        specification.
        """
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))
        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    def act(self, world: StagHuntWorld, action: int) -> float:
        """Execute the chosen action in the environment and return the reward.

        The agent interprets the model output as a human‑readable action
        string via the ``action_spec`` and then performs movement,
        turning or interaction.  The reward arises from picking up
        resources (taste reward) and from interacting with another agent
        (stag‑hunt payoff handled by the environment).
        """
        action_name = self.action_spec.get_readable_action(action)
        reward = 0.0

        # handle movement forward/backward relative to orientation
        if action_name == "move_forward" or action_name == "move_backward":
            dy, dx = StagHuntAgent.ORIENTATION_VECTORS[self.orientation]
            # invert direction for backward movement
            if action_name == "move_backward":
                dy, dx = (-dy, -dx)
            # compute target location (same layer)
            y, x, z = self.location
            new_pos = (y + dy, x + dx, z)
            # attempt to move into the new position if valid
            if world.valid_location(new_pos):
                # pick up reward associated with the entity on the top layer
                target_entity = world.observe(new_pos)
                if isinstance(target_entity, StagResource) or isinstance(target_entity, HareResource):
                    # collect resource: add to inventory and mark ready
                    self.inventory[target_entity.name] += 1
                    self.ready = True
                    reward += target_entity.value  # taste reward
                # move into the cell (if passable)
                world.move(self, new_pos)
        elif action_name == "turn_left":
            # rotate orientation counter‑clockwise
            self.orientation = (self.orientation - 1) % 4
        elif action_name == "turn_right":
            # rotate orientation clockwise
            self.orientation = (self.orientation + 1) % 4
        elif action_name == "interact":
            # fire a beam in the facing direction if ready and allow the
            # environment to process the interaction
            if self.ready:
                # compute beam cells
                dy, dx = StagHuntAgent.ORIENTATION_VECTORS[self.orientation]
                beam_cells = []
                y, x, z = self.location
                for step in range(1, world.beam_length + 1):
                    target = (y + dy * step, x + dx * step, z)
                    if not world.valid_location(target):
                        break
                    # stop if a wall is encountered on the bottom layer
                    bottom = (target[0], target[1], 0)
                    if not world.map[bottom].passable:
                        break
                    beam_cells.append(target)
                # check if any agent is hit
                #TODO: interaction should consider when there are more than 2 agents; we can first figure out all agens in the beam
                # and then randomly pick one to interact with
                for cell in beam_cells:
                    entity = world.observe(cell)
                    if isinstance(entity, StagHuntAgent) and entity.ready:
                        # delegate payoff computation to the environment
                        reward += world.environment.handle_interaction(self, entity)
                        break
        # return accumulated reward from this action
        return reward

    def is_done(self, world: StagHuntWorld) -> bool:
        """Check whether this agent is done acting.

        Agents act until the world signals termination via ``world.is_done``.
        """
        return world.is_done
