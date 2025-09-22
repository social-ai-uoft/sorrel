"""Cooking environment implementation.

This module defines ``CookingEnv`` – a concrete ``Environment`` subclass that
uses ``CookingWorld`` and the custom ``CookingAgent`` defined in the example
package.  The implementation follows the checklist in ``PLAN.md`` and mirrors the
structure of the existing ``CleanupEnv`` example.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.environment import Environment
from sorrel.examples.cleanup.entities import Wall
from sorrel.examples.cooking.agents import CookingAgent
from sorrel.examples.cooking.entities import (
    Counter,
    EmptyEntity,
    Plate,
    Stove,
    Tomato,
    Trash,
)
from sorrel.examples.cooking.world import CookingWorld
from sorrel.models.base_model import RandomModel
from sorrel.models.pytorch import PyTorchIQN

# ---------------------------------------------------------------------------
# Entity list for the observation spec – mirrors the list used in the cleanup
# example.  ``EmptyEntity`` and ``Wall`` are imported from the cleanup example
# because they are shared utilities.
# ---------------------------------------------------------------------------
ENTITY_LIST: List[str] = [
    "EmptyEntity",
    "Wall",
    "Stove",
    "Counter",
    "Plate",
    "Trash",
    "Onion",
    "Tomato",
    "CookingAgent",
]


class CookingEnv(Environment[CookingWorld]):
    """Environment inspired by Overcooked."""

    def setup_agents(self) -> None:
        """Create ``CookingAgent`` instances based on the provided config.

        Expected config layout:

            agent:
                agent:
                    num: <int>          # number of chefs
                    obs:
                        vision: <int>
                        n_frames: <int>
        """
        agents: List[Agent] = []
        num_agents = int(self.config.agent.agent.num)
        vision = int(self.config.agent.agent.obs.vision)
        n_frames = int(self.config.agent.agent.obs.n_frames)

        for _ in range(num_agents):
            observation_spec = self._create_observation_spec(vision)
            action_spec = ActionSpec(
                [
                    "up",
                    "down",
                    "left",
                    "right",
                    "pick",
                    "place",
                    "cook",
                    "serve",
                    "wash",
                ]
            )

            model_cfg = getattr(self.config.model, "iqn", None)
            if model_cfg is not None:
                model = PyTorchIQN(
                    input_size=observation_spec.input_size,
                    action_space=action_spec.n_actions,
                    seed=torch.random.seed(),
                    n_frames=n_frames,
                    **self.config.model.iqn.parameters,
                )
            else:
                # Fallback to a deterministic random model.
                model = RandomModel(
                    input_size=observation_spec.input_size,
                    action_space=action_spec.n_actions,
                    memory_size=1000,
                )

            agents.append(
                CookingAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                )
            )

        self.agents = agents

    def _create_observation_spec(self, vision: int):
        """Helper to construct an ``CookingObservation`` if needed."""
        # Import lazily to avoid circular imports when the docs are built.
        from sorrel.examples.cooking.observation_spec import (
            CookingObservation,  # type: ignore
        )

        obs = CookingObservation(entity_list=ENTITY_LIST, vision_radius=vision)
        obs.override_input_size((int(np.prod(obs.input_size)),))
        return obs

    def populate_environment(self) -> None:
        """Place static entities and agents on the grid.

        A minimal layout is created with some counter space, stove, plate, trash, and
        random agent placements.
        """
        h, w = self.world.height, self.world.width
        # Border walls
        for i in range(h):
            for j in range(w):
                if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                    self.world.add((i, j, self.world.object_layer), Wall())
        # Central stove
        stove_loc = (h // 2, w // 2, self.world.object_layer)
        self.world.add(stove_loc, Stove())
        # Counter to the left of stove
        for i in range(1, h // 2):
            counter_loc = (h // 2, w // 2 - i, self.world.object_layer)
            counter = Counter()
            if i == 1:
                counter.place(Tomato())
            self.world.add(counter_loc, counter)

        # Plate to the right of stove
        plate_loc = (h // 2, w // 2 + 1, self.world.object_layer)
        self.world.add(plate_loc, Plate())
        # Trash in top-left corner (inside walls)
        trash_loc = (1, 1, self.world.object_layer)
        self.world.add(trash_loc, Trash())
        # Randomly place agents on any empty tile (passable and not a wall)
        empty_tiles = []
        for idx, entity in np.ndenumerate(self.world.map):
            if (
                isinstance(entity, EmptyEntity)
                and entity.passable
                and idx[2] == self.world.object_layer
            ):
                empty_tiles.append(idx)
        # Ensure we have at least as many empty tiles as agents.
        assert len(empty_tiles) >= len(self.agents), "Not enough free tiles for agents"
        # Randomly place agents on the board.
        agent_locations_indices = np.random.choice(
            len(empty_tiles), size=len(self.agents), replace=False
        )
        agent_locations = [empty_tiles[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)
