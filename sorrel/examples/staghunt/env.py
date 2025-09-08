"""Experiment wrapper for the Stag Hunt environment.

This module defines :class:`StagHuntEnv`, a subclass of
``sorrel.environment.Environment`` specialised for the stag hunt game.
The environment is responsible for constructing agents, populating the
gridworld with walls, spawn points and resources, and orchestrating
interactions between agents.  It also provides a callback for handling
the payoff and respawning behaviour when agents choose to interact.

The environment relies heavily on the configuration dictionary passed
in at construction time.  See the accompanying design specification for
details on the expected configuration keys.
"""

from __future__ import annotations

import random
from typing import Any, List, Tuple

import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec

from .agents import StagHuntAgent
from .entities import Empty, Wall, Spawn, StagResource, HareResource
from .world import StagHuntWorld


class StagHuntEnv(Environment[StagHuntWorld]):
    """Environment wrapper for the stag hunt game."""

    def __init__(self, world: StagHuntWorld, config: dict | Any) -> None:
        # assign a reference from the world back to this environment so
        # that agents can call ``handle_interaction``.  This is a bit of
        # plumbing to bridge between the agent and environment logic.
        world.environment = self  # type: ignore[attr-defined]
        super().__init__(world, config)

    def setup_agents(self) -> None:
        """Create and configure the agents for the stag hunt experiment.

        The number of agents and their observation/action specifications are
        derived from the configuration.  Each agent is assigned an IQN
        model with configurable hyperparameters.
        """
        cfg = self.config
        # support both dictionary and OmegaConf DictConfig
        if hasattr(cfg, "world"):
            world_cfg = cfg.world
        else:
            world_cfg = cfg.get("world", {})
        if hasattr(cfg, "model"):
            model_cfg = cfg.model
        else:
            model_cfg = cfg.get("model", {})
        n_agents = int(world_cfg.get("num_agents", 2)) #TODO: ideally the default should be 8

        agents: List[StagHuntAgent] = []
        # list all entity kinds present in the environment for one‑hot encoding
        entity_list = [
            "Empty",
            "Wall",
            "Spawn",
            "StagResource",
            "HareResource",
            "StagHuntAgent",
        ]
        for _ in range(n_agents):
            # observation spec: uses partial view with specified vision radius
            vision_radius = int(model_cfg.get("agent_vision_radius", 5))
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=False,
                vision_radius=vision_radius,
            )
            # flatten the observation before feeding into the model
            # The base visual observation produces a tensor of shape
            # (channels, H, W).  We will append three additional features
            # (inventory counts and ready flag) in ``StagHuntAgent.pov``.  To
            # compute the correct input dimension, take the product of
            # observation_spec.input_size and add 3.
            observation_spec.override_input_size(
                np.array(observation_spec.input_size).reshape(1, -1).tolist()
            )
            base_size = int(np.prod(observation_spec.input_size))
            full_input_dim = base_size + 3 #TODO: we can make the additional size a config param

            # action spec: five discrete actions
            action_spec = ActionSpec(
                ["move_forward", "move_backward", "turn_left", "turn_right", "interact"]
            )
            # create a simple IQN model; hyperparameters can be tuned via config
            model = PyTorchIQN(
                input_size=[full_input_dim],
                action_space=action_spec.n_actions,
                layer_size=int(model_cfg.get("layer_size", 250)),
                epsilon=float(model_cfg.get("epsilon", 0.7)),
                device="cpu",
                seed=torch.random.seed(),
                n_frames=int(model_cfg.get("n_frames", 5)),
                n_step=int(model_cfg.get("n_step", 3)),
                sync_freq=int(model_cfg.get("sync_freq", 200)),
                model_update_freq=int(model_cfg.get("model_update_freq", 4)),
                batch_size=int(model_cfg.get("batch_size", 64)),
                memory_size=int(model_cfg.get("memory_size", 1024)),
                LR=float(model_cfg.get("LR", 0.00025)),
                TAU=float(model_cfg.get("TAU", 0.001)),
                GAMMA=float(model_cfg.get("GAMMA", 0.99)),
                n_quantiles=int(model_cfg.get("n_quantiles", 12)),
            )
            agent = StagHuntAgent(
                observation_spec=observation_spec,
                action_spec=action_spec,
                model=model,
            )
            agents.append(agent)
        self.agents = agents

    def populate_environment(self) -> None:
        """Populate the gridworld with terrain, resources and agents.

        Walls are placed around the border of the bottom layer.  Spawn
        locations for agents are recorded and used to respawn players after
        interactions.  Resources are initially populated at random based
        on the ``resource_density`` parameter.  Agents are placed on
        random spawn points.
        """
        world = self.world
        world.reset_spawn_points()

        # prepare a list for valid spawn locations on the top layer
        spawn_locations: List[Tuple[int, int, int]] = []

        for index in np.ndindex(world.map.shape):
            y, x, layer = index
            if layer == 0:
                # bottom layer: walls at border, spawn in interior
                if y == 0 or y == world.height - 1 or x == 0 or x == world.width - 1:
                    world.add(index, Wall())
                else:
                    # interior cells are spawnable and traversable
                    spawn_cell = Spawn()
                    world.add(index, spawn_cell)
                    world.spawn_points.append(index)
            elif layer == 1:
                # top layer: optionally place initial resources
                if (y, x, 0) not in world.spawn_points:
                    world.add(index, Empty())
                else:
                    # spawn points correspond to empty starting cell on top
                    world.add(index, Empty())

        # randomly populate resources on the top layer according to density
        # TODO: should be compatible with a ASCII map defining initial resources
        for (y, x, layer) in world.spawn_points:
            # top layer coordinates
            top = (y, x, 1)
            if np.random.random() < world.resource_density:
                # choose resource type uniformly at random
                if np.random.random() < 0.5:
                    world.add(top, StagResource(world.taste_reward, world.destroyable_health))
                else:
                    world.add(top, HareResource(world.taste_reward, world.destroyable_health))
            else:
                world.add(top, Empty())

        # choose initial agent positions uniformly from spawn points without replacement
        chosen_positions = random.sample(world.spawn_points, len(self.agents))
        for loc, agent in zip(chosen_positions, self.agents):
            # top layer coordinate for agent
            top = (loc[0], loc[1], 1)
            world.add(top, agent)

    # ------------------------------------------------------------------ #
    # Interaction logic                                                   #
    # ------------------------------------------------------------------ #
    def handle_interaction(self, agent: StagHuntAgent, other: StagHuntAgent) -> float:
        """Resolve an interaction between two ready agents.

        Determines each agent's strategy by taking the majority vote over
        their inventories.  Computes the row and column payoffs using
        ``world.payoff_matrix`` (with the column player's payoff being the
        transpose).  Adds a constant bonus for initiating an interaction
        (``interaction_reward`` hyperparameter if present).  Resets both
        agents' inventories, respawns them at random spawn points and
        returns the reward to assign to the initiating agent.

        Parameters
        ----------
        agent : StagHuntAgent
            The agent initiating the interaction (the ``row" player).
        other : StagHuntAgent
            The opponent agent (the ``column" player).

        Returns
        -------
        float
            The reward received by the initiating agent.
        """
        world = self.world
        # determine strategies via majority resource counts; tie breaks in favour of stag
        def majority_resource(inv: dict[str, int]) -> int:
            stag_count = inv.get("stag", 0)
            hare_count = inv.get("hare", 0)
            return 0 if stag_count >= hare_count else 1
        row_strategy = majority_resource(agent.inventory)
        col_strategy = majority_resource(other.inventory)
        # compute payoffs
        row_payoff = world.payoff_matrix[row_strategy][col_strategy]
        col_payoff = world.payoff_matrix[col_strategy][row_strategy]
        # interaction bonus from config; default to 1.0
        # extract interaction bonus from configuration; support both dict and OmegaConf
        cfg = self.config
        if hasattr(cfg, "world"):
            bonus = float(getattr(cfg.world, "interaction_reward", 1.0))
        else:
            bonus = float(cfg.get("world", {}).get("interaction_reward", 1.0))
        # clear inventories and ready flags
        agent.inventory = {"stag": 0, "hare": 0}
        agent.ready = False
        other.inventory = {"stag": 0, "hare": 0}
        other.ready = False
        # remove agents from their current positions on the top layer
        a_loc = agent.location
        o_loc = other.location
        world.remove(a_loc)
        world.remove(o_loc)
        # respawn at random spawn points
        spawn_points = world.spawn_points
        # choose two distinct spawn points at random
        new_a_loc, new_o_loc = random.sample(spawn_points, 2)
        world.add((new_a_loc[0], new_a_loc[1], 1), agent)
        world.add((new_o_loc[0], new_o_loc[1], 1), other)
        # reset orientations
        agent.orientation = 0
        other.orientation = 0
        # accumulate reward for both agents in world.total_reward
        world.total_reward += row_payoff + col_payoff + 2 * bonus
        # return the initiating agent's reward
        return row_payoff + bonus
