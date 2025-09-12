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
from typing import Any

import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment
from sorrel.examples.staghunt.agents import StagHuntAgent
from sorrel.examples.staghunt.entities import (
    Empty,
    HareResource,
    Spawn,
    StagResource,
    Wall,
    Sand,
)
from sorrel.examples.staghunt.world import StagHuntWorld
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec


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

        The number of agents and their observation/action specifications are derived
        from the configuration.  Each agent is assigned an IQN model with configurable
        hyperparameters.
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
        n_agents = int(
            world_cfg.get("num_agents", 2)
        )  # TODO: ideally the default should be 8
        if hasattr(world_cfg, "interaction_reward"):
            interaction_reward = world_cfg.interaction_reward
        else:
            interaction_reward = 1.0

        agents = []
        # list all entity kinds present in the environment for one‑hot encoding
        entity_list = [
            "Empty",
            "Wall",
            "Spawn",
            "StagResource",
            "HareResource",
            "StagHuntAgent",
            "Sand"
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
            full_input_dim = (
                base_size + 3
            )  # TODO: we can make the additional size a config param

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
                interaction_reward=interaction_reward,
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

        for y, x, layer in np.ndindex(world.map.shape):
            index = (y, x, layer)
            if y == 0 or y == world.height - 1 or x == 0 or x == world.width - 1:
                world.add(index, Wall())
            elif layer == 0:
                # interior cells are spawnable and traversable
                if (y, x, 1) in world.agent_spawn_points:
                    world.add(index, Spawn())
                elif (y, x, 1) not in world.resource_spawn_points:
                    world.add(index, Sand())
            elif layer == 1:
                # top layer: optionally place initial resources
                if (y, x, 1) not in world.agent_spawn_points:
                    if np.random.random() < world.resource_density:
                        # choose resource type uniformly at random
                        world.resource_spawn_points.append((y, x, 1))
                    else:
                        world.add(index, Empty())
                # else:
                #     # spawn points correspond to empty starting cell on top
                #     world.add(index, Empty())
                

        # randomly populate resources on the top layer according to density
        # TODO: should be compatible with a ASCII map defining initial resources
        for y, x, layer in world.resource_spawn_points:
            # top layer coordinates
            top = (y, x, 1)
            # choose resource type uniformly at random
            if np.random.random() < 0.5:
                world.add(top, StagResource(world.taste_reward, world.destroyable_health))
            else:
                world.add(top, HareResource(world.taste_reward, world.destroyable_health))
    

        # choose initial agent positions uniformly from spawn points without replacement
        chosen_positions = random.sample(world.agent_spawn_points, len(self.agents))
        for loc, agent in zip(chosen_positions, self.agents):
            # top layer coordinate for agent
            top = (loc[0], loc[1], 1)
            world.add(top, agent)
