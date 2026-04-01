"""PettingZoo AEC TreasureHunt environment for CleanRL-style training."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.examples.treasurehunt_cleanrl.agents import TreasureHuntCleanRLAgent
from sorrel.examples.treasurehunt_cleanrl.entities import EmptyEntity, Sand, Wall
from sorrel.examples.treasurehunt_cleanrl.world import TreasureHuntCleanRLWorld
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.pettingzoo import SorrelAECEnv, apply_recommended_wrappers
from sorrel.utils.visualization import image_from_array, render_sprite

ACTIONS = ["up", "down", "left", "right"]
ENTITY_LIST = [
    "EmptyEntity",
    "Wall",
    "Gem",
    "Bone",
    "Food",
    "TreasureHuntCleanRLAgent",
]


class TreasureHuntCleanRLEnv(SorrelAECEnv):
    """AECEnv wrapper for Sorrel TreasureHunt with turn-based stepping."""

    metadata = {"name": "treasurehunt_cleanrl_v0", "is_parallelizable": False}

    def __init__(
        self, config: DictConfig | dict, render_mode: str | None = "rgb_array"
    ):
        super().__init__(render_mode=render_mode)
        if isinstance(config, DictConfig):
            self.config = config
        else:
            self.config = OmegaConf.create(config)

        self.max_cycles = int(self.config.env.max_cycles)
        self.turn = 0
        self.world: TreasureHuntCleanRLWorld

        self._pending_rewards: dict[str, float] = {}
        self._found_treasure = False

    def _sorrel_reset(self) -> None:
        self.world = TreasureHuntCleanRLWorld(
            config=self.config,
            default_entity=EmptyEntity(),
        )
        self.turn = 0
        self.world.turn = 0
        self.world.is_done = False
        self._found_treasure = False

        action_spec = ActionSpec(ACTIONS)
        sorrel_agents: dict[str, TreasureHuntCleanRLAgent] = {}
        observation_spaces: dict[str, spaces.Space] = {}
        action_spaces: dict[str, spaces.Space] = {}

        num_agents = int(self.config.agent.num_agents)
        vision_radius = int(self.config.agent.vision_radius)

        for index in range(num_agents):
            observation_spec = OneHotObservationSpec(
                ENTITY_LIST,
                full_view=False,
                vision_radius=vision_radius,
            )
            observation_spec.override_input_size(
                (int(np.prod(observation_spec.input_size)),)
            )

            agent_id = f"agent_{index}"
            agent = TreasureHuntCleanRLAgent(
                observation_spec=observation_spec,
                action_spec=action_spec,
            )
            sorrel_agents[agent_id] = agent

            obs_dim = int(np.prod(observation_spec.input_size))
            observation_spaces[agent_id] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            action_spaces[agent_id] = spaces.Discrete(action_spec.n_actions)

        self._populate_world(list(sorrel_agents.values()))
        self._register_agents(sorrel_agents, observation_spaces, action_spaces)
        self._pending_rewards = {agent_id: 0.0 for agent_id in sorrel_agents}

    def _populate_world(self, agents: list[TreasureHuntCleanRLAgent]) -> None:
        valid_spawn_locations: list[tuple[int, int, int]] = []

        for index in np.ndindex(self.world.map.shape):
            y_coord, x_coord, layer = index
            if (
                y_coord in [0, self.world.height - 1]
                or x_coord in [0, self.world.width - 1]
            ) and layer == 1:
                self.world.add(index, Wall())
            elif layer == 0:
                self.world.add(index, Sand())
            elif layer == 1:
                valid_spawn_locations.append(index)

        if len(valid_spawn_locations) < len(agents):
            raise ValueError("Not enough valid spawn points for all agents.")

        sampled_indices = np.random.choice(
            len(valid_spawn_locations),
            size=len(agents),
            replace=False,
        )
        for sampled_index, agent in zip(sampled_indices, agents):
            self.world.add(tuple(valid_spawn_locations[sampled_index]), agent)

    def _compute_obs(self, agent_id: str) -> np.ndarray:
        return self.sorrel_agents[agent_id].pov(self.world).astype(np.float32)

    def _apply_action(self, agent_id: str, action: int) -> None:
        if action is None:
            raise ValueError("Action cannot be None for a live agent.")

        acting_agent = self.sorrel_agents[agent_id]
        reward = float(acting_agent.act(self.world, int(action)))
        self.world.total_reward += reward
        self._pending_rewards = {
            possible_agent: 0.0 for possible_agent in self.possible_agents
        }
        self._pending_rewards[agent_id] = reward

        if acting_agent.last_interaction_kind == "Gem":
            self._found_treasure = True

        for possible_agent in self.possible_agents:
            self.infos[possible_agent] = {
                "turn": self.turn,
                "treasure_found": self._found_treasure,
            }
        self.infos[agent_id]["immediate_reward"] = reward
        self.infos[agent_id]["action"] = int(action)
        self.infos[agent_id]["interaction_kind"] = acting_agent.last_interaction_kind

    def _advance_world_after_agent_act(self, agent_id: str) -> None:
        self.turn += 1
        self.world.turn = self.turn

        for _, entity in np.ndenumerate(self.world.map):
            if entity.has_transitions and not isinstance(entity, Agent):
                entity.transition(self.world)

    def _compute_terminations_truncations(
        self,
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        terminated = bool(self._found_treasure)
        truncated = bool(self.turn >= self.max_cycles)

        if terminated or truncated:
            self.world.is_done = True

        return (
            {agent_id: terminated for agent_id in self.agents},
            {agent_id: truncated for agent_id in self.agents},
        )

    def _compute_rewards(self) -> dict[str, float]:
        return {
            agent_id: float(self._pending_rewards.get(agent_id, 0.0))
            for agent_id in self.agents
        }

    def render(self):
        frame = image_from_array(render_sprite(self.world))
        return np.asarray(frame)

    def close(self):
        return


def raw_env(
    config: DictConfig | dict,
    render_mode: str | None = "rgb_array",
) -> TreasureHuntCleanRLEnv:
    return TreasureHuntCleanRLEnv(config=config, render_mode=render_mode)


def env(config: DictConfig | dict, render_mode: str | None = "rgb_array"):
    return apply_recommended_wrappers(raw_env(config=config, render_mode=render_mode))
