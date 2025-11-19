import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.agents.agent import Agent
from sorrel.entities.entity import Entity
from sorrel.environment import Environment

# sorrel imports
from sorrel.examples.territory.agents import TerritoryAgent, TerritoryObservation
from sorrel.examples.territory.entities import EmptyEntity, Province, River
from sorrel.examples.territory.gcn_iqn import GCNiRainbowModel, GCNObservationSpec
from sorrel.examples.territory.world import TerritoryWorld
from sorrel.location import Location
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.logging import Logger


class TerritoryEnvironment(Environment[TerritoryWorld]):
    """The experiment for the territory environment."""

    def __init__(self, world: TerritoryWorld, config: dict) -> None:
        super().__init__(world, config, stop_if_done=True)
        self.current_attacks = 0
        self.current_epoch = 1
        self.data = pd.DataFrame(columns=["epoch", "attacks"])

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents."""

        agent_num = 2
        agents = []
        is_blue_agent = False

        for _ in range(agent_num):
            # create the observation spec
            entity_list = [
                "blue_province",
                "red_province",
                "blue_capital",
                "red_capital",
                "river",
            ]

            observation_spec = None
            model = None

            # create the action spec
            action_spec = ActionSpec(["harvest", "attack"])

            if self.config.model.gcn:
                observation_spec = GCNObservationSpec(
                    entity_list=entity_list,
                    full_view=True,
                    env_dims=(self.world.height, self.world.width),
                    # side="red" if is_blue_agent else "blue",
                )

                model = GCNiRainbowModel(
                    input_size=observation_spec.input_size[1],
                    action_space=action_spec.n_actions,
                    layer_size=250,
                    # epsilon=0.6,
                    epsilon=0.85,
                    device="cpu",
                    seed=torch.random.seed(),
                    # n_frames=5,
                    n_frames=1,
                    n_step=3,
                    sync_freq=200,
                    model_update_freq=4,
                    batch_size=64,
                    # batch_size=32,
                    # memory_size=1024,
                    memory_size=1024,
                    # LR=0.00025,
                    LR=0.001,
                    TAU=0.001,
                    GAMMA=0.99,
                    n_quantiles=12,
                    world=self.world,
                    side="red" if is_blue_agent else "blue",
                )
            else:
                observation_spec = TerritoryObservation(
                    entity_list=entity_list,
                    env_dims=(self.world.height, self.world.width),
                    side="red" if is_blue_agent else "blue",
                )

                model = PyTorchIQN(
                    input_size=observation_spec.input_size,
                    action_space=action_spec.n_actions,
                    layer_size=250,
                    # epsilon=0.6,
                    epsilon=0.85,
                    device="cpu",
                    seed=torch.random.seed(),
                    # n_frames=5,
                    n_frames=1,
                    n_step=3,
                    sync_freq=200,
                    model_update_freq=4,
                    batch_size=64,
                    # batch_size=32,
                    memory_size=1024,
                    # LR=0.00025,
                    LR=0.001,
                    TAU=0.001,
                    GAMMA=0.99,
                    n_quantiles=12,
                )

            if not is_blue_agent:
                agents.append(
                    TerritoryAgent(
                        observation_spec=observation_spec,
                        action_spec=action_spec,
                        model=model,
                        side="blue",
                    )
                )
                is_blue_agent = True
            else:
                agents.append(
                    TerritoryAgent(
                        observation_spec=observation_spec,
                        action_spec=action_spec,
                        model=model,
                        side="red",
                    )
                )

        self.agents = agents

    def populate_environment(self):
        capital_locations = []
        blue_province_locations = []
        red_province_locations = []
        river_locations = []

        formation = "halves"

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index

            if formation == "random":
                if np.random.rand() < 0.1 and len(capital_locations) < len(self.agents):
                    capital_locations.append(index)
                else:
                    if np.random.rand() < 0.5:
                        blue_province_locations.append(index)
                    else:
                        red_province_locations.append(index)
            elif formation == "halves":
                capital_locations.append((self.world.height // 2, 0, 0))
                capital_locations.append(
                    (self.world.height // 2, self.world.width - 1, 0)
                )
                if x < self.world.width // 2:
                    if index not in capital_locations:
                        blue_province_locations.append(index)
                else:
                    if index not in capital_locations:
                        red_province_locations.append(index)
            elif formation == "halves_river":
                capital_locations.append((self.world.height // 2, 0, 0))
                capital_locations.append(
                    (self.world.height // 2, self.world.width - 1, 0)
                )
                if (
                    x == self.world.width // 2 or x == (self.world.width // 2) - 1
                ) and np.random.rand() < 1:
                    river_locations.append(index)
                elif x < self.world.width // 2:
                    if index not in capital_locations:
                        blue_province_locations.append(index)
                else:
                    if index not in capital_locations:
                        red_province_locations.append(index)

        for loc in blue_province_locations:
            loc = tuple(loc)
            self.world.add(loc, Province(side="blue"))
        for loc in red_province_locations:
            loc = tuple(loc)
            self.world.add(loc, Province(side="red"))
        for loc in river_locations:
            loc = tuple(loc)
            self.world.add(loc, River())
        for loc, agent in zip(capital_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)
            if agent.side == "blue":
                agent.provinces = [
                    entity
                    for entity in self.world.get_entities_of_kind("blue_province")
                ]
            else:
                agent.provinces = [
                    entity for entity in self.world.get_entities_of_kind("red_province")
                ]

    def take_turn(self) -> None:
        self.turn += 1
        for _, x in np.ndenumerate(self.world.map):
            x: Entity
            if x.has_transitions and not isinstance(x, Agent):
                x.transition(self.world)
        plans = []
        firsts = []
        rewards = []
        distances = []
        for agent in self.agents:
            first, plan, reward, distance = agent.transition(self.world)
            firsts.append(first)
            plans.append(plan)
            rewards.append(reward)
            distances.append(distance)

        for f, a in zip(firsts, self.agents):
            action_name = a.action_spec.get_readable_action(f[1])
            if action_name == "attack":
                self.current_attacks += 1

        values = []
        indices = []

        for sub in plans:
            seen = set()
            vals = []
            idxs = []
            for i, x in enumerate(sub):
                if x is None or x in seen:
                    continue
                seen.add(x)
                vals.append(x)
                idxs.append(i)
            values.append(vals)
            indices.append(idxs)

        plans = values

        # rewards[0] -= sum(((distances[0][j] - 1) * 4) for j in indices[0])
        # rewards[1] -= sum(((distances[1][j] - 1) * 4) for j in indices[1])

        rewards[0] += (
            sum(1 for item in plans[0] if isinstance(item, Location)) * 5
        ) - (sum(1 for item in plans[1] if isinstance(item, Location)) * 20)
        rewards[1] += (
            sum(1 for item in plans[1] if isinstance(item, Location)) * 5
        ) - (sum(1 for item in plans[0] if isinstance(item, Location)) * 20)

        for agent, p, f, r in zip(self.agents, plans, firsts, rewards):
            agent.act(self.world, p, f, r)

    def reset(self) -> None:
        super().reset()
        self.data = pd.concat(
            [
                self.data,
                pd.DataFrame(
                    {"epoch": [self.current_epoch], "attacks": [self.current_attacks]}
                ),
            ],
            ignore_index=True,
        )
        self.current_epoch += 1
        self.current_attacks = 0

    def run_experiment(
        self,
        animate: bool = True,
        logging: bool = True,
        logger: Logger | None = None,
        output_dir: Path | None = None,
    ) -> None:
        super().run_experiment(animate, logging, logger, output_dir)

        if output_dir is None:
            if hasattr(self.config.experiment, "output_dir"):
                output_dir = Path(self.config.experiment.output_dir)
            else:
                output_dir = Path(__file__).parent / "./data/"
            assert isinstance(output_dir, Path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.data.to_csv(output_dir / "attack_data.csv")
