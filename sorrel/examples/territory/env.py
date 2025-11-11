import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.agents.agent import Agent
from sorrel.entities.entity import Entity
from sorrel.environment import Environment

# sorrel imports
from sorrel.examples.territory.agents import TerritoryAgent, TerritoryObservation
from sorrel.examples.territory.entities import EmptyEntity, Province
from sorrel.examples.territory.world import TerritoryWorld
from sorrel.location import Location
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec


class TerritoryEnvironment(Environment[TerritoryWorld]):
    """The experiment for the territory environment."""

    def __init__(self, world: TerritoryWorld, config: dict) -> None:
        super().__init__(world, config, stop_if_done=True)

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
            ]
            
            observation_spec = TerritoryObservation(
                entity_list=entity_list,
                env_dims=(self.world.height, self.world.width),
                side="red" if is_blue_agent else "blue",
            )

            # create the action spec
            action_spec = ActionSpec(
                [
                    "harvest",
                    "attack"
                ]
            )

            # create the model
            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=0.6,
                device="cpu",
                seed=torch.random.seed(),
                n_frames=5,
                n_step=3,
                sync_freq=200,
                model_update_freq=4,
                batch_size=64,
                #batch_size=32,
                memory_size=1024,
                LR=0.00025,
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

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            
            if np.random.rand() < 0.1 and len(capital_locations) < len(self.agents):
                capital_locations.append(index)
            else:
                if np.random.rand() < 0.5:
                    blue_province_locations.append(index)
                else:
                    red_province_locations.append(index)

        # spawn the agents
        for loc in blue_province_locations:
            loc = tuple(loc)
            self.world.add(loc, Province(side="blue"))
        for loc in red_province_locations:
            loc = tuple(loc)
            self.world.add(loc, Province(side="red"))
        for loc, agent in zip(capital_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)
            if agent.side == "blue":
                agent.provinces = [
                    entity
                    for entity in self.world.get_entities_of_kind('blue_province')
                ]
            else:
                agent.provinces = [
                    entity
                    for entity in self.world.get_entities_of_kind('red_province')
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
        for agent in self.agents:
            first, plan, reward = agent.transition(self.world)
            firsts.append(first)
            plans.append(plan)
            rewards.append(reward)

        plans = [[x for x in set(sub) if x is not None] for sub in plans]

        rewards[0] += ((sum(1 for item in plans[0] if isinstance(item, Location)) * 5) - (sum(1 for item in plans[1] if isinstance(item, Location)) * 10))
        rewards[1] += ((sum(1 for item in plans[1] if isinstance(item, Location)) * 5) - (sum(1 for item in plans[0] if isinstance(item, Location)) * 10))

        for agent, p, f, r in zip(self.agents, plans, firsts, rewards):
            agent.act(self.world, p, f, r)
