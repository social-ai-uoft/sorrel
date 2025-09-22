import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.taxi.agents import TaxiAgent
from sorrel.examples.taxi.entities import Destination, EmptyEntity, Passenger, PassengerPoint, Road, Wall
from sorrel.examples.taxi.world import TaxiWorld

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec

class TaxiEnv(Environment[TaxiWorld]):
    """A simple taxi environment."""

    def __init__(self, world: TaxiWorld, config: dict) -> None:
        self.world = world
        super().__init__(world, config)
    
    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents."""

        agent_num = 1
        agents = []
        for _ in range(agent_num):
            # create the observation spec
            entity_list = ["EmptyEntity", "Wall", "TaxiAgent", "Road", "PassengerPoint", "Passenger", "Destination"]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=True,
                env_dims=(self.world.height, self.world.width),
            )
            observation_spec.override_input_size(
                np.array(observation_spec.input_size).reshape(1, -1).tolist()
            )

            #print(observation_spec.input_size)
            #raise Exception("Debugging")

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right", "pickup", "dropoff"])

            # create the model
            model = PyTorchIQN(
                #input_size=observation_spec.input_size,
                input_size=(500,),
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=0.8,
                device="cpu",
                seed=torch.random.seed(),
                n_frames=1,
                n_step=3,
                sync_freq=200,
                model_update_freq=4,
                batch_size=64,
                memory_size=1024,
                #LR=0.00025,
                LR=0.001,
                TAU=0.001,
                GAMMA=0.99,
                n_quantiles=12,
                #n_quantiles=64,
            )

            agents.append(
                TaxiAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                    world=self.world
                )
            )

        self.agents = agents

    def populate_environment(self):
        valid_spawn_locations = []
        passenger_points = [[1, 1, 1], [1, 4, 1], [5, 2, 1], [5, 5, 1]]  # fixed passenger points defined here [y, x, z]

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put road there
                if y == passenger_points[0][0] and x == passenger_points[0][1]:
                    self.world.add(index, PassengerPoint(point_id=1))
                elif y== passenger_points[1][0] and x == passenger_points[1][1]:
                    self.world.add(index, PassengerPoint(point_id=2))
                elif y == passenger_points[2][0] and x == passenger_points[2][1]:
                    self.world.add(index, PassengerPoint(point_id=3))
                elif y == passenger_points[3][0] and x == passenger_points[3][1]:
                    self.world.add(index, PassengerPoint(point_id=4))
                else:
                    self.world.add(index, Road())
            elif (
                z == 2
            ):  # if location is on the top layer, indicate that it's possible for an agent to spawn there
                # valid spawn location
                valid_spawn_locations.append(index)

        # spawn the agents
        # using np.random.choice, we choose indices in valid_spawn_locations
        agent_locations_indices = np.random.choice(
            len(valid_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)

        passenger_destination_location_indices = np.random.choice(
            len(passenger_points), size=2, replace=False
        )
        passenger_destination_locations = [passenger_points[i] for i in passenger_destination_location_indices]

        self.world.add(tuple(passenger_destination_locations[0]), Passenger())
        self.world.passenger_loc = passenger_destination_location_indices[0]
        self.world.add(tuple(passenger_destination_locations[1]), Destination())
        self.world.destination_loc = passenger_destination_location_indices[1]
        #print("Passenger location index:", self.world.passenger_loc, "destination location index: ", self.world.destination_loc)