# begin imports
# general imports
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.iowa.agents import GamblingAgent
from sorrel.examples.iowa.entities import EmptyEntity, Sand, Wall
from sorrel.examples.iowa.world import GamblingWorld

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.logging import ConsoleLogger, Logger
from sorrel.utils.visualization import ImageRenderer

# end imports


# begin gambling environment
class GamblingEnv(Environment[GamblingWorld]):
    """Environment inspired by the Iowa Gambling Task (IGT)."""

    def __init__(self, world: GamblingWorld, config: dict) -> None:
        super().__init__(world, config)

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined.
        """
        agent_num = 2
        agents = []
        for i in range(agent_num):
            # create the observation spec
            entity_list = [
                "EmptyEntity",
                "Wall",
                "Sand",
                "DeckA",
                "DeckB",
                "DeckC",
                "DeckD",
                "GamblingAgent",
            ]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=False,
                # note that here we require self.config to have the entry model.agent_vision_radius
                # don't forget to pass it in as part of config when creating this experiment!
                vision_radius=self.config.model.agent_vision_radius,
            )
            observation_spec.override_input_size(
                np.array(observation_spec.input_size).reshape(1, -1).tolist()
            )

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right"])

            # create the model
            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                seed=torch.random.seed(),
                **self.config.model.parameters,
            )

            if hasattr(self.config.model, "load_weights"):
                model.load(
                    file_path=Path(__file__).parent
                    / f"./checkpoints/{self.config.model.load_weights}-agent-{i}.pkl"
                )

            agents.append(
                GamblingAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                )
            )

        self.agents: list[GamblingAgent] = agents  # type: ignore

    def populate_environment(self):
        """Populate the gambling world by creating walls, then randomly spawning the
        agents.

        Note that self.world.map is already created with the specified dimensions, and
        every space is filled with EmptyEntity, as part of super().__init__() when this
        experiment is constructed.
        """
        valid_spawn_locations = []

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put sand there
                self.world.add(index, Sand())
            elif (
                z == 1
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

    def run_experiment(
        self,
        animate: bool = True,
        logging: bool = True,
        logger: Logger | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Run the experiment.

        Required config parameters:
            - experiment.epochs: The number of epochs to run the experiment for.
            - experiment.max_turns: The maximum number of turns each epoch.
            - (Only if `animate` is true) experiment.record_period: The time interval at which to record the experiment.

        If `animate` is true,
        animates the experiment every `self.config.experiment.record_period` epochs.

        If `logging` is true, logs the total loss and total rewards each epoch.

        Args:
            animate: Whether to animate the experiment. Defaults to True.
            logging: Whether to log the experiment. Defaults to True.
            logger: The logger to use. Defaults to a ConsoleLogger.
            output_dir: The directory to save the animations to. Defaults to "./data/" (relative to current working directory).
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "./data/"
        renderer = None
        if animate:
            renderer = ImageRenderer(
                experiment_name=self.__class__.__name__,
                record_period=self.config.experiment.record_period,
                num_turns=self.config.experiment.max_turns,
            )
        for epoch in range(self.config.experiment.epochs + 1):
            # Reset the environment at the start of each epoch
            self.reset()

            # Determine whether to animate this turn.
            animate_this_turn = animate and (
                epoch % self.config.experiment.record_period == 0
            )

            # start epoch action for each agent model
            for agent in self.agents:
                agent.model.start_epoch_action(epoch=epoch)

            # run the environment for the specified number of turns
            while not self.turn >= self.config.experiment.max_turns:
                # renderer should never be None if animate is true; this is just written for pyright to not complain
                if animate_this_turn and renderer is not None:
                    renderer.add_image(self.world)
                self.take_turn()

            self.world.is_done = True

            # generate the gif if animation was done
            if animate_this_turn and renderer is not None:
                renderer.save_gif(epoch, output_dir / "./gifs/")

            # end epoch action for each agent model
            for agent in self.agents:
                agent.model.end_epoch_action(epoch=epoch)

            # # At the end of each epoch, train the agents.
            # with Pool() as pool:
            #     # Use multiprocessing to train agents in parallel
            #     models = [agent.model for agent in self.agents]
            #     total_loss = sum(pool.map(lambda model: model.train_step(), models))
            total_loss = 0
            encounters = {"DeckA": 0, "DeckB": 0, "DeckC": 0, "DeckD": 0}
            for agent in self.agents:
                total_loss = agent.model.train_step()
                for key in encounters.keys():
                    encounters[key] += agent.encounters[key]

            # Log the information
            if logging:
                if not logger:
                    logger = ConsoleLogger(self.config.experiment.epochs)
                logger.record_turn(
                    epoch,
                    total_loss,
                    self.world.total_reward,
                    self.agents[0].model.epsilon,
                    encounters=encounters,
                )

            # update epsilon
            for i, agent in enumerate(self.agents):
                if hasattr(self.config.model, "epsilon_decay"):
                    agent.model.epsilon_decay(self.config.model.epsilon_decay)
                if epoch % self.config.experiment.record_period == 0:
                    if not os.path.exists(output_dir / "./checkpoints/"):
                        os.makedirs(output_dir / "./checkpoints")
                    agent.model.save(
                        output_dir
                        / f"./checkpoints/{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}-agent-{i}.pkl"
                    )
