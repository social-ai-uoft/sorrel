import os
from abc import abstractmethod
from multiprocessing import Pool
from pathlib import Path

from numpy import ndenumerate
from omegaconf import DictConfig, OmegaConf

from sorrel.agents import Agent
from sorrel.entities import Entity
from sorrel.utils.logging import ConsoleLogger, Logger
from sorrel.utils.visualization import ImageRenderer
from sorrel.worlds import Gridworld


class Environment[W: Gridworld]:
    """An abstract wrapper class for running experiments with agents and environments.

    Attributes:
        world: The world that the experiment includes.
        config: The configurations for the experiment.

            .. note::
                Some default methods provided by this class, such as `run_experiment`, require certain config parameters to be defined.
                These parameters are listed in the docstring of the method.
    """

    world: W
    config: DictConfig
    agents: list[Agent]

    def __init__(self, world: W, config: DictConfig | dict | list) -> None:

        if isinstance(config, DictConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = OmegaConf.create(config)
        else:
            # note that if config is a list, we assume it is a dotlist
            self.config = OmegaConf.from_dotlist(config)

        self.setup_agents()

        self.world = world
        self.turn = 0
        self.world.create_world()
        self.populate_environment()

    @abstractmethod
    def setup_agents(self) -> None:
        """This method should create a list of agents, and assign it to self.agents."""
        pass

    @abstractmethod
    def populate_environment(self) -> None:
        """This method should populate self.world.map.

        Note that self.world.map is already created with the specified dimensions, and
        every space is filled with the default entity of the environment, as part of
        self.world.create_world() when this experiment is constructed. One simply needs
        to place the agents and any additional entitites in self.world.map.
        """
        pass

    def reset(self) -> None:
        """Reset the experiment, including the environment and the agents."""
        self.turn = 0
        self.world.is_done = False
        self.world.create_world()
        self.populate_environment()
        for agent in self.agents:
            agent.reset()

    def take_turn(self) -> None:
        """Performs a full step in the environment.

        This function iterates through the environment and performs transition() for
        each entity, then transitions each agent.
        """
        self.turn += 1
        for _, x in ndenumerate(self.world.map):
            x: Entity
            if x.has_transitions and not isinstance(x, Agent):
                x.transition(self.world)
        for agent in self.agents:
            agent.transition(self.world)

    # TODO: ability to save/load?
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
        renderer = None
        if animate:
            renderer = ImageRenderer(
                experiment_name=self.world.__class__.__name__,
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
                if output_dir is None:
                    output_dir = Path(os.getcwd()) / "./data/"
                renderer.save_gif(epoch, output_dir)

            # At the end of each epoch, train the agents.
            with Pool() as pool:
                # Use multiprocessing to train agents in parallel
                models = [agent.model for agent in self.agents]
                total_loss = sum(pool.map(lambda model: model.train_step(), models))

            # Log the information
            if logging:
                if not logger:
                    logger = ConsoleLogger(self.config.experiment.epochs)
                logger.record_turn(
                    epoch,
                    total_loss,
                    self.world.total_reward,
                    self.agents[0].model.epsilon,
                )

            # update epsilon
            for agent in self.agents:
                agent.model.epsilon_decay(self.config.model.epsilon_decay)
