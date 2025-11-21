import os
from abc import abstractmethod
from datetime import datetime
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
        stop_if_done: Whether to end the epoch if the world is done. Defaults to False.

            .. note::
                Some default methods provided by this class, such as `run_experiment`, require certain config parameters to be defined.
                These parameters are listed in the docstring of the method.
    """

    world: W
    config: DictConfig
    agents: list[Agent]
    stop_if_done: bool

    simultaneous_moves: bool

    def __init__(
        self,
        world: W,
        config: DictConfig | dict | list,
        stop_if_done: bool = False,
        simultaneous_moves: bool = False,
    ) -> None:

        if isinstance(config, DictConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = OmegaConf.create(config)
        else:
            # note that if config is a list, we assume it is a dotlist
            self.config = OmegaConf.from_dotlist(config)

        self.world = world
        self.turn = 0
        self.world.create_world()
        self.stop_if_done = stop_if_done
        self.simultaneous_moves = simultaneous_moves

        self.setup_agents()
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

        if not self.simultaneous_moves:
            # Original sequential logic
            for agent in self.agents:
                agent.transition(self.world)
        else:
            # Simultaneous logic
            proposals = []
            destinations = {}  # location -> list of agent indices

            # 1. Get all proposals
            for i, agent in enumerate(self.agents):
                proposal = agent.get_proposed_action(self.world)
                proposals.append(proposal)

                new_loc = proposal["new_location"]
                if new_loc is not None:
                    if new_loc not in destinations:
                        destinations[new_loc] = []
                    destinations[new_loc].append(i)

            # 2. Resolve conflicts and finalize
            for i, agent in enumerate(self.agents):
                proposal = proposals[i]
                new_loc = proposal["new_location"]

                allowed = True
                if new_loc is not None:
                    # Conflict if more than one agent wants to go to this location
                    if len(destinations[new_loc]) > 1:
                        allowed = False

                agent.finalize_turn(self.world, proposal, allowed=allowed)

    # TODO: ability to save/load?
    def run_experiment(
        self,
        animate: bool = True,
        logging: bool = True,
        logger: Logger | None = None,
        output_dir: Path | None = None,
        async_training: bool = False,
        train_interval: float = 0.0,
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
            logger: Optional logger instance. If None, ConsoleLogger will be used.
            output_dir: Optional output directory for checkpoints and gifs.
            async_training: Whether to use asynchronous background training. Defaults to False.
            train_interval: Minimum seconds between async training steps. Defaults to 0.0.
        """
        renderer = None
        if output_dir is None:
            if hasattr(self.config.experiment, "output_dir"):
                output_dir = Path(self.config.experiment.output_dir)
            else:
                output_dir = Path(__file__).parent / "./data/"
            assert isinstance(output_dir, Path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if animate:
            renderer = ImageRenderer(
                experiment_name=self.__class__.__name__,
                record_period=self.config.experiment.record_period,
                num_turns=self.config.experiment.max_turns,
            )

        # Set up async trainers if requested
        async_trainers = []
        if async_training:
            from sorrel.training import AsyncTrainer

            for agent in self.agents:
                trainer = AsyncTrainer(agent.model, train_interval=train_interval)
                trainer.start()
                async_trainers.append(trainer)

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

                if self.world.is_done and self.stop_if_done:
                    break

            self.world.is_done = True

            # generate the gif if animation was done
            if animate_this_turn and renderer is not None:
                renderer.save_gif(epoch, output_dir / "./gifs/")

            # end epoch action for each agent model
            for agent in self.agents:
                agent.model.end_epoch_action(epoch=epoch)

            # Train the agents (sync or async)
            total_loss = 0
            if not async_training:
                # Synchronous training: one train_step per epoch
                for agent in self.agents:
                    total_loss = agent.model.train_step()
            else:
                # Async training: get stats from background trainers
                for trainer in async_trainers:
                    stats = trainer.get_stats()
                    total_loss += stats["avg_loss"]
                # Average across trainers
                if async_trainers:
                    total_loss /= len(async_trainers)

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
            for i, agent in enumerate(self.agents):
                if hasattr(self.config.model, "epsilon_decay"):
                    agent.model.epsilon_decay(self.config.model.epsilon_decay)
                if epoch % self.config.experiment.record_period == 0:
                    agent.model.save(
                        output_dir
                        / f"./checkpoints/{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}-agent-{i}.pkl"
                    )

        # Stop async trainers
        if async_training:
            for trainer in async_trainers:
                trainer.stop()
