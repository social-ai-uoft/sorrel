import os
from abc import abstractmethod
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy import ndenumerate
from omegaconf import DictConfig, OmegaConf

from sorrel.agents import Agent
from sorrel.buffers import SavedGames
from sorrel.entities import Entity
from sorrel.utils.logging import ConsoleLogger, Logger
from sorrel.utils.turn_stats import AgentTurnStats, TurnStats
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

    def __init__(
        self, world: W, config: DictConfig | dict | list, stop_if_done: bool = False
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
        self._epoch_turn_stats: list[TurnStats] = []
        self._active_logger: Logger | None = None

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
        self._epoch_turn_stats = []

    def take_turn(self, epoch: int = 0) -> None:
        """Performs a full step in the environment.

        This function iterates through the environment and performs transition() for
        each entity, then transitions each agent. After all transitions, calls
        :meth:`_collect_turn_stats` and :meth:`_on_turn_end` to collect and dispatch
        per-turn statistics.

        Args:
            epoch: Current epoch index, passed to :meth:`_collect_turn_stats` for
                attribution. Defaults to 0 for backward-compatible callers such as
                :meth:`generate_memories`.
        """
        self.turn += 1
        for _, x in ndenumerate(self.world.map):
            x: Entity
            if x.has_transitions and not isinstance(x, Agent):
                x.transition(self.world)
        for agent in self.agents:
            agent.transition(self.world)
        stats = self._collect_turn_stats(epoch)
        self._on_turn_end(stats)

    def _model_start_epoch_action(self, agent: Agent, epoch: int) -> None:
        """Run per-epoch model start hook."""
        agent.model.start_epoch_action(epoch=epoch)

    def _model_end_epoch_action(self, agent: Agent, epoch: int) -> None:
        """Run per-epoch model end hook."""
        agent.model.end_epoch_action(epoch=epoch)

    def _model_train_step(self, agent: Agent):
        """Run model train step."""
        return agent.model.train_step()

    def _collect_turn_stats(self, epoch: int) -> TurnStats:
        """Collect per-turn statistics after all transitions have run.

        Called automatically by :meth:`take_turn`. Override in subclasses to
        populate :attr:`~sorrel.utils.turn_stats.TurnStats.extra` with
        domain-specific metrics. Always call ``super()`` first to obtain the
        base snapshot, then augment the returned object.

        Args:
            epoch: Current epoch index.

        Returns:
            A :class:`~sorrel.utils.turn_stats.TurnStats` snapshot for this turn.
        """
        agent_stats: list[AgentTurnStats] = []
        for i, agent in enumerate(self.agents):
            mem = agent.model.memory
            if mem.size == 0:
                continue
            tail = (mem.idx - 1) % mem.capacity
            agent_stats.append(
                AgentTurnStats(
                    agent_id=i,
                    location=tuple(agent.location),
                    last_action=int(mem.actions[tail]),
                    last_reward=float(mem.rewards[tail]),
                    last_done=bool(mem.dones[tail]),
                )
            )
        return TurnStats(
            epoch=epoch,
            turn=self.turn,
            total_world_reward=float(self.world.total_reward),
            agent_stats=agent_stats,
        )

    def _on_turn_end(self, stats: TurnStats) -> None:
        """Called after every turn with the collected :class:`~sorrel.utils.turn_stats.TurnStats`.

        Default implementation buffers ``stats`` into
        :attr:`_epoch_turn_stats` and delegates to the active logger's
        :meth:`~sorrel.utils.logging.Logger.record_turn` if one is set.
        Override to add side-effects; call ``super()`` to preserve buffering
        and logger dispatch.

        Args:
            stats: The :class:`~sorrel.utils.turn_stats.TurnStats` snapshot
                for the completed turn.
        """
        self._epoch_turn_stats.append(stats)
        if self._active_logger is not None:
            self._active_logger.record_turn(stats)

    def _aggregate_epoch_stats(self) -> dict[str, float]:
        """Aggregate buffered per-turn stats into epoch-level scalars.

        Called once per epoch in :meth:`run_experiment`, before
        :meth:`~sorrel.utils.logging.Logger.record_epoch`. The returned dict
        is forwarded as ``**kwargs`` to ``record_epoch``. Override to produce
        different aggregations; call ``super()`` and update the result dict.

        Returns:
            Dict of scalar values keyed by metric name. Default implementation
            returns mean and max per-agent reward across the epoch's turns.
            Returns ``{}`` if no turns were recorded.
        """
        if not self._epoch_turn_stats:
            return {}
        all_rewards = np.array(
            [
                a.last_reward
                for ts in self._epoch_turn_stats
                for a in ts.agent_stats
            ],
            dtype=np.float32,
        )
        if all_rewards.size == 0:
            return {}
        return {
            "turn_reward_mean": float(all_rewards.mean()),
            "turn_reward_max": float(all_rewards.max()),
        }

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
        if logging:
            if not logger:
                logger = ConsoleLogger(self.config.experiment.epochs)
            self._active_logger = logger
        for epoch in range(self.config.experiment.epochs + 1):
            # Reset the environment at the start of each epoch
            self.reset()

            # Determine whether to animate this turn.
            animate_this_turn = animate and (
                epoch % self.config.experiment.record_period == 0
            )

            # start epoch action for each agent model
            for agent in self.agents:
                self._model_start_epoch_action(agent, epoch)

            # run the environment for the specified number of turns
            while not self.turn >= self.config.experiment.max_turns:
                # renderer should never be None if animate is true; this is just written for pyright to not complain
                if animate_this_turn and renderer is not None:
                    renderer.add_image(self.world)
                self.take_turn(epoch)

                if self.world.is_done and self.stop_if_done:
                    break

            self.world.is_done = True

            # generate the gif if animation was done
            if animate_this_turn and renderer is not None:
                renderer.save_gif(epoch, output_dir / "./gifs/")

            # end epoch action for each agent model
            for agent in self.agents:
                self._model_end_epoch_action(agent, epoch)

            # # At the end of each epoch, train the agents.
            # with Pool() as pool:
            #     # Use multiprocessing to train agents in parallel
            #     models = [agent.model for agent in self.agents]
            #     total_loss = sum(pool.map(lambda model: model.train_step(), models))
            total_loss = 0
            for agent in self.agents:
                total_loss = self._model_train_step(agent)

            # Log the information
            if logging and self._active_logger is not None:
                epoch_kwargs = self._aggregate_epoch_stats()
                self._active_logger.record_epoch(
                    epoch,
                    total_loss,
                    self.world.total_reward,
                    self.agents[0].model.epsilon,
                    **epoch_kwargs,
                )

            # update epsilon
            for i, agent in enumerate(self.agents):
                if hasattr(self.config.model, "epsilon_decay"):
                    agent.model.epsilon_decay(self.config.model.epsilon_decay)
                if epoch % self.config.experiment.record_period == 0:
                    if hasattr(self.config.model, "save_weights"):
                        if self.config.model.save_weights:
                            agent.model.save(
                                output_dir
                                / f"./checkpoints/{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}-agent-{i}.pkl"
                            )

        self._active_logger = None

    def generate_memories(
        self,
        num_games: int = 1000,
        animate: bool = False,
        output_dir: Path | None = None,
    ) -> None:
        """Using the existing models, generate a memory buffer for the specified number
        of games."""
        if output_dir is None:
            if hasattr(self.config.experiment, "output_dir"):
                output_dir = Path(self.config.experiment.output_dir)
            else:
                output_dir = Path(__file__).parent / "./data/"
            assert isinstance(output_dir, Path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # self.setup_agents()

        saved_games: list[SavedGames] = []

        for agent in self.agents:
            if hasattr(agent.model, "n_frames"):
                n_frames = agent.model.n_frames  # type: ignore
            else:
                n_frames = 1
            agent_saved_games = SavedGames(
                capacity=num_games * self.config.experiment.max_turns,
                obs_shape=agent.observation_spec.input_size,
                n_frames=n_frames,
            )
            if hasattr(agent.model, "eval"):
                agent.model.eval()  # type: ignore
            saved_games.append(agent_saved_games)

        # Setup renderer
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

        for game in range(num_games):
            self.reset()
            # Determine whether to animate this turn.
            animate_this_turn = animate and (
                game % self.config.experiment.record_period == 0
            )

            # start epoch action for each agent model
            for agent in self.agents:
                self._model_start_epoch_action(agent, game)

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
                renderer.save_gif(game, output_dir / "./gifs/")

            # end epoch action for each agent model
            for agent, agent_saved_games in zip(self.agents, saved_games):
                self._model_end_epoch_action(agent, game)
                agent_saved_games.add_from_buffer(agent.model.memory)

        for i, agent_saved_games in enumerate(saved_games):
            os.makedirs(output_dir / f"./memories/", exist_ok=True)
            agent_saved_games.save(output_dir / f"./memories/agent{i}.npz")
