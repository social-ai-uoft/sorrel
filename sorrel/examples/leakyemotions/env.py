# begin imports
# general imports
import numpy as np
import omegaconf
import os
import torch
from pathlib import Path

# sorrel imports
from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.environment import Environment
from sorrel.models.pytorch.ppo import PyTorchPPO
from sorrel.models.pytorch.ppo import PyTorchPPO
from sorrel.utils.logging import Logger, ConsoleLogger, TensorboardLogger
from sorrel.utils.visualization import ImageRenderer

# imports from our example
from sorrel.examples.leakyemotions.agents import LeakyEmotionsAgent, Wolf
from sorrel.examples.leakyemotions.custom_observation_spec import LeakyEmotionsObservationSpec
from sorrel.examples.leakyemotions.entities import EmptyEntity, Bush, Wall, Grass
from sorrel.examples.leakyemotions.wolf_model import WolfModel
from sorrel.examples.leakyemotions.world import LeakyEmotionsWorld



# end imports
ENTITY_LIST = ["EmptyEntity", "Bush", "Wall", "Grass", "LeakyEmotionsAgent", "Wolf"]

# begin leakyemotions environment
class LeakyEmotionsEnv(Environment[LeakyEmotionsWorld]):
    """The experiment for Leaky Emotions."""

    def __init__(self, world: LeakyEmotionsWorld, config: dict | omegaconf.DictConfig) -> None:
        super().__init__(world, config)

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined.
        """
        agent_num = self.config.world.agents
        agents = []
        for _ in range(agent_num):
            # create the observation spec
            entity_list = ENTITY_LIST
            observation_spec = LeakyEmotionsObservationSpec(
                entity_list,
                full_view=False,
                # note that here we require self.config to have the entry model.agent_vision_radius
                # don't forget to pass it in as part of config when creating this experiment!
                vision_radius=self.config.model.agent_vision_radius,
            )
            
            observation_spec.override_input_size(
                np.zeros(observation_spec.input_size, dtype=int).reshape(1, -1).shape
            )
            

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right"])

            # create the model
            model = PyTorchPPO(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=0.7,
                device="cpu",
                seed=torch.random.seed(),
                entropy_coef=0.01,
                eps_clip=0.2,
                gamma=0.9,
                k_epochs=10,
                lr_actor=.001,
                lr_critic=.0005,
                max_turns=100
            )

            agents.append(
                LeakyEmotionsAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                )
            )
            agents.append(
                Wolf(
                    observation_spec=observation_spec, 
                    action_spec=action_spec, 
                    model=WolfModel(1, 4, 1)
            )
        )

        self.agents = agents

    def override_agents(self, agents: list[Agent]) -> None:
        """Override the current agent configuration with a list of new agents and resets
        the environment.

        Args:
            agents: A list of new agents
        """
        self.agents = agents

    def populate_environment(self):
        """
        Populate the leakyemotions world by creating walls, then randomly spawning the agents.
        Note that every space is already filled with EmptyEntity as part of super().__init__().
        """
        valid_agent_spawn_locations = []

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom (first) layer, put grass there
                self.world.add(index, Grass())
            elif z == 1: # if location is on third layer, rabbit agents and wolves can appear here 
                valid_agent_spawn_locations.append(index)

        # spawn the agents (rabbits)
        # using np.random.choice, we choose indices in valid_agent_spawn_locations
        agent_locations_indices = np.random.choice(
            len(valid_agent_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_agent_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)

    def reset(self) -> None:
        """Reset the experiment, including the environment and the agents."""
        self.turn = 0
        self.world.is_done = False
        self.world.dead_agents = []
        self.world.create_world()
        self.populate_environment()
        for agent in self.agents:
            agent.reset()

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

            bunnies_left = sum([isinstance(agent, LeakyEmotionsAgent) for agent in self.agents]) - len(self.world.dead_agents)

            # run the environment for the specified number of turns
            while not (self.turn >= self.config.experiment.max_turns) and (bunnies_left > 0):
                # renderer should never be None if animate is true; this is just written for pyright to not complain
                if animate_this_turn and renderer is not None:
                    renderer.add_image(self.world)
                self.take_turn()
                bunnies_left = sum([isinstance(agent, LeakyEmotionsAgent) for agent in self.agents]) - len(self.world.dead_agents)

            self.world.is_done = True

            # generate the gif if animation was done
            if animate_this_turn and renderer is not None:
                if output_dir is None:
                    output_dir = Path(os.getcwd()) / "./data/"
                renderer.save_gif(epoch, output_dir)

            # At the end of each epoch, train the agents.
            total_loss = 0
            for agent in self.agents:
                loss = agent.model.train_step()
                total_loss += loss

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

