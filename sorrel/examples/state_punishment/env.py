"""Environment for the state punishment game."""

from pathlib import Path
from typing import List, override, Dict

import numpy as np
import torch
from numpy import ndenumerate

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.environment import Environment
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.logging import Logger
from sorrel.utils.visualization import ImageRenderer

from .agents import StatePunishmentAgent
from .entities import A, B, C, D, E, EmptyEntity, Sand, Wall
from .entity_map_shuffler import EntityMapShuffler
from .world import StatePunishmentWorld


class PunishmentTracker:
    """Minimal punishment tracker that hooks into existing flow."""
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.last_turn_punishments = {i: False for i in range(num_agents)}
        self.current_turn_punishments = {i: False for i in range(num_agents)}
    
    def record_punishment(self, agent_id: int):
        """Record that an agent was punished this turn."""
        self.current_turn_punishments[agent_id] = True
    
    def end_turn(self):
        """Move current turn data to last turn data."""
        self.last_turn_punishments = self.current_turn_punishments.copy()
        self.current_turn_punishments = {i: False for i in range(self.num_agents)}
    
    def get_other_punishments(self, agent_id: int, disable_info: bool = False) -> List[float]:
        """Get punishment status of other agents from last turn.
        
        Args:
            agent_id: ID of the current agent
            disable_info: If True, return zeros instead of actual punishment info
            
        Returns:
            List of punishment status (1.0 if punished, 0.0 if not, or 0.0 if disabled)
        """
        if disable_info:
            return [0.0] * (self.num_agents - 1)
        
        punishments = []
        for i in range(self.num_agents):
            if i != agent_id:
                punishments.append(1.0 if self.last_turn_punishments[i] else 0.0)
        return punishments


class MultiWorldImageRenderer:
    """Custom image renderer for multi-world environments that combines all worlds into
    a 2x3 grid."""

    def __init__(
        self,
        experiment_name: str,
        record_period: int,
        num_turns: int,
        individual_envs: List,
    ):
        """Initialize the multi-world image renderer.

        Args:
            experiment_name: Name of the experiment
            record_period: How often to create an animation
            num_turns: Number of turns per game
            individual_envs: List of individual environments to render
        """
        self.experiment_name = experiment_name
        self.record_period = record_period
        self.num_turns = num_turns
        self.individual_envs = individual_envs
        self.frames = []

    def clear(self):
        """Zero out the frames."""
        del self.frames[:]

    def add_image(self, individual_envs: List, punishment_level: float = None) -> None:
        """Add a combined image of all worlds to the frames.

        Args:
            individual_envs: List of individual environments to render
            punishment_level: Current punishment level to display
        """
        from PIL import Image

        from sorrel.utils.visualization import image_from_array, render_sprite

        # Render each individual world
        world_images = []
        for env in individual_envs:
            full_sprite = render_sprite(env.world)
            world_img = image_from_array(full_sprite)
            world_images.append(world_img)

        # Create a 2x3 grid layout (always 2 rows, 3 columns)
        # Calculate grid dimensions
        num_worlds = len(world_images)
        rows, cols = 2, 3  # Always use 2x3 grid for consistency

        # Get dimensions of individual images
        if world_images:
            img_width, img_height = world_images[0].size
        else:
            return

        # Create combined image
        combined_width = cols * img_width
        combined_height = rows * img_height
        combined_img = Image.new(
            "RGB", (combined_width, combined_height), (255, 255, 255)
        )

        # Place each world image in the grid
        for i, world_img in enumerate(world_images):
            if i >= rows * cols:
                break

            row = i // cols
            col = i % cols

            x = col * img_width
            y = row * img_height

            combined_img.paste(world_img, (x, y))

        # Fill empty slots with blank spaces (for 3 agents: 3 worlds + 3 empty slots)
        # This ensures consistent 2x3 grid layout

        # Add labels for each world
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(combined_img)

        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Add world labels (only for actual worlds, not empty slots)
        for i, world_img in enumerate(world_images):
            if i >= rows * cols:
                break

            row = i // cols
            col = i % cols

            x = col * img_width + 5
            y = row * img_height + 5

            # World label
            draw.text((x, y), f"World {i+1}", fill=(0, 0, 0), font=font)

        # Add global punishment level in bottom right corner
        if punishment_level is not None:
            punishment_text = f"Punishment Level: {punishment_level:.3f}"
            # Calculate position for bottom right corner
            text_x = combined_width - 200  # Leave some margin from right edge
            text_y = combined_height - 30  # Leave some margin from bottom edge
            draw.text((text_x, text_y), punishment_text, fill=(255, 0, 0), font=font)

        self.frames.append(combined_img)

    def save_gif(self, epoch: int, folder: Path) -> None:
        """Save a gif to disk.

        Args:
            epoch: The epoch number
            folder: The destination folder
        """
        from sorrel.utils.visualization import animate_gif

        animate_gif(self.frames, f"{self.experiment_name}_epoch{epoch}", folder)
        # Clear frames
        self.clear()


class MultiAgentStatePunishmentEnv(Environment[StatePunishmentWorld]):
    """Multi-agent environment that coordinates multiple individual environments."""

    def __init__(
        self,
        individual_envs: list["StatePunishmentEnv"],
        shared_state_system,
        shared_social_harm,
    ):
        """Initialize the multi-agent environment.

        Args:
            individual_envs: List of individual StatePunishmentEnv instances
            shared_state_system: Shared state system across all environments
            shared_social_harm: Shared social harm tracking
        """
        # Use the first environment's world and config as the base
        self.individual_envs = individual_envs
        self.shared_state_system = shared_state_system
        self.shared_social_harm = shared_social_harm

        # Set up the world and config without calling super().__init__()
        # to avoid double population issues
        self.world = individual_envs[0].world
        self.config = individual_envs[0].config
        self.turn = 0

        # Set up simplified multi-agent coordination for all individual environments
        for i, env in enumerate(self.individual_envs):
            # Just set the agent ID - no complex coordination needed
            env.agents[0].agent_id = i

        # Initialize punishment tracker if needed
        self.punishment_tracker = None
        if any(env.config.experiment.get("observe_other_punishments", False) for env in individual_envs):
            self.punishment_tracker = PunishmentTracker(len(individual_envs))

    @override
    def take_turn(self) -> None:
        """Coordinate turns across all individual environments."""
        # Increment the turn counter for the multi-agent environment
        self.turn += 1

        # Handle entity transitions in all environments
        for env in self.individual_envs:
            for _, x in ndenumerate(env.world.map):
                if x.has_transitions and not isinstance(x, Agent):
                    x.transition(env.world)

        # Simplified agent transition logic
        self._handle_agent_transitions()

        # Record punishment level for all environments
        for env in self.individual_envs:
            env.world.record_punishment_level()

        # End turn for punishment tracker
        if self.punishment_tracker is not None:
            self.punishment_tracker.end_turn()

    def _handle_agent_transitions(self) -> None:
        """Handle agent transitions with simplified composite view logic."""
        # Collect all agents and their environments
        all_agents = []
        all_envs = []
        for env in self.individual_envs:
            for agent in env.agents:
                all_agents.append(agent)
                all_envs.append(env)

        # Check if any environment uses composite views
        use_composite = any(env.use_composite_views for env in self.individual_envs)

        if use_composite:
            # Generate composite observations for all agents
            composite_observations = self._generate_composite_observations(
                all_agents, all_envs
            )

        # Process each agent
        for i, (agent, env) in enumerate(zip(all_agents, all_envs)):
            if use_composite and env.use_composite_views:
                # Use composite observation and add agent-specific scalars
                state = agent._add_scalars_to_composite_state(
                    composite_observations[i],
                    self.shared_state_system,
                    self.shared_social_harm,
                    punishment_tracker=self.punishment_tracker
                )
            else:
                # Use single agent view
                state = agent.generate_single_view(
                    env.world, self.shared_state_system, self.shared_social_harm,
                    punishment_tracker=self.punishment_tracker
                )

            # Execute agent transition
            self._execute_agent_transition(agent, env, state)

    def _generate_composite_observations(self, all_agents, all_envs):
        """Generate composite observations for all agents."""
        composite_observations = []

        for i, (agent, env) in enumerate(zip(all_agents, all_envs)):
            if not env.use_composite_views:
                # Single view for this agent
                composite_observations.append(
                    agent.generate_single_view(
                        env.world, self.shared_state_system, self.shared_social_harm,
                        punishment_tracker=self.punishment_tracker
                    )
                )
                continue

            # Generate composite view by collecting observations from all agents
            all_views = [
                self._get_agent_observation_without_scalars(a, e.world)
                for a, e in zip(all_agents, all_envs)
            ]
            composite_observations.append(np.concatenate(all_views, axis=1))

        return composite_observations

    def _get_agent_observation_without_scalars(self, agent, world):
        """Get agent observation excluding all scalar values."""
        # Get only the visual field observation (no scalar features)
        image = agent.observation_spec.observe(world, agent.location)
        visual_field = image.reshape(1, -1)

        return visual_field

    def _execute_agent_transition(self, agent, env, state):
        """Execute agent transition with the given state."""
        # Get action from model
        action = agent.get_action(state)

        # Execute action with shared state system and social harm
        if self.punishment_tracker is not None:
            # Use new interface with info return
            reward, info = agent.act(
                env.world, action, self.shared_state_system, self.shared_social_harm, return_info=True
            )
            
            # Track punishment if it occurred
            if info.get('is_punished', False):
                self.punishment_tracker.record_punishment(agent.agent_id)
        else:
            # Use original interface (backward compatible)
            reward = agent.act(
                env.world, action, self.shared_state_system, self.shared_social_harm
            )

        # Update individual score
        agent.individual_score += reward

        # Check if done
        done = agent.is_done(env.world)

        # Add to memory
        env.world.total_reward += reward
        agent.add_memory(state.flatten(), action, reward, done)

    @override
    def reset(self) -> None:
        """Reset all individual environments."""
        self.turn = 0
        self.world.is_done = False
        # Reset shared social harm centrally
        self.shared_social_harm = {i: 0.0 for i in range(len(self.individual_envs))}

        for env in self.individual_envs:
            env.reset()
            for agent in env.agents:
                agent.reset()

    @override
    def run_experiment(
        self,
        animate: bool = True,
        logging: bool = True,
        logger: Logger | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Run the multi-agent experiment with coordination."""
        renderer = None
        if animate:
            renderer = MultiWorldImageRenderer(
                experiment_name=self.world.__class__.__name__,
                record_period=self.config.experiment.record_period,
                num_turns=self.config.experiment.max_turns,
                individual_envs=self.individual_envs,
            )

        for epoch in range(self.config.experiment.epochs + 1):
            # Check if entity appearance shuffling should occur
            shuffle_occurred = False
            if (self.config.experiment.enable_appearance_shuffling and 
                epoch > 0 and 
                epoch % self.config.experiment.shuffle_frequency == 0):
                
                # Shuffle entity appearances in all environments using shared mapping
                if self.individual_envs and self.individual_envs[0].entity_map_shuffler is not None:
                    # Use the first environment's shuffler to generate the mapping
                    shared_mapping = self.individual_envs[0].entity_map_shuffler.shuffle_appearances()
                    
                    # Apply the same mapping to all environments
                    for env in self.individual_envs:
                        if env.entity_map_shuffler is not None:
                            env.entity_map_shuffler.current_mapping = shared_mapping.copy()
                            # Apply shuffled mapping to all agents' observation specs
                            for agent in env.agents:
                                if hasattr(agent, 'observation_spec'):
                                    agent.observation_spec.entity_map = env.entity_map_shuffler.apply_to_entity_map(
                                        agent.observation_spec.entity_map
                                    )
                    
                    shuffle_occurred = True
                    print(f"Epoch {epoch}: Entity appearances shuffled: {shared_mapping}")
            
            # Log appearance mapping for this epoch (even if no shuffle) - only log once
            if self.config.experiment.csv_logging and self.individual_envs:
                # Use the first environment's shuffler to log (they should all be the same)
                self.individual_envs[0].log_entity_appearances(epoch, shuffle_occurred)
            
            # Reset all environments
            self.reset()

            # Start epoch action for all agents
            for env in self.individual_envs:
                for agent in env.agents:
                    agent.model.start_epoch_action(epoch=epoch)

            # Determine whether to animate this epoch
            animate_this_epoch = animate and (
                epoch % self.config.experiment.record_period == 0
            )

            # Run the environment for the specified number of turns
            while not self.turn >= self.config.experiment.max_turns:
                # Render if needed
                if animate_this_epoch and renderer is not None:
                    # Get current punishment level from shared state system
                    current_punishment = self.shared_state_system.prob
                    renderer.add_image(
                        self.individual_envs, punishment_level=current_punishment
                    )

                # Take turn in this environment (which coordinates with others)
                self.take_turn()

                # Check if any environment is done
                if any(env.world.is_done for env in self.individual_envs):
                    break

            # Set all environments as done
            for env in self.individual_envs:
                env.world.is_done = True

            # Generate the gif if animation was done for this epoch
            if animate_this_epoch and renderer is not None:
                if output_dir is None:
                    output_dir = Path("./data/")
                renderer.save_gif(epoch, output_dir)

            # End epoch action for all agents
            for env in self.individual_envs:
                for agent in env.agents:
                    agent.model.end_epoch_action(epoch=epoch)

            # Train all agents at the end of each epoch
            total_loss = 0.0
            loss_count = 0
            for env in self.individual_envs:
                for agent in env.agents:
                    if (
                        hasattr(agent.model, "train_step")
                        and len(agent.model.memory) >= agent.model.batch_size
                    ):
                        loss = agent.model.train_step()
                        if loss is not None:
                            total_loss += float(loss)
                            loss_count += 1

            # Log results
            if logging and logger is not None:

                total_reward = sum(
                    env.world.total_reward for env in self.individual_envs
                )

                avg_loss = total_loss / loss_count if loss_count > 0 else 0.0

                # Get current epsilon from the first agent's model
                current_epsilon = (
                    np.mean(
                        [
                            self.individual_envs[k].agents[0].model.epsilon
                            for k in range(len(self.individual_envs))
                        ]
                    )
                    if self.individual_envs
                    else 0.0
                )

                logger.record_turn(
                    epoch, avg_loss, total_reward, epsilon=current_epsilon
                )

            # Update epsilon for all agents
            for env in self.individual_envs:
                for agent in env.agents:
                    agent.model.epsilon_decay(self.config.model.epsilon_decay)

            # Save models every X epochs
            if epoch > 0 and epoch % self.config.experiment.save_models_every == 0:
                self._save_models(epoch)

            # Print progress
            if epoch % 100 == 0:
                avg_punishment = (
                    self.shared_state_system.get_average_punishment_level()
                    if hasattr(self.shared_state_system, "get_average_punishment_level")
                    else self.shared_state_system.prob
                )
                current_punishment = self.shared_state_system.prob
                print(
                    f"Epoch {epoch}: Current punishment level: {current_punishment:.3f}, Average: {avg_punishment:.3f}"
                )
                print(
                    f"  Total reward: {sum(env.world.total_reward for env in self.individual_envs):.2f}"
                )

        # Save final models at the end of training
        self._save_models(self.config.experiment.epochs)

    def _save_models(self, epoch: int) -> None:
        """Save all agent models to the models directory.
        
        Args:
            epoch: Current epoch number (for logging purposes)
        """
        from pathlib import Path
        
        # Create models directory if it doesn't exist
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Get experiment name from config
        experiment_name = self.config.experiment.get("run_name", "experiment")
        
        # Save each agent's model (overwrite previous versions)
        for env_idx, env in enumerate(self.individual_envs):
            for agent_idx, agent in enumerate(env.agents):
                # Create filename with experiment name, environment, and agent info (no epoch)
                model_filename = f"{experiment_name}_env_{env_idx}_agent_{agent_idx}.pth"
                model_path = models_dir / model_filename
                
                # Save the model (overwrites previous version)
                agent.model.save(model_path)
                
        print(f"Saved models for epoch {epoch} to {models_dir.absolute()}")


class StatePunishmentEnv(Environment[StatePunishmentWorld]):
    """Environment for the state punishment game."""

    def __init__(self, world: StatePunishmentWorld, config: dict) -> None:
        self.use_composite_views = config.get("use_composite_views", False)
        self.use_composite_actions = config.get("use_composite_actions", False)
        self.use_multi_env_composite = config.get("use_multi_env_composite", False)
        self.simple_foraging = config.get("simple_foraging", False)
        self.use_random_policy = config.get("use_random_policy", False)

        # Initialize entity map shuffler for appearance shuffling BEFORE calling super().__init__()
        self.entity_map_shuffler = None
        if config.experiment.get("enable_appearance_shuffling", False):
            resource_entities = ["A", "B", "C", "D", "E"]
            # Create entity_mappings folder and use run_folder as prefix in filename
            # We'll get the run_folder from the main.py when it's passed to the environment
            csv_file_path = Path(__file__).parent / "data" / "entity_mappings" / "entity_appearances.csv"
            self.entity_map_shuffler = EntityMapShuffler(
                resource_entities=resource_entities,
                csv_file_path=csv_file_path,
                enable_logging=config.experiment.get("csv_logging", False),
                shuffle_constraint=config.experiment.get("shuffle_constraint", "no_fixed")
            )

        # Simplified - no complex coordination needed
        super().__init__(world, config)

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents."""
        agent_num = self.config.experiment.num_agents
        agents = []

        for i in range(agent_num):
            # Create the observation spec with separate entity types for each agent
            entity_list = [
                "EmptyEntity",
                "Wall",
                "Sand",
                "A",
                "B",
                "C",
                "D",
                "E",
                "StatePunishmentAgent",
            ]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=self.config.model.full_view,
                vision_radius=self.config.model.agent_vision_radius,
                env_dims=(
                    (self.config.world.height, self.config.world.width)
                    if self.config.model.full_view
                    else None
                ),
            )

            # Apply shuffled entity map if shuffling is enabled
            if self.entity_map_shuffler is not None:
                observation_spec.entity_map = self.entity_map_shuffler.apply_to_entity_map(observation_spec.entity_map)

            # Don't override input size - let the observation spec handle it naturally

            # Create the action spec
            if self.use_composite_actions:
                # Composite actions: 4 movements × 3 voting options + noop = 13 actions
                action_names = [
                    "up_no_vote",
                    "down_no_vote",
                    "left_no_vote",
                    "right_no_vote",
                    "up_increase",
                    "down_increase",
                    "left_increase",
                    "right_increase",
                    "up_decrease",
                    "down_decrease",
                    "left_decrease",
                    "right_decrease",
                    "noop",
                ]
            else:
                # Simple actions: 4 movements + 2 voting + noop = 7 actions
                action_names = [
                    "up",
                    "down",
                    "left",
                    "right",
                    "vote_increase",
                    "vote_decrease",
                    "noop",
                ]

            action_spec = ActionSpec(action_names)

            # Create the model with extra features (punishment_level, social_harm, random_noise)
            # The input_size should be a tuple representing the flattened dimensions
            # We always add 3 extra features: punishment_level (accessible value or 0), social_harm, random_noise
            base_flattened_size = (
                observation_spec.input_size[0]
                * observation_spec.input_size[1]
                * observation_spec.input_size[2]
                + 3
            )
            
            # Add punishment observation features if enabled
            if self.config.experiment.get("observe_other_punishments", False):
                # Add features for other agents' punishment status (total_num_agents - 1)
                total_num_agents = self.config.experiment.get("total_num_agents", self.config.experiment.num_agents)
                num_other_agents = total_num_agents - 1
                base_flattened_size += num_other_agents

            # Adjust for composite views (multiply by number of views)
            if self.use_composite_views:
                # Composite views use all agent perspectives
                flattened_size = base_flattened_size * self.config.experiment.num_agents
            else:
                flattened_size = base_flattened_size
            model = PyTorchIQN(
                input_size=(flattened_size,),
                action_space=action_spec.n_actions,
                layer_size=self.config.model.layer_size,
                epsilon=self.config.model.epsilon,
                epsilon_min=self.config.model.epsilon_min,
                device=self.config.model.device,
                seed=torch.random.seed(),
                n_frames=self.config.model.n_frames,
                n_step=self.config.model.n_step,
                sync_freq=self.config.model.sync_freq,
                model_update_freq=self.config.model.model_update_freq,
                batch_size=self.config.model.batch_size,
                memory_size=self.config.model.memory_size,
                LR=self.config.model.LR,
                TAU=self.config.model.TAU,
                GAMMA=self.config.model.GAMMA,
                n_quantiles=self.config.model.n_quantiles,
            )

            agents.append(
                StatePunishmentAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                    agent_id=i,
                    use_composite_views=self.use_composite_views,
                    use_composite_actions=self.use_composite_actions,
                    simple_foraging=self.simple_foraging,
                    use_random_policy=self.use_random_policy,
                    punishment_level_accessible=self.config.experiment.get("punishment_level_accessible", False),
                    social_harm_accessible=self.config.experiment.get("social_harm_accessible", False),
                    delayed_punishment=self.config.experiment.get("delayed_punishment", False),
                    important_rule=self.config.experiment.get("important_rule", False),
                    punishment_observable=self.config.experiment.get("punishment_observable", False),
                    disable_punishment_info=self.config.experiment.get("disable_punishment_info", False),
                )
            )

        self.agents = agents

    def populate_environment(self):
        """Populate the state punishment world by creating walls, placing initial
        resources, then randomly spawning the agents."""
        valid_spawn_locations = []

        # Create walls around the edges and sand layer
        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                # Add walls around the edge of the world
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put sand there
                self.world.add(index, Sand())
            elif (
                z == 1
            ):  # if location is on the top layer, indicate that it's possible for an agent to spawn there
                # valid spawn location
                valid_spawn_locations.append(index)

        # Spawn agents
        agent_locations_indices = np.random.choice(
            len(valid_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)

        # Remove agent locations from valid spawn locations for resources
        remaining_spawn_locations = [
            loc for loc in valid_spawn_locations if loc not in agent_locations
        ]

        # Place initial resources
        initial_resources = self.config.experiment.get("initial_resources", 10)
        resource_locations_indices = np.random.choice(
            len(remaining_spawn_locations),
            size=min(initial_resources, len(remaining_spawn_locations)),
            replace=False,
        )
        resource_locations = [
            remaining_spawn_locations[i] for i in resource_locations_indices
        ]

        for loc in resource_locations:
            # Use complex entity spawning
            self.world.spawn_entity(loc)

    def get_metrics(self) -> dict:
        """Get current metrics for logging."""
        metrics = {}

        # Individual agent metrics
        for i, agent in enumerate(self.agents):
            metrics[f"Agent_{i}/individual_score"] = agent.individual_score
            metrics[f"Agent_{i}/punishment_level"] = self.world.state_system.prob

            # Encounter metrics
            for resource_type, count in agent.encounters.items():
                metrics[f"Agent_{i}/{resource_type}_encounters"] = count

        # Global metrics
        metrics["Global/punishment_level"] = self.world.state_system.prob
        metrics["Global/total_votes"] = len(self.world.state_system.vote_history)
        metrics["Global/mean_individual_score"] = np.mean(
            [agent.individual_score for agent in self.agents]
        )

        # Vote tracking metrics
        vote_stats = self.world.state_system.get_epoch_vote_stats()
        metrics["Global/epoch_vote_up"] = vote_stats["vote_up"]
        metrics["Global/epoch_vote_down"] = vote_stats["vote_down"]
        metrics["Global/epoch_total_votes"] = vote_stats["total_votes"]

        # Transgression and punishment statistics
        transgression_stats = self.world.state_system.get_transgression_stats()
        metrics.update(transgression_stats)

        return metrics

    def override_agents(self, agents: list[Agent]) -> None:
        """Override the current agent configuration with a list of new agents and resets
        the environment.

        Args:
            agents: A list of new agents
        """
        self.agents = agents

    def shuffle_entity_appearances(self) -> None:
        """Shuffle entity appearances in observation specs."""
        if self.entity_map_shuffler is None:
            return
        
        # Shuffle the appearance mapping
        self.entity_map_shuffler.shuffle_appearances()
        
        # Apply shuffled mapping to all agents' observation specs
        for agent in self.agents:
            if hasattr(agent, 'observation_spec'):
                agent.observation_spec.entity_map = self.entity_map_shuffler.apply_to_entity_map(
                    agent.observation_spec.entity_map
                )
        
        print(f"Entity appearances shuffled: {self.entity_map_shuffler.get_current_mapping()}")

    def log_entity_appearances(self, epoch: int, shuffle_occurred: bool = False) -> None:
        """Log current entity appearance mapping to CSV."""
        if self.entity_map_shuffler is not None:
            self.entity_map_shuffler.log_to_csv(epoch, shuffle_occurred)
