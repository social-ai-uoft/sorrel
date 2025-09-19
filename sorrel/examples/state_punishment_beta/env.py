"""Environment for the state punishment game."""

import numpy as np
import torch
from typing import List, override
from pathlib import Path
from numpy import ndenumerate

from sorrel.environment import Environment
from sorrel.action.action_spec import ActionSpec
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.models.pytorch import PyTorchIQN
from sorrel.agents import Agent
from sorrel.utils.visualization import ImageRenderer
from sorrel.utils.logging import Logger

from .world import StatePunishmentWorld
from .agents import StatePunishmentAgent
from .entities import EmptyEntity, Wall, A, B, C, D, E


class MultiAgentStatePunishmentEnv(Environment[StatePunishmentWorld]):
    """Multi-agent environment that coordinates multiple individual environments."""
    
    def __init__(self, individual_envs: list["StatePunishmentEnv"], shared_state_system, shared_social_harm):
        """
        Initialize the multi-agent environment.
        
        Args:
            individual_envs: List of individual StatePunishmentEnv instances
            shared_state_system: Shared state system across all environments
            shared_social_harm: Shared social harm tracking
        """
        # Use the first environment's world and config as the base
        self.individual_envs = individual_envs
        self.shared_state_system = shared_state_system
        self.shared_social_harm = shared_social_harm
        
        # Initialize with the first environment's world and config
        super().__init__(individual_envs[0].world, individual_envs[0].config)
        
        # Set up multi-agent coordination for all individual environments
        for i, env in enumerate(self.individual_envs):
            other_envs = [self.individual_envs[j] for j in range(len(self.individual_envs)) if j != i]
            # env.set_multi_agent_coordination(other_envs, shared_state_system, i)
            env.agents[0].set_multi_agent_coordination(other_envs, shared_state_system, i)
    
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
        
        # Handle agent transitions with multi-agent coordination
        for env in self.individual_envs:
            for agent in env.agents:
                agent.transition(env.world, 
                               state_system=self.shared_state_system,
                               other_environments=[e for e in self.individual_envs if e != env],
                               use_composite_views=env.use_composite_views)
        
        # Record punishment level for all environments
        for env in self.individual_envs:
            env.world.record_punishment_level()
    
    @override
    def reset(self) -> None:
        """Reset all individual environments."""
        self.turn = 0
        self.world.is_done = False
        for env in self.individual_envs:
            env.reset()
            for agent in env.agents:
                agent.reset()
            env.world.social_harm = {i: 0.0 for i in range(len(self.individual_envs))}
        
        # # Ensure all environments have resources after reset
        # for i, env in enumerate(self.individual_envs):
        #     # Count existing resources
        #     resource_count = 0
        #     for row in env.world.map:
        #         for cell in row:
        #             if hasattr(cell, 'kind') and cell.kind in ['A', 'B', 'C', 'D', 'E']:
        #                 resource_count += 1
            
        #     # If no resources, spawn some
        #     if resource_count == 0:
        #         valid_locations = []
        #         for index in np.ndindex(env.world.map.shape):
        #             y, x, z = index
        #             if (y not in [0, env.world.height - 1] and 
        #                 x not in [0, env.world.width - 1] and
        #                 not isinstance(env.world.map[index], Agent)):
        #                 valid_locations.append(index)
                
        #         # Spawn initial resources
        #         initial_resources = min(10, len(valid_locations))
        #         if initial_resources > 0:
        #             resource_locations = np.random.choice(
        #                 len(valid_locations), 
        #                 size=initial_resources, 
        #                 replace=False
        #             )
        #             for idx in resource_locations:
        #                 env.world.spawn_entity(valid_locations[idx])
    
    @override
    def run_experiment(self, animate: bool = True, logging: bool = True, logger: Logger | None = None, output_dir: Path | None = None) -> None:
        """Run the multi-agent experiment with coordination."""
        renderer = None
        if animate:
            renderer = ImageRenderer(
                experiment_name=self.world.__class__.__name__,
                record_period=self.config.experiment.record_period,
                num_turns=self.config.experiment.max_turns,
            )
        
        for epoch in range(self.config.experiment.epochs + 1):
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
                    renderer.add_image(self.world)
                
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
                    if hasattr(agent.model, 'train_step') and len(agent.model.memory) >= agent.model.batch_size:
                        loss = agent.model.train_step()
                        if loss is not None:
                            total_loss += float(loss)
                            loss_count += 1
            
            # Log results
            if logging and logger is not None:

                total_reward = sum(env.world.total_reward for env in self.individual_envs)

                avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
                logger.record_turn(epoch, avg_loss, total_reward, epsilon=self.config.model.epsilon)
            
            # Update epsilon for all agents
            for env in self.individual_envs:
                for agent in env.agents:
                    agent.model.epsilon_decay(self.config.model.epsilon_decay)
            
            # Print progress
            if epoch % 100 == 0:
                avg_punishment = self.shared_state_system.get_average_punishment_level() if hasattr(self.shared_state_system, 'get_average_punishment_level') else self.shared_state_system.prob
                current_punishment = self.shared_state_system.prob
                print(f"Epoch {epoch}: Current punishment level: {current_punishment:.3f}, Average: {avg_punishment:.3f}")
                print(f"  Total reward: {sum(env.world.total_reward for env in self.individual_envs):.2f}")


class StatePunishmentEnv(Environment[StatePunishmentWorld]):
    """Environment for the state punishment game."""

    def __init__(self, world: StatePunishmentWorld, config: dict) -> None:
        self.use_composite_views = config.get("use_composite_views", False)
        self.use_composite_actions = config.get("use_composite_actions", False)
        self.use_multi_env_composite = config.get("use_multi_env_composite", False)
        
        # Multi-agent coordination
        self.other_environments = []  # Will be set by main.py
        self.shared_state_system = None  # Will be set by main.py
        self.agent_id = 0  # This environment's agent ID
        
        super().__init__(world, config)

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents."""
        agent_num = self.config.experiment.num_agents
        agents = []
        
        for i in range(agent_num):
            # Create the observation spec with separate entity types for each agent
            entity_list = ["EmptyEntity", "Wall", "A", "B", "C", "D", "E"] + [f"Agent{i}" for i in range(agent_num)]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=self.config.model.full_view,
                vision_radius=self.config.model.agent_vision_radius,
                env_dims=(self.config.world.height, self.config.world.width) if self.config.model.full_view else None,
            )
            
            # Give each agent different entity representations
            self._create_agent_specific_representations(observation_spec, i, agent_num)
            
            # Don't override input size - let the observation spec handle it naturally

            # Create the action spec
            if self.use_composite_actions:
                # Composite actions: 4 movements × 3 voting options + noop = 13 actions
                action_names = [
                    "up_no_vote", "down_no_vote", "left_no_vote", "right_no_vote",
                    "up_increase", "down_increase", "left_increase", "right_increase", 
                    "up_decrease", "down_decrease", "left_decrease", "right_decrease",
                    "noop"
                ]
            else:
                # Simple actions: 4 movements + 2 voting + noop = 7 actions
                action_names = ["up", "down", "left", "right", "vote_increase", "vote_decrease", "noop"]
                
            action_spec = ActionSpec(action_names)

            # Create the model with extra features (3 additional: punishment_level, social_harm, random_noise)
            # The input_size should be a tuple representing the flattened dimensions
            # We need to add 3 to the total flattened size
            base_flattened_size = observation_spec.input_size[0] * observation_spec.input_size[1] * observation_spec.input_size[2] + 3
            
            # Adjust for composite views (multiply by number of views)
            if self.use_composite_views:
                # Composite views use 6 different agent perspectives (state_stack_size)
                flattened_size = base_flattened_size * 6
            else:
                flattened_size = base_flattened_size
            model = PyTorchIQN(
                input_size=(flattened_size,),
                action_space=action_spec.n_actions,
                layer_size=self.config.model.layer_size,
                epsilon=self.config.model.epsilon,
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
                    use_multi_env_composite=self.use_multi_env_composite,
                )
            )

        # Set unique entity types for each agent
        for i, agent in enumerate(agents):
            agent.kind = f"Agent{i}"
        
        self.agents = agents

    def _create_agent_specific_representations(self, observation_spec, agent_id, total_agents):
        """Create different visual representations for each agent."""
        import numpy as np
        
        # Get the base entity map
        base_entity_map = observation_spec.entity_map.copy()
        
        # Create agent-specific entity list with separate agent types
        entity_list = ["EmptyEntity", "Wall", "A", "B", "C", "D", "E"] + [f"Agent{i}" for i in range(total_agents)]
        num_classes = len(entity_list)
        
        # Create agent-specific one-hot encodings
        agent_entity_map = {}
        
        for i, entity_name in enumerate(entity_list):
            if entity_name == "EmptyEntity":
                # EmptyEntity always gets all zeros
                agent_entity_map[entity_name] = np.zeros(num_classes)
            elif entity_name.startswith("Agent"):
                # Each agent type gets its own unique one-hot representation
                agent_num = int(entity_name.replace("Agent", ""))
                agent_entity_map[entity_name] = np.zeros(num_classes)
                agent_entity_map[entity_name][7 + agent_num] = 1.0  # Position 7+ for agents
            else:
                # Other entities keep their standard representations
                agent_entity_map[entity_name] = base_entity_map[entity_name].copy()
        
        # Apply the agent-specific entity map
        observation_spec.override_entity_map(agent_entity_map)

    def populate_environment(self):
        """Populate the state punishment world by creating walls, placing initial resources,
        then randomly spawning the agents."""
        valid_spawn_locations = []

        # Create walls around the edges
        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                self.world.add(index, Wall())
            else:
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
        remaining_spawn_locations = [loc for loc in valid_spawn_locations if loc not in agent_locations]

        # Place initial resources
        initial_resources = self.config.experiment.get("initial_resources", 10)
        resource_locations_indices = np.random.choice(
            len(remaining_spawn_locations), 
            size=min(initial_resources, len(remaining_spawn_locations)), 
            replace=False
        )
        resource_locations = [remaining_spawn_locations[i] for i in resource_locations_indices]
        
        for loc in resource_locations:
            # Use complex entity spawning
            self.world.spawn_entity(loc)

    # def take_turn(self):
    #     """Override take_turn to handle multi-agent coordination."""
    #     if self.other_environments:
    #         # Multi-agent mode: coordinate with other environments
    #         self._take_multi_agent_turn()
    #     else:
    #         # Single agent mode: use parent implementation
    #         super().take_turn()
    #         self.world.record_punishment_level()
    
    # def _take_multi_agent_turn(self):
    #     """Handle turn in multi-agent mode with coordination."""
    #     self.turn += 1
        
    #     # Handle entity transitions in this environment
    #     for _, x in ndenumerate(self.world.map):
    #         if x.has_transitions and not isinstance(x, Agent):
    #             x.transition(self.world)
        
    #     # Handle agent transition with multi-agent coordination
    #     for agent in self.agents:
    #         agent.transition(self.world, 
    #                        state_system=self.shared_state_system,
    #                        other_environments=self.other_environments,
    #                        use_composite_views=self.use_composite_views)
        
    #     # Record punishment level for this turn
    #     self.world.record_punishment_level()
    
    def set_multi_agent_coordination(self, other_environments, shared_state_system, agent_id):
        """Set up multi-agent coordination parameters."""
        self.other_environments = other_environments
        self.shared_state_system = shared_state_system
        self.agent_id = agent_id
    
    # def run_experiment(self, animate: bool = True, logging: bool = True, logger: Logger | None = None, output_dir: Path | None = None) -> None:
    #     """Override run_experiment to handle multi-agent coordination."""
    #     if self.other_environments:
    #         # Multi-agent mode: coordinate with other environments
    #         self._run_multi_agent_experiment(animate, logging, logger, output_dir)
    #     else:
    #         # Single agent mode: use parent implementation
    #         super().run_experiment(animate, logging, logger, output_dir)
    
    # def _run_multi_agent_experiment(self, animate: bool = True, logging: bool = True, logger: Logger | None = None, output_dir: Path | None = None) -> None:
    #     """Run experiment in multi-agent mode with coordination."""
    #     renderer = None
    #     if animate:
    #         renderer = ImageRenderer(
    #             experiment_name=self.world.__class__.__name__,
    #             record_period=self.config.experiment.record_period,
    #             num_turns=self.config.experiment.max_turns,
    #         )
        
    #     for epoch in range(self.config.experiment.epochs + 1):
    #         # Reset all environments
    #         for env in [self] + self.other_environments:
    #             env.reset()
    #             for agent in env.agents:
    #                 agent.reset()
    #             env.world.social_harm = {i: 0.0 for i in range(len([self] + self.other_environments))}
            
    #         # Start epoch action for all agents
    #         for env in [self] + self.other_environments:
    #             for agent in env.agents:
    #                 agent.model.start_epoch_action(epoch=epoch)
            
    #         # Run the environment for the specified number of turns
    #         while not self.turn >= self.config.experiment.max_turns:
    #             # Determine whether to animate this turn
    #             animate_this_turn = animate and (
    #                 epoch % self.config.experiment.record_period == 0
    #             )
                
    #             # Render if needed
    #             if animate_this_turn and renderer is not None:
    #                 renderer.add_image(self.world)
                
    #             # Take turn in this environment (which coordinates with others)
    #             self.take_turn()
                
    #             # Check if any environment is done
    #             if any(env.world.is_done for env in [self] + self.other_environments):
    #                 break
            
    #         # Log results
    #         if logging and logger is not None:
    #             total_reward = sum(env.world.total_reward for env in [self] + self.other_environments)
    #             # Use 0 for loss since get_loss() doesn't exist on the model
    #             total_loss = 0.0
    #             logger.record_turn(epoch, total_loss, total_reward, epsilon=self.config.model.epsilon)
            
    #         # Print progress
    #         if epoch % 100 == 0:
    #             avg_punishment = self.shared_state_system.get_average_punishment_level() if hasattr(self.shared_state_system, 'get_average_punishment_level') else self.shared_state_system.prob
    #             current_punishment = self.shared_state_system.prob
    #             print(f"Epoch {epoch}: Current punishment level: {current_punishment:.3f}, Average: {avg_punishment:.3f}")
    #             print(f"  Total reward: {sum(env.world.total_reward for env in [self] + self.other_environments):.2f}")
        
    #     # Save animations if needed
    #     if animate and renderer is not None:
    #         renderer.save_gif(epoch, output_dir or Path("./data/"))
        
    # def reset(self):
    #     """Reset the environment."""
    #     # Call parent reset to handle turn counter and other base functionality
    #     super().reset()
        
    #     # Reset world (parent already calls create_world, but we need our custom reset)
    #     self.world.reset()
        
    #     # Reset epoch tracking in state system
    #     self.world.state_system.reset_epoch()
        
    #     # Repopulate environment (parent already calls this, but we need our custom population)
    #     self.populate_environment()
        
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
        metrics["Global/mean_individual_score"] = np.mean([agent.individual_score for agent in self.agents])
        
        # Vote tracking metrics
        vote_stats = self.world.state_system.get_epoch_vote_stats()
        metrics["Global/epoch_vote_up"] = vote_stats['vote_up']
        metrics["Global/epoch_vote_down"] = vote_stats['vote_down']
        metrics["Global/epoch_total_votes"] = vote_stats['total_votes']
        
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
