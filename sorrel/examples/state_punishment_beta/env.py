"""Environment for the state punishment game."""

import numpy as np
import torch
from typing import List

from sorrel.environment import Environment
from sorrel.action.action_spec import ActionSpec
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.models.pytorch import PyTorchIQN

from .world import StatePunishmentWorld
from .agents import StatePunishmentAgent
from .entities import EmptyEntity, Wall, Gem, Coin, Bone


class StatePunishmentEnv(Environment[StatePunishmentWorld]):
    """Environment for the state punishment game."""

    def __init__(self, world: StatePunishmentWorld, config: dict) -> None:
        self.use_composite_views = config.get("use_composite_views", False)
        self.use_composite_actions = config.get("use_composite_actions", False)
        super().__init__(world, config)

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents."""
        agent_num = self.config.experiment.num_agents
        agents = []
        
        for i in range(agent_num):
            # Create the observation spec
            entity_list = ["EmptyEntity", "Wall", "Gem", "Coin", "Bone", "StatePunishmentAgent"]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=self.config.model.full_view,
                vision_radius=self.config.model.agent_vision_radius,
                env_dims=(self.config.world.height, self.config.world.width) if self.config.model.full_view else None,
            )
            
            # Don't override input size - let the observation spec handle it naturally

            # Create the action spec
            if self.use_composite_actions:
                # Composite actions: 4 movements × 3 voting options + noob = 13 actions
                action_names = [
                    "up_no_vote", "down_no_vote", "left_no_vote", "right_no_vote",
                    "up_increase", "down_increase", "left_increase", "right_increase", 
                    "up_decrease", "down_decrease", "left_decrease", "right_decrease",
                    "noob"
                ]
            else:
                # Simple actions: 4 movements + 2 voting + noob = 7 actions
                action_names = ["up", "down", "left", "right", "vote_increase", "vote_decrease", "noob"]
                
            action_spec = ActionSpec(action_names)

            # Create the model
            model = PyTorchIQN(
                input_size=observation_spec.input_size,
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
                )
            )

        self.agents = agents

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
            # Randomly choose which resource to place
            resource_type = np.random.choice(["gem", "coin", "bone"])
            
            if resource_type == "gem":
                self.world.add(loc, Gem(self.world.gem_value))
            elif resource_type == "coin":
                self.world.add(loc, Coin(self.world.coin_value))
            elif resource_type == "bone":
                self.world.add(loc, Bone(self.world.bone_value))

    def take_turn(self):
        """Override take_turn to add resource spawning after agent turns."""
        # Call parent take_turn to handle agent transitions
        super().take_turn()
        
        # Record punishment level for this turn
        self.world.record_punishment_level()
        
        # Spawn new resources randomly after all agents have taken their turns
        self._spawn_resources()
        
    def _spawn_resources(self):
        """Randomly spawn new resources in empty locations."""
        spawned_count = 0
        for index in np.ndindex(self.world.map.shape):
            if (isinstance(self.world.map[index], EmptyEntity) and 
                np.random.random() < self.world.spawn_prob):
                
                # Randomly choose resource type
                resource_type = np.random.choice(["gem", "coin", "bone"])
                
                if resource_type == "gem":
                    self.world.add(index, Gem(self.world.gem_value))
                elif resource_type == "coin":
                    self.world.add(index, Coin(self.world.coin_value))
                elif resource_type == "bone":
                    self.world.add(index, Bone(self.world.bone_value))
                
                spawned_count += 1
        
                    
    def reset(self):
        """Reset the environment."""
        # Call parent reset to handle turn counter and other base functionality
        super().reset()
        
        # Reset world (parent already calls create_world, but we need our custom reset)
        self.world.reset()
        
        # Repopulate environment (parent already calls this, but we need our custom population)
        self.populate_environment()
        
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
        
        return metrics
