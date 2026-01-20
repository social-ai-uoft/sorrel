"""Environment for the state punishment game."""

import random
from pathlib import Path
from typing import List, override, Dict

import numpy as np
import torch
from numpy import ndenumerate

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.environment import Environment
from sorrel.models.pytorch import PyTorchIQN
from sorrel.models.pytorch.recurrent_ppo import DualHeadRecurrentPPO
from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM
from sorrel.models.pytorch.recurrent_ppo_lstm_cpc import RecurrentPPOLSTMCPC
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.logging import Logger
from sorrel.utils.visualization import ImageRenderer

from .agents import StatePunishmentAgent
from .entities import A, B, C, D, E, EmptyEntity, Sand, Wall
from .entity_map_shuffler import EntityMapShuffler
from .world import StatePunishmentWorld


class AgentIdentityManager:
    """Manages fixed random identity vectors for agents."""
    
    def __init__(self, num_agents: int, d: int = 16, seed: int = 0, max_agents: int = None):
        """Initialize agent identity manager.
        
        Args:
            num_agents: Number of agents (minimum to generate vectors for)
            d: Dimensionality of identity vectors (default: 16)
            seed: Random seed for reproducibility (default: 0)
            max_agents: Maximum number of agent IDs to pre-generate vectors for.
                       If None, uses num_agents. Useful for preparing vectors for agent replacement.
        """
        self.num_agents = num_agents
        self.max_agents = max_agents if max_agents is not None else num_agents
        self.d = d
        # Generate vectors for max_agents to prepare for replacements/new agents
        self.identity_vectors = self.make_random_id_codes(
            num_agents=self.max_agents,
            d=d,
            seed=seed,
            rademacher=True,
            normalize=True,
        )
    
    def get_identity_vector(self, agent_id: int) -> np.ndarray:
        """Get identity vector for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Identity vector of shape (d,). Returns zeros if agent_id is invalid or out of pre-generated range.
        """
        if 0 <= agent_id < self.max_agents:
            return self.identity_vectors[agent_id].copy()
        else:
            return np.zeros(self.d, dtype=np.float32)
    
    @staticmethod
    def make_random_id_codes(
        num_agents: int,
        d: int = 16,
        seed: int = 0,
        rademacher: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate fixed random identity codes for agents.
        
        Implementation from observation_encoding.md.
        
        Args:
            num_agents: Number of agents
            d: Dimensionality of identity vectors
            seed: Random seed
            rademacher: If True, use Rademacher distribution (±1), else Gaussian
            normalize: If True, L2-normalize vectors
            
        Returns:
            Array of shape (num_agents, d) with identity vectors
        """
        rng = np.random.default_rng(seed)
        
        if rademacher:
            codes = rng.choice(
                np.array([-1.0, 1.0], dtype=np.float32),
                size=(num_agents, d),
                replace=True,
            )
        else:
            codes = rng.standard_normal(size=(num_agents, d)).astype(np.float32)
        
        if normalize:
            norms = np.linalg.norm(codes, axis=1, keepdims=True)
            codes = codes / (norms + 1e-12)
        
        return codes


class PunishmentHistoryTracker:
    """Tracks punishment history for X-step persistence."""
    
    def __init__(self, num_agents: int, persistence_steps: int = 2):
        """Initialize punishment history tracker.
        
        Args:
            num_agents: Number of agents
            persistence_steps: Number of steps to persist punishment flag (X)
        """
        self.num_agents = num_agents
        self.persistence_steps = persistence_steps
        # Store punishment steps for each agent
        self.punishment_history = {i: [] for i in range(num_agents)}
    
    def record_punishment(self, agent_id: int, step: int):
        """Record that an agent was punished at a specific step.
        
        Args:
            agent_id: ID of the agent
            step: Step number when punishment occurred
        """
        if 0 <= agent_id < self.num_agents:
            self.punishment_history[agent_id].append(step)
            # Keep only recent history (last persistence_steps entries)
            if len(self.punishment_history[agent_id]) > self.persistence_steps:
                self.punishment_history[agent_id] = self.punishment_history[agent_id][-self.persistence_steps:]
    
    def was_punished_recently(self, agent_id: int, current_step: int) -> bool:
        """Check if agent was punished in last X steps.
        
        Args:
            agent_id: ID of the agent
            current_step: Current step number
            
        Returns:
            True if agent was punished in window [current_step - persistence_steps + 1, current_step]
        """
        if agent_id not in self.punishment_history:
            return False
        
        # Check if any punishment occurred in the window
        window_start = current_step - self.persistence_steps + 1
        for step in self.punishment_history[agent_id]:
            if window_start <= step <= current_step:
                return True
        return False
    
    def step(self):
        """Optional: cleanup old entries if needed."""
        # History is automatically trimmed in record_punishment()
        pass


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
        
        # Initialize shared identity manager and punishment history tracker (if slot-based encoding enabled)
        # Check config from first individual env
        use_slot_based = individual_envs[0].config.observation.get("use_slot_based_encoding", True)
        
        # NEW: Initialize agent name management BEFORE passing to individual envs
        self._max_agent_name = -1  # Will be initialized based on initial agents
        self._agent_name_map = {}  # Maps agent_id -> agent_name
        
        if use_slot_based:
            num_agents = len(individual_envs)
            # Optionally pre-generate extra identity vectors for agent replacement
            # Check if replacement is enabled and calculate max agents needed
            max_agents = num_agents
            if any(env.config.experiment.get("enable_agent_replacement", False) for env in individual_envs):
                # Pre-generate extra vectors (e.g., 2x the number of agents) for replacements
                # This ensures identity vectors are ready even if agent IDs exceed initial count
                max_agents = max(num_agents * 2, num_agents + 100)  # Generate extra for safety
            
            self.agent_identity_manager = AgentIdentityManager(
                num_agents=num_agents,
                d=16,
                seed=0,
                max_agents=max_agents
            )
            self.punishment_history_tracker = PunishmentHistoryTracker(
                num_agents=num_agents,
                persistence_steps=individual_envs[0].config.observation.get("punishment_persistence_steps", 2)
            )
            
            # Pass to individual environments (they'll use it when creating observation specs)
            for env in individual_envs:
                env.agent_identity_manager = self.agent_identity_manager
                env.punishment_history_tracker = self.punishment_history_tracker
                env.agent_name_map = self._agent_name_map  # Pass agent_name_map for identity encoding
        else:
            self.agent_identity_manager = None
            self.punishment_history_tracker = None
        
        # Track last replacement epoch for minimum epochs between replacements
        self.last_replacement_epoch = -1
        
        # Initialize agent names (this populates _agent_name_map)
        self._initialize_agent_names()
        
        # NEW: Track agent creation/replacement epochs for tenure-based replacement
        # Maps agent_id -> epoch when agent was created/replaced
        self._agent_creation_epochs = {}
        
        # Initialize creation epochs for all initial agents (epoch 0)
        for i, env in enumerate(self.individual_envs):
            self._agent_creation_epochs[i] = 0

    def _initialize_agent_names(self) -> None:
        """Initialize agent names for all existing agents.
        
        When replacement is disabled: names are 0 to num_agents-1
        When replacement is enabled: names continue from max_agent_name
        """
        replacement_enabled = self.config.experiment.get("enable_agent_replacement", False)
        
        if not replacement_enabled:
            # Without replacement: names are 0 to X-1
            for i, env in enumerate(self.individual_envs):
                agent = env.agents[0]
                agent.agent_name = i
                agent.ancestor_name = i  # Initial agents: ancestor_name = agent_name
                self._agent_name_map[agent.agent_id] = i
                self._max_agent_name = max(self._max_agent_name, i)
        else:
            # With replacement: continue from max_agent_name
            for i, env in enumerate(self.individual_envs):
                agent = env.agents[0]
                if agent.agent_name is None:
                    # Assign new name
                    self._max_agent_name += 1
                    agent.agent_name = self._max_agent_name
                    agent.ancestor_name = self._max_agent_name  # Initial agents: ancestor_name = agent_name
                    self._agent_name_map[agent.agent_id] = self._max_agent_name
                else:
                    # Preserve existing name
                    self._agent_name_map[agent.agent_id] = agent.agent_name
                    self._max_agent_name = max(self._max_agent_name, agent.agent_name)
                    # If ancestor_name is not set, set it to agent_name (for initial agents)
                    if not hasattr(agent, 'ancestor_name') or agent.ancestor_name is None:
                        agent.ancestor_name = agent.agent_name

    def _setup_agent_name_recording(self, output_dir: Path = None) -> Path:
        """Set up directory for agent name recording.
        
        Args:
            output_dir: Base output directory (if None, uses current directory)
        
        Returns:
            Path to agent_generation_reference directory
        """
        if output_dir is None:
            output_dir = Path("./data/")
        
        agent_ref_dir = output_dir / "agent_generation_reference"
        agent_ref_dir.mkdir(parents=True, exist_ok=True)
        
        return agent_ref_dir

    def _record_agent_names(self, epoch: int, output_dir: Path = None) -> None:
        """Record all agent names for the current epoch.
        
        Args:
            epoch: Current epoch number
            output_dir: Base output directory
        """
        import pandas as pd
        
        # Set up recording directory
        agent_ref_dir = self._setup_agent_name_recording(output_dir)
        
        # Collect all agent names for this epoch
        agent_data = []
        for env in self.individual_envs:
            agent = env.agents[0]
            agent_data.append({
                'Name': agent.agent_name,
                'Ancestor_Name': getattr(agent, 'ancestor_name', agent.agent_name),  # Include ancestor_name
                'Epoch': epoch
            })
        
        # Create DataFrame
        df = pd.DataFrame(agent_data)
        
        # Determine filename (append mode or create new)
        filename = agent_ref_dir / "agent_names.csv"
        
        # Append to existing file or create new
        if filename.exists():
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        # Save to CSV
        df.to_csv(filename, index=False)

    def get_agent_name(self, agent_id: int) -> int:
        """Get the name for a given agent ID.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Agent name
        """
        return self._agent_name_map.get(agent_id, None)

    def get_all_agent_names(self) -> Dict[int, int]:
        """Get all agent ID to name mappings.
        
        Returns:
            Dictionary mapping agent_id -> agent_name
        """
        return self._agent_name_map.copy()

    def get_current_agent_names(self) -> List[int]:
        """Get list of all current agent names.
        
        Returns:
            List of agent names in order of individual_envs
        """
        return [env.agents[0].agent_name for env in self.individual_envs]

    @override
    def take_turn(self) -> None:
        """Coordinate turns across all individual environments."""
        # Update phased voting status BEFORE incrementing turn counter
        # This ensures phased voting status is set before agent actions
        if hasattr(self.shared_state_system, 'update_phased_voting'):
            self.shared_state_system.update_phased_voting()
        
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
        
        # Record punishment level for shared state system (for global average calculation)
        if hasattr(self.shared_state_system, "record_punishment_level"):
            self.shared_state_system.record_punishment_level()

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

        # Randomize agent order if configured (maintain agent-env pairing)
        # Note: all_agents and all_envs are local lists, so shuffling doesn't modify
        # env.agents or any other persistent state - only affects processing order
        if self.config.experiment.get("randomize_agent_order", False):
            # Pair agents with their environments, shuffle, then unzip
            paired = list(zip(all_agents, all_envs))
            random.shuffle(paired)  # Only shuffles the local paired list
            all_agents, all_envs = zip(*paired)
            all_agents = list(all_agents)
            all_envs = list(all_envs)

        # Check if any environment uses composite views
        use_composite = any(env.use_composite_views for env in self.individual_envs)

        if use_composite:
            # Generate composite observations for all agents
            composite_observations = self._generate_composite_observations(
                all_agents, all_envs
            )

        # Process each agent
        for i, (agent, env) in enumerate(zip(all_agents, all_envs)):
            # Check if agent uses separate models (needs time-to-vote feature)
            from sorrel.examples.state_punishment.agents import SeparateModelStatePunishmentAgent, construct_move_observation
            if isinstance(agent, SeparateModelStatePunishmentAgent):
                # For separate models, use construct_move_observation (adds time-to-vote feature)
                # construct_move_observation internally calls generate_single_view, so we don't need to get base_state
                state = construct_move_observation(
                    agent, env.world, self.shared_state_system, self.shared_social_harm,
                    self.turn, agent.vote_window_size
                )
            else:
                # Standard agent: use existing observation generation
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
                    from sorrel.observation.slot_based_observation_spec import SlotBasedObservationSpec
                    if isinstance(agent.observation_spec, SlotBasedObservationSpec):
                        # Slot-based encoding: pass current_step and use_me_encoding
                        state = agent.generate_single_view(
                            env.world, 
                            self.shared_state_system, 
                            self.shared_social_harm,
                            punishment_tracker=self.punishment_tracker,
                            current_step=self.turn,
                            use_me_encoding=True,
                        )
                    else:
                        # Legacy one-hot encoding
                        state = agent.generate_single_view(
                            env.world, 
                            self.shared_state_system, 
                            self.shared_social_harm,
                            punishment_tracker=self.punishment_tracker
                        )

            # Execute agent transition
            self._execute_agent_transition(agent, env, state)

    def _generate_composite_observations(self, all_agents, all_envs):
        """Generate composite observations for all agents."""
        from sorrel.observation.slot_based_observation_spec import SlotBasedObservationSpec
        
        # Check if any agent uses slot-based encoding
        use_slot_based = any(
            isinstance(a.observation_spec, SlotBasedObservationSpec) 
            for a in all_agents
        )
        
        composite_observations = []

        for i, (agent, env) in enumerate(zip(all_agents, all_envs)):
            if not env.use_composite_views:
                # Single view for this agent
                if isinstance(agent.observation_spec, SlotBasedObservationSpec):
                    # Use slot-based encoding with current_step
                    composite_observations.append(
                        agent.generate_single_view(
                            env.world, 
                            self.shared_state_system, 
                            self.shared_social_harm,
                            punishment_tracker=self.punishment_tracker,
                            current_step=self.turn,
                            use_me_encoding=True,
                        )
                    )
                else:
                    # Legacy one-hot encoding
                    composite_observations.append(
                        agent.generate_single_view(
                            env.world, 
                            self.shared_state_system, 
                            self.shared_social_harm,
                            punishment_tracker=self.punishment_tracker
                        )
                    )
                continue

            # Generate composite view
            if use_slot_based:
                # Slot-based encoding: two-version generation
                # Step 1: Generate all views with use_me_encoding=False (treat self as "other")
                all_views_other = []
                for a, e in zip(all_agents, all_envs):
                    if isinstance(a.observation_spec, SlotBasedObservationSpec):
                        # Generate observation with self as "other"
                        obs = a.observation_spec.observe(
                            world=e.world,
                            location=a.location,
                            observing_agent_id=None,  # No "me" encoding
                            current_step=self.turn,
                            use_me_encoding=False,
                        )
                        visual_field = obs.reshape(1, -1)
                    else:
                        # Legacy one-hot encoding
                        visual_field = self._get_agent_observation_without_scalars(a, e.world)
                    all_views_other.append(visual_field)
                
                # Step 2: Stack all views
                stacked_visual = np.concatenate(all_views_other, axis=1)
                
                # Step 3: For each agent, replace its own "other" encoding with "me" encoding
                agent_self_views = {}
                for a, e in zip(all_agents, all_envs):
                    if isinstance(a.observation_spec, SlotBasedObservationSpec):
                        # Generate "me" version for this agent
                        obs_me = a.observation_spec.observe(
                            world=e.world,
                            location=a.location,
                            observing_agent_id=a.agent_name,  # Use agent_name for identity encoding
                            current_step=self.turn,
                            use_me_encoding=True,
                        )
                        agent_self_views[a.agent_id] = obs_me.reshape(1, -1)
                
                # Step 4: Replace each agent's portion in the composite
                final_composite_observations = []
                for agent, env in zip(all_agents, all_envs):
                    if not env.use_composite_views:
                        continue
                    
                    # Copy stacked visual
                    agent_composite = stacked_visual.copy()
                    
                    # Find this agent's position in the stack
                    agent_idx = agent.agent_id
                    visual_size = agent.observation_spec.input_size[0] * agent.observation_spec.input_size[1] * agent.observation_spec.input_size[2]
                    
                    # Replace agent's portion with "me" version
                    start_idx = agent_idx * visual_size
                    end_idx = (agent_idx + 1) * visual_size
                    
                    if isinstance(agent.observation_spec, SlotBasedObservationSpec):
                        agent_me_visual = agent_self_views[agent.agent_id][:, :visual_size]
                        agent_composite[:, start_idx:end_idx] = agent_me_visual
                    
                    # Add scalar features
                    final_obs = agent._add_scalars_to_composite_state(
                        agent_composite,
                        self.shared_state_system,
                        self.shared_social_harm,
                        punishment_tracker=self.punishment_tracker
                    )
                    final_composite_observations.append(final_obs)
                
                composite_observations.extend(final_composite_observations)
            else:
                # Legacy one-hot encoding: use existing logic
                all_views = [
                    self._get_agent_observation_without_scalars(a, e.world)
                    for a, e in zip(all_agents, all_envs)
                ]
                stacked_visual = np.concatenate(all_views, axis=1)
                
                # All agents with composite views get same visual, but different scalars
                for agent, env in zip(all_agents, all_envs):
                    if not env.use_composite_views:
                        continue
                    
                    # Add agent-specific scalars
                    final_obs = agent._add_scalars_to_composite_state(
                        stacked_visual,
                        self.shared_state_system,
                        self.shared_social_harm,
                        punishment_tracker=self.punishment_tracker
                    )
                    composite_observations.append(final_obs)

        return composite_observations

    def _get_agent_observation_without_scalars(self, agent, world):
        """Get agent observation excluding all scalar values."""
        from sorrel.observation.slot_based_observation_spec import SlotBasedObservationSpec
        
        # Get only the visual field observation (no scalar features)
        if isinstance(agent.observation_spec, SlotBasedObservationSpec):
            # For slot-based encoding, use observe() with appropriate parameters
            image = agent.observation_spec.observe(
                world=world,
                location=agent.location,
                observing_agent_id=agent.agent_name,  # Use agent_name for identity encoding
                current_step=self.turn,
                use_me_encoding=True,
            )
        else:
            # Legacy one-hot encoding
            image = agent.observation_spec.observe(world, agent.location)
        
        visual_field = image.reshape(1, -1)
        return visual_field

    def _execute_agent_transition(self, agent, env, state):
        """Execute agent transition with the given state."""
        # Get action from model
        action = agent.get_action(state)

        # Execute action with shared state system and social harm
        # Pass step_count for separate models (for vote epoch detection)
        from sorrel.examples.state_punishment.agents import SeparateModelStatePunishmentAgent
        info = None
        if isinstance(agent, SeparateModelStatePunishmentAgent):
            # For separate models, pass step_count
            if self.punishment_tracker is not None:
                # Use new interface with info return
                reward, info = agent.act(
                    env.world, action, self.shared_state_system, self.shared_social_harm, 
                    return_info=True, step_count=self.turn
                )
                
                # Track punishment if it occurred
                if info.get('is_punished', False):
                    self.punishment_tracker.record_punishment(agent.agent_id)
                    # Also record in punishment history tracker (for slot-based encoding)
                    if self.punishment_history_tracker is not None:
                        self.punishment_history_tracker.record_punishment(agent.agent_id, self.turn)
            else:
                # Use original interface (backward compatible)
                reward = agent.act(
                    env.world, action, self.shared_state_system, self.shared_social_harm,
                    step_count=self.turn
                )
        else:
            # Standard agent (no step_count needed)
            if self.punishment_tracker is not None:
                # Use new interface with info return
                reward, info = agent.act(
                    env.world, action, self.shared_state_system, self.shared_social_harm, return_info=True
                )
                
                # Track punishment if it occurred
                if info.get('is_punished', False):
                    self.punishment_tracker.record_punishment(agent.agent_id)
                    # Also record in punishment history tracker (for slot-based encoding)
                    if self.punishment_history_tracker is not None:
                        self.punishment_history_tracker.record_punishment(agent.agent_id, self.turn)
            else:
                # Use original interface (backward compatible)
                reward = agent.act(
                    env.world, action, self.shared_state_system, self.shared_social_harm
                )

        # Update individual score
        agent.individual_score += reward

        # Check if done
        done = agent.is_done(env.world)

        # Add to memory (skip if action was blocked)
        env.world.total_reward += reward
        # Check if action was blocked - don't store blocked transitions in memory
        action_blocked = info is not None and info.get('action_blocked') is not None
        if not action_blocked:
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

    def replace_agent_model(
        self,
        agent_id: int,
        model_path: str = None,
        replacement_epoch: int = None,  # NEW: Track when replacement happens
    ) -> None:
        """Replace an agent's model and memory buffer, resetting all tracking attributes.
        
        Args:
            agent_id: ID of the agent to replace
            model_path: Path to pretrained model checkpoint. If None, creates fresh model.
            replacement_epoch: Epoch when replacement occurs (for tenure tracking)
        
        Raises:
            ValueError: If agent_id is invalid
            FileNotFoundError: If model_path is specified but file doesn't exist
        """
        # Validate agent_id
        if agent_id < 0 or agent_id >= len(self.individual_envs):
            raise ValueError(f"Invalid agent_id: {agent_id}. Must be between 0 and {len(self.individual_envs) - 1}")
        
        # Get the environment and agent
        env = self.individual_envs[agent_id]
        old_agent = env.agents[0]
        
        # Store configuration from old agent (to preserve settings)
        observation_spec = old_agent.observation_spec
        action_spec = old_agent.action_spec
        agent_id_value = old_agent.agent_id  # Keep the same agent_id
        
        # NEW: Assign new name for replaced agent (increment max_agent_name)
        self._max_agent_name += 1
        new_agent_name = self._max_agent_name
        
        # NEW: Inherit ancestor_name from old agent (lineage tracking)
        ancestor_name = getattr(old_agent, 'ancestor_name', old_agent.agent_name)
        
        # Store all configuration flags
        use_composite_views = old_agent.use_composite_views
        use_composite_actions = old_agent.use_composite_actions
        simple_foraging = old_agent.simple_foraging
        use_random_policy = old_agent.use_random_policy
        punishment_level_accessible = old_agent.punishment_level_accessible
        social_harm_accessible = old_agent.social_harm_accessible
        delayed_punishment = old_agent.delayed_punishment
        important_rule = old_agent.important_rule
        punishment_observable = old_agent.punishment_observable
        disable_punishment_info = old_agent.disable_punishment_info
        
        # Calculate model input size (same as old agent)
        # NOTE: Scalar features are now 3: punishment_level, social_harm, is_phased_voting (third_feature removed for slot-based encoding)
        base_flattened_size = (
            observation_spec.input_size[0]
            * observation_spec.input_size[1]
            * observation_spec.input_size[2]
            + 3  # punishment_level, social_harm, is_phased_voting (third_feature removed)
        )
        
        # Add punishment observation features if enabled
        if env.config.experiment.get("observe_other_punishments", False):
            total_num_agents = env.config.experiment.get("total_num_agents", len(self.individual_envs))
            num_other_agents = total_num_agents - 1
            base_flattened_size += num_other_agents
        
        # Adjust for composite views
        if use_composite_views:
            flattened_size = base_flattened_size * env.config.experiment.get("total_num_agents", len(self.individual_envs))
        else:
            flattened_size = base_flattened_size
        
        # Get model type from config
        model_type = env.config.model.get("type", "iqn")
        
        # Create new model with same architecture
        if model_type == "ppo":
            use_dual_head = env.config.model.get("use_dual_head", True)
            obs_dim = (
                observation_spec.input_size[0],
                observation_spec.input_size[1],
                observation_spec.input_size[2],
            )
            new_model = DualHeadRecurrentPPO(
                input_size=(flattened_size,),
                action_space=action_spec.n_actions,
                layer_size=env.config.model.layer_size,
                epsilon=env.config.model.epsilon,
                epsilon_min=env.config.model.epsilon_min,
                device=env.config.model.device,
                seed=torch.random.seed(),  # Fresh random seed
                use_dual_head=use_dual_head,
                obs_dim=obs_dim,
                n_move_actions=4,
                n_vote_actions=3,  # Always 3: 0=no_vote, 1=vote_increase, 2=vote_decrease
                gamma=env.config.model.GAMMA,
                lr=env.config.model.LR,
                clip_param=env.config.model.ppo_clip_param,
                K_epochs=env.config.model.ppo_k_epochs,
                batch_size=env.config.model.batch_size,
                entropy_start=env.config.model.ppo_entropy_start,
                entropy_end=env.config.model.ppo_entropy_end,
                entropy_decay_steps=env.config.model.ppo_entropy_decay_steps,
                max_grad_norm=env.config.model.ppo_max_grad_norm,
                gae_lambda=env.config.model.ppo_gae_lambda,
                use_composite_actions=old_agent.use_composite_actions,
                rollout_length=env.config.model.ppo_rollout_length,
            )
        elif model_type == "ppo_lstm":
            obs_dim = (
                observation_spec.input_size[0],
                observation_spec.input_size[1],
                observation_spec.input_size[2],
            )
            new_model = RecurrentPPOLSTM(
                input_size=(flattened_size,),
                action_space=action_spec.n_actions,
                layer_size=env.config.model.layer_size,
                epsilon=env.config.model.epsilon,
                epsilon_min=env.config.model.epsilon_min,
                device=env.config.model.device,
                seed=torch.random.seed(),  # Fresh random seed
                # Observation processing - use "flattened" since we have extra scalar features
                obs_type="flattened",
                obs_dim=obs_dim,
                # PPO hyperparameters
                gamma=env.config.model.GAMMA,
                lr=env.config.model.LR,
                clip_param=env.config.model.ppo_clip_param,
                K_epochs=env.config.model.ppo_k_epochs,
                batch_size=env.config.model.batch_size,
                entropy_start=env.config.model.ppo_entropy_start,
                entropy_end=env.config.model.ppo_entropy_end,
                entropy_decay_steps=env.config.model.ppo_entropy_decay_steps,
                max_grad_norm=env.config.model.ppo_max_grad_norm,
                gae_lambda=env.config.model.ppo_gae_lambda,
                rollout_length=env.config.model.ppo_rollout_length,
                # LSTM-specific parameters
                hidden_size=256,
            )
        else:
            new_model = PyTorchIQN(
                input_size=(flattened_size,),
                action_space=action_spec.n_actions,
                layer_size=env.config.model.layer_size,
                epsilon=env.config.model.epsilon,
                epsilon_min=env.config.model.epsilon_min,
                device=env.config.model.device,
                seed=torch.random.seed(),  # Fresh random seed
                n_frames=env.config.model.n_frames,
                n_step=env.config.model.n_step,
                sync_freq=env.config.model.sync_freq,
                model_update_freq=env.config.model.model_update_freq,
                batch_size=env.config.model.batch_size,
                memory_size=env.config.model.memory_size,
                LR=env.config.model.LR,
                TAU=env.config.model.TAU,
                GAMMA=env.config.model.GAMMA,
                n_quantiles=env.config.model.n_quantiles,
            )
        
        # Load pretrained model if path is specified
        if model_path is not None and model_path != "":
            from pathlib import Path
            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(
                    f"Model checkpoint not found at specified path: {model_path}"
                )
            
            try:
                new_model.load(model_file)
                print(f"Loaded pretrained model for agent {agent_id} from {model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model from {model_path} for agent {agent_id}: {e}"
                )
        
        # Extract norm enforcer config (same as in setup_agents)
        norm_enforcer_config = None
        if env.config.get("norm_enforcer", {}).get("enabled", False):
            norm_enforcer_config = {
                "decay_rate": env.config.norm_enforcer.get("decay_rate", 0.995),
                "internalization_threshold": env.config.norm_enforcer.get("internalization_threshold", 5.0),
                "max_norm_strength": env.config.norm_enforcer.get("max_norm_strength", 10.0),
                "intrinsic_scale": env.config.norm_enforcer.get("intrinsic_scale", -0.5),
                "use_state_punishment": env.config.norm_enforcer.get("use_state_punishment", True),
                "harmful_resources": env.config.norm_enforcer.get("harmful_resources", ["A", "B", "C", "D", "E"]),
                "device": env.config.norm_enforcer.get("device", env.config.model.device),
            }
        
        # Create new agent with same configuration but new model
        # NOTE: New agent gets a fresh norm enforcer (norm_strength = 0.0) since
        # a new NormEnforcer instance is created during agent initialization.
        # This ensures replaced agents start with no internalized norms.
        new_agent = StatePunishmentAgent(
            observation_spec=observation_spec,
            action_spec=action_spec,
            model=new_model,
            agent_id=agent_id_value,
            agent_name=new_agent_name,  # NEW: Assign new name for replaced agent
            ancestor_name=ancestor_name,  # NEW: Inherit ancestor_name from old agent
            use_composite_views=use_composite_views,
            use_composite_actions=use_composite_actions,
            simple_foraging=simple_foraging,
            use_random_policy=use_random_policy,
            punishment_level_accessible=punishment_level_accessible,
            social_harm_accessible=social_harm_accessible,
            delayed_punishment=delayed_punishment,
            important_rule=important_rule,
            punishment_observable=punishment_observable,
            disable_punishment_info=disable_punishment_info,
            use_norm_enforcer=env.config.get("norm_enforcer", {}).get("enabled", False),
            norm_enforcer_config=norm_enforcer_config,
        )
        
        # Explicitly reset norm enforcer for replaced agent (redundant but explicit)
        # Since a new agent is created, it already has a fresh norm enforcer,
        # but we reset it here to be explicit about the intent.
        if new_agent.norm_enforcer is not None:
            new_agent.norm_enforcer.reset()
        
        # Preserve agent location in the world
        old_location = None
        if hasattr(old_agent, 'location') and old_agent.location is not None:
            old_location = old_agent.location
            # Remove old agent from world
            env.world.remove(old_location)
        
        # Replace the agent in the environment
        env.agents[0] = new_agent
        
        # Add new agent to world at the same location (if it existed)
        if old_location is not None:
            env.world.add(old_location, new_agent)
        
        # Reset shared_social_harm for this agent (if it exists)
        if agent_id in self.shared_social_harm:
            self.shared_social_harm[agent_id] = 0.0
        
        # NEW: Update agent name map with new name
        self._agent_name_map[agent_id] = new_agent_name
        
        # Update agent_name_map in all environments and observation specs (for slot-based encoding)
        for env in self.individual_envs:
            if hasattr(env, 'agent_name_map'):
                env.agent_name_map = self._agent_name_map
            # Update observation spec's agent_name_map if it exists (for slot-based encoding)
            for agent in env.agents:
                if hasattr(agent.observation_spec, 'agent_name_map'):
                    agent.observation_spec.agent_name_map = self._agent_name_map
        
        # NEW: Record replacement epoch for tenure tracking
        if replacement_epoch is not None:
            self._agent_creation_epochs[agent_id] = replacement_epoch

    def replace_agents(
        self,
        agent_ids: List[int],
        model_path: str = None,
        replacement_epoch: int = None,  # NEW: Track when replacement happens
    ) -> None:
        """Replace multiple agents' models and memory buffers.
        
        Args:
            agent_ids: List of agent IDs to replace
            model_path: Path to pretrained model checkpoint. If None, creates fresh models.
            replacement_epoch: Epoch when replacement occurs (for tenure tracking)
        
        Raises:
            ValueError: If any agent_id is invalid or list is empty
        """
        if not agent_ids:
            return  # Nothing to do
        
        # Validate all agent IDs
        for agent_id in agent_ids:
            if agent_id < 0 or agent_id >= len(self.individual_envs):
                raise ValueError(
                    f"Invalid agent_id: {agent_id}. Must be between 0 and {len(self.individual_envs) - 1}"
                )
        
        # Replace each agent
        for agent_id in agent_ids:
            self.replace_agent_model(agent_id, model_path, replacement_epoch)

    def select_agents_to_replace(
        self,
        num_agents: int = None,
        selection_mode: str = "first_n",
        specified_ids: List[int] = None,
        replacement_probability: float = 0.1,
        current_epoch: int = 0,  # NEW: Current epoch for tenure calculation
    ) -> List[int]:
        """Select which agents to replace based on selection mode.
        
        Args:
            num_agents: Number of agents to select (ignored for "probability" mode)
            selection_mode: "first_n", "random", "specified_ids", "probability", or "random_with_tenure"
            specified_ids: List of agent IDs (used when selection_mode is "specified_ids")
            replacement_probability: Probability of each agent being replaced (used when selection_mode is "probability")
            current_epoch: Current epoch number (used for tenure calculation in "random_with_tenure" mode)
        
        Returns:
            List of agent IDs to replace
        
        Raises:
            ValueError: If selection_mode is invalid or parameters are invalid
        """
        total_agents = len(self.individual_envs)
        
        if selection_mode == "probability":
            # Probability-based selection: each agent independently evaluated
            if not (0.0 <= replacement_probability <= 1.0):
                raise ValueError(
                    f"replacement_probability must be between 0.0 and 1.0, got {replacement_probability}"
                )
            
            import random
            agent_ids = []
            for agent_id in range(total_agents):
                if random.random() < replacement_probability:
                    agent_ids.append(agent_id)
            
            return agent_ids
        
        # For other modes, num_agents is required
        if num_agents is None:
            raise ValueError("num_agents must be provided when selection_mode is not 'probability'")
        
        if num_agents <= 0:
            return []
        
        if num_agents > total_agents:
            raise ValueError(
                f"Cannot replace {num_agents} agents when only {total_agents} exist"
            )
        
        if selection_mode == "first_n":
            # Select first N agents
            return list(range(num_agents))
        
        elif selection_mode == "random":
            # Select N random agents
            import random
            return random.sample(range(total_agents), num_agents)
        
        elif selection_mode == "specified_ids":
            # Use specified IDs
            if specified_ids is None:
                raise ValueError("specified_ids must be provided when selection_mode is 'specified_ids'")
            
            # Validate specified IDs
            for agent_id in specified_ids:
                if agent_id < 0 or agent_id >= total_agents:
                    raise ValueError(
                        f"Invalid agent_id in specified_ids: {agent_id}. "
                        f"Must be between 0 and {total_agents - 1}"
                    )
            
            # Return up to num_agents from specified_ids
            return specified_ids[:num_agents]
        
        elif selection_mode == "random_with_tenure":
            # NEW: Random selection with minimum tenure constraint
            if num_agents is None:
                raise ValueError("num_agents must be provided when selection_mode is 'random_with_tenure'")
            
            if num_agents <= 0:
                return []
            
            # Get configuration parameters
            initial_agents_count = self.config.experiment.get("replacement_initial_agents_count", 0)
            minimum_tenure = self.config.experiment.get("replacement_minimum_tenure_epochs", 10)
            replacement_start_epoch = self.config.experiment.get("replacement_start_epoch", 0)
            
            # Find eligible agents (those that can be replaced)
            # Eligibility rule: agent can be replaced when:
            #   current_epoch >= max(replacement_start_epoch, creation_epoch + minimum_tenure_epochs)
            # This ensures both conditions are met:
            #   1. Replacement has started (current_epoch >= replacement_start_epoch)
            #   2. Agent has minimum tenure (current_epoch >= creation_epoch + minimum_tenure_epochs)
            eligible_agent_ids = []
            
            for agent_id in range(total_agents):
                # Get when this agent was created/replaced
                creation_epoch = self._agent_creation_epochs.get(agent_id, 0)
                
                # Calculate minimum epoch when this agent can be replaced
                # Must wait for: (1) replacement to start, (2) minimum tenure to pass
                earliest_replacement_epoch = max(
                    replacement_start_epoch,  # When replacement feature starts
                    creation_epoch + minimum_tenure  # When minimum tenure is met
                )
                
                # Agent is eligible if current epoch >= earliest replacement epoch
                if current_epoch >= earliest_replacement_epoch:
                    eligible_agent_ids.append(agent_id)
            
            # Check if we have enough eligible agents
            if len(eligible_agent_ids) < num_agents:
                # Not enough eligible agents - return all eligible ones (or empty list)
                import random
                return random.sample(eligible_agent_ids, min(len(eligible_agent_ids), num_agents)) if eligible_agent_ids else []
            
            # Randomly select from eligible agents
            import random
            return random.sample(eligible_agent_ids, num_agents)
        
        else:
            raise ValueError(
                f"Invalid selection_mode: {selection_mode}. "
                f"Must be 'first_n', 'random', 'specified_ids', 'probability', or 'random_with_tenure'"
            )

    @override
    def run_experiment(
        self,
        animate: bool = True,
        logging: bool = True,
        logger: Logger | None = None,
        output_dir: Path | None = None,
        probe_test_logger = None,
    ) -> None:
        """Run the multi-agent experiment with coordination and optional probe tests."""
        renderer = None
        if animate:
            renderer = MultiWorldImageRenderer(
                experiment_name=self.world.__class__.__name__,
                record_period=self.config.experiment.record_period,
                num_turns=self.config.experiment.max_turns,
                individual_envs=self.individual_envs,
            )

        # Initialize probe test environment if probe test logger is provided
        probe_test_env = None
        if probe_test_logger is not None:
            from sorrel.examples.state_punishment.probe_test import setup_probe_test_environment, PROBE_TEST_CONFIG
            probe_test_env, _, _ = setup_probe_test_environment(
                self.config, 
                getattr(self, 'args', None), 
                PROBE_TEST_CONFIG["use_important_rule"]
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
            
            # Reset all environments (this also resets epoch-specific tracking)
            self.reset()
            
            # Reset epoch-specific tracking for shared state system
            if hasattr(self.shared_state_system, "reset_epoch"):
                self.shared_state_system.reset_epoch()
            
            # ============================================================
            # AGENT REPLACEMENT LOGIC (NEW)
            # ============================================================
            # IMPORTANT: This entire block is only executed when enable_agent_replacement=True
            # When False (default), this code is completely skipped - no performance impact
            # Check if agent replacement should occur this epoch
            replacement_config = self.config.experiment
            if replacement_config.get("enable_agent_replacement", False):
                # All replacement code is inside this block - safe when feature is disabled
                agents_to_replace = replacement_config.get("agents_to_replace_per_epoch", 0)
                start_epoch = replacement_config.get("replacement_start_epoch", 0)
                end_epoch = replacement_config.get("replacement_end_epoch", None)
                
                # Get selection mode to determine if we should check replacement conditions
                selection_mode = replacement_config.get("replacement_selection_mode", "first_n")
                min_epochs_between = replacement_config.get("replacement_min_epochs_between", 0)
                
                # Check if enough epochs have passed since last replacement
                epochs_since_last_replacement = epoch - self.last_replacement_epoch if self.last_replacement_epoch >= 0 else float('inf')
                enough_epochs_passed = epochs_since_last_replacement >= min_epochs_between
                
                # For probability mode, check probability > 0 instead of agents_to_replace > 0
                if selection_mode == "probability":
                    replacement_prob = replacement_config.get("replacement_probability", 0.0)
                    should_replace = (
                        replacement_prob > 0.0 and
                        epoch >= start_epoch and
                        (end_epoch is None or epoch <= end_epoch) and
                        enough_epochs_passed
                    )
                else:
                    # For other modes, check agents_to_replace > 0
                    should_replace = (
                        agents_to_replace > 0 and
                        epoch >= start_epoch and
                        (end_epoch is None or epoch <= end_epoch) and
                        enough_epochs_passed
                    )
                
                if should_replace:
                    try:
                        # Get selection mode and model path
                        specified_ids = replacement_config.get("replacement_agent_ids", None)
                        model_path = replacement_config.get("new_agent_model_path", None)
                        replacement_prob = replacement_config.get("replacement_probability", 0.1)
                        
                        # Select agents to replace
                        if selection_mode == "probability":
                            # Probability mode: num_agents is ignored
                            agent_ids = self.select_agents_to_replace(
                                num_agents=None,
                                selection_mode=selection_mode,
                                replacement_probability=replacement_prob,
                                current_epoch=epoch,  # NEW: Pass current epoch
                            )
                        elif selection_mode == "random_with_tenure":
                            # NEW: Handle random_with_tenure mode
                            agent_ids = self.select_agents_to_replace(
                                num_agents=agents_to_replace,
                                selection_mode=selection_mode,
                                current_epoch=epoch,  # NEW: Pass current epoch
                            )
                        else:
                            # Other modes: use num_agents
                            agent_ids = self.select_agents_to_replace(
                                num_agents=agents_to_replace,
                                selection_mode=selection_mode,
                                specified_ids=specified_ids,
                                current_epoch=epoch,  # NEW: Pass current epoch
                            )
                        
                        # Replace selected agents
                        if agent_ids:
                            self.replace_agents(agent_ids, model_path, replacement_epoch=epoch)  # NEW: Pass epoch
                            self.last_replacement_epoch = epoch  # Update last replacement epoch
                            print(f"Epoch {epoch}: Replaced {len(agent_ids)} agent(s) "
                                  f"(IDs: {agent_ids}, mode: {selection_mode})")
                        
                    except (ValueError, FileNotFoundError, RuntimeError) as e:
                        # If replacement fails, log and continue
                        print(f"Epoch {epoch}: Agent replacement skipped - {e}")
                elif not enough_epochs_passed and min_epochs_between > 0:
                    # Only log if we're in the replacement window but skipped due to minimum epochs
                    if (epoch >= start_epoch and (end_epoch is None or epoch <= end_epoch)):
                        epochs_needed = min_epochs_between - epochs_since_last_replacement
                        # Only print occasionally to avoid spam (every 100 epochs)
                        if epoch % 100 == 0:
                            print(f"Epoch {epoch}: Replacement skipped - need {epochs_needed} more epoch(s) "
                                  f"(minimum {min_epochs_between} epochs between replacements)")
            # ============================================================
            # END AGENT REPLACEMENT LOGIC
            # ============================================================
            
            # Start epoch action for all agents
            for env in self.individual_envs:
                for agent in env.agents:
                    from sorrel.examples.state_punishment.agents import SeparateModelStatePunishmentAgent
                    if isinstance(agent, SeparateModelStatePunishmentAgent):
                        # Separate models: call start_epoch_action on both models
                        agent.start_epoch_action(epoch=epoch)
                    else:
                        # Standard agent: call on model
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
                    from sorrel.examples.state_punishment.agents import SeparateModelStatePunishmentAgent
                    if isinstance(agent, SeparateModelStatePunishmentAgent):
                        # Separate models: call end_epoch_action on both models
                        agent.move_model.end_epoch_action(epoch=epoch)
                        agent.vote_model.end_epoch_action(epoch=epoch)
                    else:
                        # Standard agent: call on model
                        agent.model.end_epoch_action(epoch=epoch)

            # Train all agents at the end of each epoch
            total_loss = 0.0
            loss_count = 0
            for env in self.individual_envs:
                for agent in env.agents:
                    # Check if agent uses separate models
                    from sorrel.examples.state_punishment.agents import SeparateModelStatePunishmentAgent
                    if isinstance(agent, SeparateModelStatePunishmentAgent):
                        # === Train Move Model ===
                        # Note: start_epoch_action() is already called at epoch start (line 1107-1116)
                        # Check sampleable size (reduced by n_frames + 1 due to frame stacking)
                        sampleable_size = max(1, len(agent.move_model.memory) - agent.move_model.n_frames - 1)
                        if sampleable_size >= agent.move_model.batch_size:
                            loss = agent.move_model.train_step()  # Uses self.GAMMA (standard discount)
                            if loss is not None and loss != 0.0:
                                total_loss += float(loss)
                                loss_count += 1
                        
                        # === Train Vote Model ===
                        # Train when window completes and we have enough samples
                        sampleable_size = max(1, len(agent.vote_model.memory) - agent.vote_model.n_frames - 1)
                        if agent.should_train_vote_model(self.turn) and sampleable_size >= agent.vote_model.batch_size:
                            # Use macro discount: custom_gamma = γ^X
                            # train_step() internally computes: discount = custom_gamma**n_step
                            loss = agent.vote_model.train_step(custom_gamma=agent.vote_gamma)
                            if loss is not None and loss != 0.0:
                                total_loss += float(loss)
                                loss_count += 1
                    elif hasattr(agent.model, "train_step"):
                        # Existing training code for single-model agents
                        # Check if it's a PPO model (uses rollout_memory) or IQN model (uses memory)
                        from sorrel.models.pytorch.recurrent_ppo import DualHeadRecurrentPPO
                        from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM
                        from sorrel.models.pytorch.recurrent_ppo_lstm_cpc import RecurrentPPOLSTMCPC
                        if isinstance(agent.model, (DualHeadRecurrentPPO, RecurrentPPOLSTM, RecurrentPPOLSTMCPC)):
                            # PPO: train if we have any data (train_step handles the rest)
                            if len(agent.model.rollout_memory["states"]) > 0:
                                loss = agent.model.train_step()
                                if loss is not None and loss != 0.0:
                                    total_loss += float(loss)
                                    loss_count += 1
                        else:
                            # IQN: check memory length against batch_size
                            if len(agent.model.memory) >= agent.model.batch_size:
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
                # For separate model agents, we log move_model epsilon as primary (for backward compatibility)
                # and vote_model epsilon as a separate metric
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
                
                # Also collect vote epsilon for separate model agents
                vote_epsilons = []
                for env in self.individual_envs:
                    agent = env.agents[0]
                    from sorrel.examples.state_punishment.agents import SeparateModelStatePunishmentAgent
                    if isinstance(agent, SeparateModelStatePunishmentAgent):
                        vote_epsilons.append(agent.vote_model.epsilon)

                logger.record_turn(
                    epoch, avg_loss, total_reward, epsilon=current_epsilon,
                    vote_epsilon=np.mean(vote_epsilons) if vote_epsilons else None
                )

            # Update epsilon for all agents
            for env in self.individual_envs:
                for agent in env.agents:
                    from sorrel.examples.state_punishment.agents import SeparateModelStatePunishmentAgent
                    if isinstance(agent, SeparateModelStatePunishmentAgent):
                        # Decay epsilon for both move and vote models
                        agent.move_model.epsilon_decay(self.config.model.epsilon_decay)
                        agent.vote_model.epsilon_decay(self.config.model.epsilon_decay)
                    else:
                        # Standard agent: decay epsilon for single model
                        agent.model.epsilon_decay(self.config.model.epsilon_decay)

            # Run probe test at specified intervals
            if (probe_test_logger is not None and 
                probe_test_env is not None and
                epoch > 0 and 
                PROBE_TEST_CONFIG.get("frequency", 0) > 0 and
                epoch % PROBE_TEST_CONFIG["frequency"] == 0):
                
                print(f"\n--- Running Probe Test at Training Epoch {epoch} ---")
                
                # Run probe test
                from sorrel.examples.state_punishment.probe_test import run_probe_test, save_probe_test_models
                probe_results = run_probe_test(
                    self,  # training environment
                    probe_test_env,  # probe test environment
                    epoch,
                    PROBE_TEST_CONFIG["epochs"]
                )
                
                # Log probe test results
                probe_test_logger.record_probe_test(epoch, probe_results)
                
                # Save model checkpoints if requested
                if PROBE_TEST_CONFIG["save_models"]:
                    experiment_name = self.config.experiment.get("run_name", "experiment")
                    save_probe_test_models(probe_test_env, epoch, experiment_name)
                
                print(f"Probe test completed. Avg reward: {probe_results['avg_total_reward']:.2f}")
                print("--- End Probe Test ---\n")

            # Save models every X epochs
            if epoch > 0 and epoch % self.config.experiment.save_models_every == 0:
                self._save_models(epoch)

            # NEW: Record agent names for this epoch
            self._record_agent_names(epoch, output_dir)

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
        
        # Save final probe test results if probe test logger was used
        if probe_test_logger is not None:
            probe_test_logger.save_probe_test_results()

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
            # Set up mapping file path if provided
            mapping_file_path = config.experiment.get("mapping_file_path")
            if mapping_file_path:
                mapping_file_path = Path(mapping_file_path)
            
            self.entity_map_shuffler = EntityMapShuffler(
                resource_entities=resource_entities,
                csv_file_path=csv_file_path,
                enable_logging=config.experiment.get("csv_logging", False),
                shuffle_constraint=config.experiment.get("shuffle_constraint", "no_fixed"),
                mapping_file_path=mapping_file_path
            )

        # Initialize agent identity manager and punishment history tracker BEFORE super().__init__()
        # (because super().__init__() calls setup_agents() which needs these)
        # In multi-agent setup, these will be overwritten by MultiAgentStatePunishmentEnv
        # Safely access observation config (might be dict or OmegaConf)
        try:
            if hasattr(config, "observation"):
                observation_config = config.observation
                if isinstance(observation_config, dict):
                    use_slot_based = observation_config.get("use_slot_based_encoding", True)
                    persistence_steps = observation_config.get("punishment_persistence_steps", 2)
                else:
                    # OmegaConf object
                    use_slot_based = getattr(observation_config, "use_slot_based_encoding", True)
                    persistence_steps = getattr(observation_config, "punishment_persistence_steps", 2)
            else:
                # Fallback: assume slot-based encoding is enabled
                use_slot_based = True
                persistence_steps = 2
        except (AttributeError, KeyError):
            # Fallback: assume slot-based encoding is enabled
            use_slot_based = True
            persistence_steps = 2
        if use_slot_based:
            num_agents = config.experiment.num_agents
            # Optionally pre-generate extra identity vectors for agent replacement
            max_agents = num_agents
            if config.experiment.get("enable_agent_replacement", False):
                # Pre-generate extra vectors for replacements
                max_agents = max(num_agents * 2, num_agents + 100)
            
            self.agent_identity_manager = AgentIdentityManager(
                num_agents=num_agents,
                d=16,
                seed=0,
                max_agents=max_agents
            )
            self.punishment_history_tracker = PunishmentHistoryTracker(
                num_agents=config.experiment.num_agents,
                persistence_steps=persistence_steps
            )
        else:
            self.agent_identity_manager = None
            self.punishment_history_tracker = None

        # Simplified - no complex coordination needed
        super().__init__(world, config)

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents."""
        agent_num = self.config.experiment.num_agents
        agents = []

        for i in range(agent_num):
            # Check if separate models are enabled (needed for entity_list)
            use_separate_models = self.config.experiment.get("use_separate_models", False)
            
            # Create the observation spec with separate entity types for each agent
            # Include the appropriate agent class name in entity list
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
            # Add separate model agent class if using separate models
            if use_separate_models:
                entity_list.append("SeparateModelStatePunishmentAgent")
            
            # Create observation spec based on config
            use_slot_based = self.config.observation.get("use_slot_based_encoding", True)
            
            if use_slot_based:
                from sorrel.observation.slot_based_observation_spec import SlotBasedObservationSpec
                observation_spec = SlotBasedObservationSpec(
                    entity_list=entity_list,
                    full_view=self.config.model.full_view,
                    vision_radius=self.config.model.agent_vision_radius,
                    env_dims=(
                        (self.config.world.height, self.config.world.width)
                        if self.config.model.full_view
                        else None
                    ),
                    agent_identity_manager=self.agent_identity_manager,
                    punishment_history_tracker=self.punishment_history_tracker,
                    agent_name_map=getattr(self, 'agent_name_map', {}),  # Use agent_name_map if available
                )
            else:
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

            # Create the model with extra features (punishment_level, social_harm, is_phased_voting)
            # The input_size should be a tuple representing the flattened dimensions
            # We always add 3 extra features: punishment_level (accessible value or 0), social_harm, is_phased_voting
            # NOTE: third_feature removed for slot-based encoding
            base_flattened_size = (
                observation_spec.input_size[0]
                * observation_spec.input_size[1]
                * observation_spec.input_size[2]
                + 3  # punishment_level, social_harm, is_phased_voting (third_feature removed)
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
            
            # Get model type from config
            model_type = self.config.model.get("type", "iqn")
            
            if model_type == "ppo":
                # PPO model (GRU-based)
                use_dual_head = self.config.model.get("use_dual_head", True)
                # Derive obs_dim from observation spec
                obs_dim = (
                    observation_spec.input_size[0],  # C
                    observation_spec.input_size[1],  # H
                    observation_spec.input_size[2],  # W
                )
                model = DualHeadRecurrentPPO(
                    input_size=(flattened_size,),
                    action_space=action_spec.n_actions,
                    layer_size=self.config.model.layer_size,
                    epsilon=self.config.model.epsilon,
                    epsilon_min=self.config.model.epsilon_min,
                    device=self.config.model.device,
                    seed=torch.random.seed(),
                    use_dual_head=use_dual_head,
                    obs_dim=obs_dim,
                    n_move_actions=4,
                    n_vote_actions=3,  # Always 3: 0=no_vote, 1=vote_increase, 2=vote_decrease
                    gamma=self.config.model.GAMMA,
                    lr=self.config.model.LR,
                    clip_param=self.config.model.ppo_clip_param,
                    K_epochs=self.config.model.ppo_k_epochs,
                    batch_size=self.config.model.batch_size,
                    entropy_start=self.config.model.ppo_entropy_start,
                    entropy_end=self.config.model.ppo_entropy_end,
                    entropy_decay_steps=self.config.model.ppo_entropy_decay_steps,
                    max_grad_norm=self.config.model.ppo_max_grad_norm,
                    gae_lambda=self.config.model.ppo_gae_lambda,
                    use_composite_actions=self.use_composite_actions,
                    rollout_length=self.config.model.ppo_rollout_length,
                )
            elif model_type == "ppo_lstm":
                # PPO LSTM model (LSTM-based, single-head)
                # Derive obs_dim from observation spec
                obs_dim = (
                    observation_spec.input_size[0],  # C
                    observation_spec.input_size[1],  # H
                    observation_spec.input_size[2],  # W
                )
                model = RecurrentPPOLSTM(
                    input_size=(flattened_size,),
                    action_space=action_spec.n_actions,
                    layer_size=self.config.model.layer_size,
                    epsilon=self.config.model.epsilon,
                    epsilon_min=self.config.model.epsilon_min,
                    device=self.config.model.device,
                    seed=torch.random.seed(),
                    # Observation processing - use "flattened" since we have extra scalar features
                    obs_type="flattened",  # Process entire flattened vector with FC layers
                    obs_dim=obs_dim,  # Still provide for reference, but won't be used in flattened mode
                    # PPO hyperparameters
                    gamma=self.config.model.GAMMA,
                    lr=self.config.model.LR,
                    clip_param=self.config.model.ppo_clip_param,
                    K_epochs=self.config.model.ppo_k_epochs,
                    batch_size=self.config.model.batch_size,
                    entropy_start=self.config.model.ppo_entropy_start,
                    entropy_end=self.config.model.ppo_entropy_end,
                    entropy_decay_steps=self.config.model.ppo_entropy_decay_steps,
                    max_grad_norm=self.config.model.ppo_max_grad_norm,
                    gae_lambda=self.config.model.ppo_gae_lambda,
                    rollout_length=self.config.model.ppo_rollout_length,
                    # LSTM-specific parameters
                    hidden_size=256,  # LSTM hidden size (can match layer_size if desired)
                )
            elif model_type == "ppo_lstm_cpc":
                # PPO LSTM with CPC model (LSTM-based with Contrastive Predictive Coding)
                # Derive obs_dim from observation spec
                obs_dim = (
                    observation_spec.input_size[0],  # C
                    observation_spec.input_size[1],  # H
                    observation_spec.input_size[2],  # W
                )
                model = RecurrentPPOLSTMCPC(
                    input_size=(flattened_size,),
                    action_space=action_spec.n_actions,
                    layer_size=self.config.model.layer_size,
                    epsilon=self.config.model.epsilon,
                    epsilon_min=self.config.model.epsilon_min,
                    device=self.config.model.device,
                    seed=torch.random.seed(),
                    # Observation processing - use "flattened" since we have extra scalar features
                    obs_type="flattened",  # Process entire flattened vector with FC layers
                    obs_dim=obs_dim,  # Still provide for reference, but won't be used in flattened mode
                    # PPO hyperparameters
                    gamma=self.config.model.GAMMA,
                    lr=self.config.model.LR,
                    clip_param=self.config.model.ppo_clip_param,
                    K_epochs=self.config.model.ppo_k_epochs,
                    batch_size=self.config.model.batch_size,
                    entropy_start=self.config.model.ppo_entropy_start,
                    entropy_end=self.config.model.ppo_entropy_end,
                    entropy_decay_steps=self.config.model.ppo_entropy_decay_steps,
                    max_grad_norm=self.config.model.ppo_max_grad_norm,
                    gae_lambda=self.config.model.ppo_gae_lambda,
                    rollout_length=self.config.model.ppo_rollout_length,
                    # LSTM-specific parameters
                    hidden_size=256,  # LSTM hidden size (can match layer_size if desired)
                    # CPC-specific parameters
                    use_cpc=self.config.model.get("use_cpc", True),
                    cpc_horizon=self.config.model.get("cpc_horizon", 30),
                    cpc_weight=self.config.model.get("cpc_weight", 1.0),
                    cpc_projection_dim=self.config.model.get("cpc_projection_dim", None),
                    cpc_temperature=self.config.model.get("cpc_temperature", 0.07),
                )
            else:
                # IQN model (default)
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

            # Extract norm enforcer config
            norm_enforcer_config = None
            if self.config.get("norm_enforcer", {}).get("enabled", False):
                norm_enforcer_config = {
                    "decay_rate": self.config.norm_enforcer.get("decay_rate", 0.995),
                    "internalization_threshold": self.config.norm_enforcer.get("internalization_threshold", 5.0),
                    "max_norm_strength": self.config.norm_enforcer.get("max_norm_strength", 10.0),
                    "intrinsic_scale": self.config.norm_enforcer.get("intrinsic_scale", -0.5),
                    "use_state_punishment": self.config.norm_enforcer.get("use_state_punishment", True),
                    "harmful_resources": self.config.norm_enforcer.get("harmful_resources", ["A", "B", "C", "D", "E"]),
                    "device": self.config.norm_enforcer.get("device", self.config.model.device),
                }
            
            # Check if separate models are enabled
            use_separate_models = self.config.experiment.get("use_separate_models", False)
            
            # Validate that separate models are only used with IQN
            if use_separate_models and model_type != "iqn":
                raise ValueError(
                    f"use_separate_models=True is only supported with model_type='iqn', "
                    f"but got model_type='{model_type}'. "
                    f"Please set model_type='iqn' or set use_separate_models=False."
                )
            
            if use_separate_models and model_type == "iqn":
                # Create separate move and vote models
                from sorrel.examples.state_punishment.agents import SeparateModelStatePunishmentAgent
                
                # Move model uses full observation (with time-to-vote feature added)
                # Add 1 for time-to-vote feature
                move_flattened_size = flattened_size + 1
                move_model = PyTorchIQN(
                    input_size=(move_flattened_size,),  # Single observation size (IQN handles frame stacking internally)
                    action_space=4,  # Only movement actions: up, down, left, right
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
                
                # Vote model uses low-dimensional summary
                vote_window_size = self.config.experiment.get("vote_window_size", 10)
                use_window_stats = self.config.experiment.get("use_window_stats", False)
                # Base features: [punishment_level, normalized_return, normalized_social_harm] = 3
                # Must match construct_vote_observation() return shape
                vote_obs_dim = 3  # Base: punishment_level + aggregated_return + total_social_harm
                if use_window_stats:
                    # Additional features: [mean_reward_norm, violation_rate, punishment_rate] = 3
                    vote_obs_dim += 3  # Additional: mean_reward, violation_rate, punishment_rate
                
                vote_model = PyTorchIQN(
                    input_size=(vote_obs_dim,),
                    action_space=3,  # no_vote, vote_increase, vote_decrease
                    layer_size=self.config.model.layer_size,
                    epsilon=self.config.model.epsilon,
                    epsilon_min=self.config.model.epsilon_min,
                    device=self.config.model.device,
                    seed=torch.random.seed(),
                    n_frames=1,  # No frame stacking for vote model
                    n_step=1,  # Use n_step=1 for vote model (recommended in plan)
                    sync_freq=self.config.model.sync_freq,
                    model_update_freq=self.config.model.model_update_freq,
                    batch_size=self.config.model.batch_size,
                    memory_size=self.config.model.memory_size,  # Same buffer size as move model
                    LR=self.config.model.LR,
                    TAU=self.config.model.TAU,
                    GAMMA=self.config.model.GAMMA,
                    n_quantiles=self.config.model.n_quantiles,
                )
                
                agents.append(
                    SeparateModelStatePunishmentAgent(
                        observation_spec=observation_spec,
                        action_spec=action_spec,
                        move_model=move_model,
                        vote_model=vote_model,
                        agent_id=i,
                        vote_window_size=vote_window_size,
                        use_window_stats=use_window_stats,
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
                        use_norm_enforcer=self.config.get("norm_enforcer", {}).get("enabled", False),
                        norm_enforcer_config=norm_enforcer_config,
                    )
                )
            else:
                # Standard single-model agent
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
                        use_norm_enforcer=self.config.get("norm_enforcer", {}).get("enabled", False),
                        norm_enforcer_config=norm_enforcer_config,
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
