"""Enhanced Pygame implementation of Stag Hunt Physical that uses the exact same sprites and
ASCII maps as the original Sorrel framework, with ATTACK/PUNISH actions and health system."""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import sorrel modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

import numpy as np
import pygame
from PIL import Image

from sorrel.action.action_spec import ActionSpec
from sorrel.examples.staghunt_physical.agents_v2 import StagHuntAgent, StagHuntObservation
from sorrel.examples.staghunt_physical.entities import (
    Empty,
    HareResource,
    InteractionBeam,
    AttackBeam,
    PunishBeam,
    Sand,
    Spawn,
    StagResource,
    Wall,
)
from sorrel.examples.staghunt_physical.world import StagHuntWorld


class StagHuntPhysicalASCIIPygame:
    """Enhanced pygame class for Stag Hunt Physical with ASCII maps and health system."""

    def __init__(self, config=None):
        # Initialize pygame
        pygame.init()

        # Default configuration using ASCII map
        if config is None:
            config = {
                "world": {
                    "generation_mode": "ascii_map",
                    "ascii_map_file": "docs/stag_hunt_ascii_map_clean.txt",
                    "num_agents": 1,
                    "stag_reward": 1.0,  # Higher reward for stag (requires coordination)
                    "hare_reward": 0.1,  # Lower reward for hare (solo achievable)
                    "taste_reward": 0.1,  # Legacy parameter
                    "stag_health": 12,
                    "hare_health": 3,
                    "agent_health": 5,
                    "health_regeneration_rate": 0.1,
                    "reward_sharing_radius": 3,
                    "beam_length": 3,
                    "beam_radius": 1,
                    "attack_cooldown": 3,
                    "punish_cooldown": 5,
                    "beam_cooldown": 3,  # Legacy parameter
                    "respawn_lag": 10,
                    "payoff_matrix": [[4, 0], [2, 2]],
                    "interaction_reward": 1.0,
                    "freeze_duration": 5,
                    "respawn_delay": 10,
                },
                "display": {
                    "tile_size": 32,  # Same as original Sorrel
                    "fps": 10,
                    "show_health": True,
                },
            }

        self.config = config
        self.tile_size = config["display"]["tile_size"]
        self.fps = config["display"]["fps"]

        # Initialize game world with ASCII map
        self.world = StagHuntWorld(config, Empty())

        # Create environment for agent state management (frozen/respawn logic)
        from sorrel.examples.staghunt_physical.env import StagHuntEnv

        self.env = StagHuntEnv(
            self.world, []
        )  # Empty agents list, we'll add them later
        self.world.env = self.env  # Link world to environment

        # Calculate screen dimensions
        self.screen_width = self.world.width * self.tile_size
        self.screen_height = (
            self.world.height * self.tile_size + 120
        )  # Extra space for UI

        # Create screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(
            "Stag Hunt Physical ASCII Pygame - Enhanced with Health System"
        )

        # Load sprites exactly like Sorrel does
        self.sprite_cache = {}
        self.load_sprites()

        # Populate the world with entities from ASCII map
        self.populate_world()

        # Initialize agents
        self.agents = self.create_agents()
        self.current_agent_index = 0  # Index of currently controlled agent
        self.human_player = self.agents[0] if self.agents else None

        # Update environment with our agents for proper state management
        self.env.agents = self.agents
        self.turn_based = (
            len(self.agents) > 1
        )  # Enable turn-based mode for multiple agents
        self.turn_timer = 0
        self.turn_duration = self.config["display"].get(
            "turn_duration", 30
        )  # Frames per turn
        self.turn_action_taken = (
            False  # Track if current agent has taken an action this turn
        )
        self.agent_scores = [0.0] * len(self.agents)  # Individual scores for each agent
        self.full_turn_count = 0  # Count complete turns (all agents acted)
        self.agents_acted_this_turn = set()  # Track which agents have acted this turn

        # Game state
        self.running = True
        self.clock = pygame.time.Clock()
        self.score = 0
        self.turn_count = 0

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

    def load_sprites(self):
        """Load sprites exactly like Sorrel does."""
        # Get sprite paths from entities
        entities = {
            "wall": Wall(),
            "sand": Sand(),
            "spawn": Spawn(),
            "stag": StagResource(1.0, 12),  # Use separate reward values
            "hare": HareResource(0.1, 3),
            "empty": Empty(),
            "beam": InteractionBeam(),
            "attack_beam": AttackBeam(),
            "punish_beam": PunishBeam(),
        }

        # Load agent sprites for different orientations
        agent_sprites = {
            "agent_north": Path(__file__).parent.parent / "assets" / "hero-back.png",
            "agent_east": Path(__file__).parent.parent / "assets" / "hero-right.png",
            "agent_south": Path(__file__).parent.parent / "assets" / "hero.png",
            "agent_west": Path(__file__).parent.parent / "assets" / "hero-left.png",
        }

        # Load entity sprites
        for name, entity in entities.items():
            if hasattr(entity, "sprite"):
                sprite_path = entity.sprite
                if isinstance(sprite_path, Path):
                    sprite_path = str(sprite_path)
                try:
                    # Load and resize sprite exactly like Sorrel
                    pil_image = (
                        Image.open(sprite_path)
                        .resize((self.tile_size, self.tile_size))
                        .convert("RGBA")
                    )
                    # Convert PIL to pygame surface
                    pygame_image = pygame.image.fromstring(
                        pil_image.tobytes(), pil_image.size, pil_image.mode
                    )
                    self.sprite_cache[name] = pygame_image
                    print(f"Loaded sprite: {name} from {sprite_path}")
                except Exception as e:
                    print(f"Warning: Could not load sprite {sprite_path}: {e}")
                    # Create a colored placeholder
                    placeholder = pygame.Surface(
                        (self.tile_size, self.tile_size), pygame.SRCALPHA
                    )
                    if name == "wall":
                        placeholder.fill((100, 100, 100, 255))  # Gray for walls
                    elif name == "sand":
                        placeholder.fill((194, 178, 128, 255))  # Beige for sand
                    elif name == "spawn":
                        placeholder.fill((255, 255, 0, 255))  # Yellow for spawn
                    elif name == "stag":
                        placeholder.fill((255, 0, 0, 255))  # Red for stag
                    elif name == "hare":
                        placeholder.fill((0, 255, 0, 255))  # Green for hare
                    elif name == "attack_beam":
                        placeholder.fill((255, 100, 100, 128))  # Red for attack beam
                    elif name == "punish_beam":
                        placeholder.fill((255, 255, 100, 128))  # Yellow for punish beam
                    else:
                        placeholder.fill((200, 200, 200, 255))  # Light gray for empty
                    self.sprite_cache[name] = placeholder

        # Load agent sprites
        for name, sprite_path in agent_sprites.items():
            try:
                pil_image = (
                    Image.open(sprite_path)
                    .resize((self.tile_size, self.tile_size))
                    .convert("RGBA")
                )
                pygame_image = pygame.image.fromstring(
                    pil_image.tobytes(), pil_image.size, pil_image.mode
                )
                self.sprite_cache[name] = pygame_image
            except Exception as e:
                print(f"Warning: Could not load agent sprite {sprite_path}: {e}")
                # Create a placeholder
                self.sprite_cache[name] = pygame.Surface(
                    (self.tile_size, self.tile_size), pygame.SRCALPHA
                )

    def populate_world(self):
        """Populate the world with entities from the ASCII map, exactly like Sorrel
        does."""
        world = self.world

        # Reset spawn points
        world.reset_spawn_points()

        if hasattr(world, "map_generator") and world.map_generator is not None:
            self._populate_from_ascii_map()
        else:
            self._populate_randomly()

    def _populate_from_ascii_map(self):
        """Populate environment using ASCII map layout - PRESERVES ALL ORIGINAL LOGIC."""
        world = self.world
        map_data = world.map_generator.parse_map()

        print(f"ASCII Map loaded: {world.width}x{world.height}")
        print(f"Spawn points: {len(map_data.spawn_points)}")
        print(f"Resource locations: {len(map_data.resource_locations)}")
        print(f"Wall locations: {len(map_data.wall_locations)}")
        print(f"Empty locations: {len(map_data.empty_locations)}")

        # Initialize all layers with default entities first
        for y, x, layer in np.ndindex(world.map.shape):
            index = (y, x, layer)
            world.add(index, Empty())

        # Place walls EXACTLY where map specifies (all layers)
        for y, x in map_data.wall_locations:
            for layer in [world.terrain_layer, world.dynamic_layer, world.beam_layer]:
                world.add((y, x, layer), Wall())

        # Set spawn points EXACTLY where map specifies
        world.agent_spawn_points = [
            (y, x, world.dynamic_layer) for y, x in map_data.spawn_points
        ]

        # Create resource spawn points from map resource locations
        world.resource_spawn_points = [
            (y, x, world.dynamic_layer) for y, x, _ in map_data.resource_locations
        ]

        # Place terrain layer entities (Spawn/Sand) for ALL locations
        for y, x, layer in np.ndindex(world.map.shape):
            if layer == world.terrain_layer:
                terrain_loc = (y, x, layer)
                # Skip if it's a wall (walls are already placed)
                if (y, x) in map_data.wall_locations:
                    continue
                # Place Spawn entity for spawn points
                elif (y, x) in map_data.spawn_points:
                    world.add(terrain_loc, Spawn())
                # Place Sand entity for all other locations
                else:
                    # Use original Sand logic - can_convert_to_resource based on resource locations
                    can_convert = (
                        y,
                        x,
                        world.dynamic_layer,
                    ) in world.resource_spawn_points

                    # Determine resource type for this location
                    resource_type = None
                    if can_convert:
                        # Find the resource type for this location
                        for ry, rx, rtype in map_data.resource_locations:
                            if ry == y and rx == x:
                                resource_type = rtype
                                break

                    world.add(
                        terrain_loc,
                        Sand(
                            can_convert_to_resource=can_convert,
                            respawn_ready=True,
                            resource_type=resource_type,
                        ),
                    )

        # Place resources EXACTLY where map specifies with physical health values
        for y, x, resource_type in map_data.resource_locations:
            dynamic_loc = (y, x, world.dynamic_layer)
            if resource_type == "stag":
                world.add(
                    dynamic_loc,
                    StagResource(world.stag_reward, world.stag_health, regeneration_cooldown=world.stag_regeneration_cooldown),
                )
            elif resource_type == "hare":
                world.add(
                    dynamic_loc,
                    HareResource(world.hare_reward, world.hare_health, regeneration_cooldown=world.hare_regeneration_cooldown),
                )
            elif resource_type == "random":
                # Use stag_probability parameter for random resource type selection
                if np.random.random() < world.stag_probability:
                    world.add(
                        dynamic_loc,
                        StagResource(world.stag_reward, world.stag_health, regeneration_cooldown=world.stag_regeneration_cooldown),
                    )
                else:
                    world.add(
                        dynamic_loc,
                        HareResource(world.hare_reward, world.hare_health, regeneration_cooldown=world.hare_regeneration_cooldown),
                    )

        # Place empty entities on dynamic layer for non-resource, non-spawn locations
        for y, x in map_data.empty_locations:
            dynamic_loc = (y, x, world.dynamic_layer)
            if (
                dynamic_loc not in world.agent_spawn_points
                and dynamic_loc not in world.resource_spawn_points
            ):
                world.add(dynamic_loc, Empty())

        # Initialize beam layer with empty entities (preserve walls)
        for y, x, layer in np.ndindex(world.map.shape):
            if layer == world.beam_layer:
                # Only place Empty if it's not a wall location
                if (y, x) not in map_data.wall_locations:
                    world.add((y, x, layer), Empty())

    def _populate_randomly(self):
        """Populate environment using random generation (fallback)."""
        world = self.world

        for y, x, layer in np.ndindex(world.map.shape):
            index = (y, x, layer)
            if y == 0 or y == world.height - 1 or x == 0 or x == world.width - 1:
                world.add(index, Wall())
            elif layer == world.terrain_layer:
                # interior cells are spawnable and traversable
                if (y, x, world.dynamic_layer) in world.agent_spawn_points:
                    world.add(index, Spawn())
                elif (y, x, world.dynamic_layer) not in world.resource_spawn_points:
                    # Non-resource locations get Sand that cannot convert to resources
                    world.add(
                        index, Sand(can_convert_to_resource=False, respawn_ready=True)
                    )
                else:
                    # Resource spawn points get Sand that can convert to resources
                    world.add(
                        index, Sand(can_convert_to_resource=True, respawn_ready=True)
                    )
            elif layer == world.dynamic_layer:
                # dynamic layer: optionally place initial resources
                if (y, x, world.dynamic_layer) not in world.agent_spawn_points:
                    if np.random.random() < world.resource_density:
                        # choose resource type uniformly at random
                        world.resource_spawn_points.append((y, x, world.dynamic_layer))
                    else:
                        # non-resource locations get Empty entities (attributes inherited from terrain)
                        world.add(index, Empty())
            elif layer == world.beam_layer:
                # beam layer: initially empty (attributes inherited from terrain)
                world.add(index, Empty())

        # randomly populate resources on the dynamic layer according to density
        for y, x, layer in world.resource_spawn_points:
            # dynamic layer coordinates
            dynamic = (y, x, world.dynamic_layer)
            # choose resource type based on stag_probability parameter
            if np.random.random() < world.stag_probability:
                world.add(
                    dynamic, StagResource(world.stag_reward, world.stag_health)
                )
            else:
                world.add(
                    dynamic, HareResource(world.hare_reward, world.hare_health)
                )

    def create_agents(self):
        """Create multiple agents for the game."""
        num_agents = self.config["world"]["num_agents"]
        agents = []

        # Create observation spec
        entity_list = [
            "Empty",
            "Wall",
            "Spawn",
            "StagResource",
            "HareResource",
            "StagHuntAgentNorth",
            "StagHuntAgentEast",
            "StagHuntAgentSouth",
            "StagHuntAgentWest",
            "Sand",
            "InteractionBeam",
            "AttackBeam",
            "PunishBeam",
        ]
        observation_spec = StagHuntObservation(
            entity_list, full_view=False, vision_radius=2
        )

        # Create action spec with ATTACK and PUNISH actions
        action_spec = ActionSpec(
            [
                "NOOP",
                "FORWARD",
                "BACKWARD",
                "STEP_LEFT",
                "STEP_RIGHT",
                "TURN_LEFT",
                "TURN_RIGHT",
                "ATTACK",
                "PUNISH",
            ]
        )

        # Create a dummy model
        class DummyModel:
            def take_action(self, state):
                return 0

            def reset(self):
                pass

            @property
            def memory(self):
                return self

            def current_state(self):
                return np.zeros((4, 278))

            def add(self, state, action, reward, done):
                pass  # Dummy memory add method

        # Create agents
        for i in range(num_agents):
            agent = StagHuntAgent(
                observation_spec=observation_spec,
                action_spec=action_spec,
                model=DummyModel(),
                interaction_reward=self.config["world"]["interaction_reward"],
                max_health=self.config["world"]["agent_health"]
            )

            # Set the agent kind based on orientation
            agent.update_agent_kind()

            # Place agent at spawn point
            if i < len(self.world.agent_spawn_points):
                spawn_location = self.world.agent_spawn_points[i]
                self.world.add(spawn_location, agent)
                print(f"Agent {i+1} placed at spawn point: {spawn_location}")
            else:
                print(f"Warning: No spawn point available for agent {i+1}")

            agents.append(agent)

        return agents

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_keypress(event.key)

    def handle_keypress(self, key):
        """Handle keyboard input for player actions."""
        if not self.agents or not self.human_player:
            return

        # In turn-based mode, only allow actions for the current agent
        if (
            self.turn_based
            and self.human_player != self.agents[self.current_agent_index]
        ):
            return

        # In turn-based mode, only allow one action per turn
        if self.turn_based and self.turn_action_taken:
            return

        # Check if agent can act (not frozen)
        if not self.human_player.can_act():
            # Agent is frozen - advance turn immediately
            if self.turn_based:
                print(
                    f"Agent {self.current_agent_index + 1} is frozen, advancing turn..."
                )
                self.end_turn()
            return

        # Agent switching controls (only in non-turn-based mode or for debugging)
        if key == pygame.K_TAB and not self.turn_based:
            self.switch_agent()
            return

        # Map keys to actions - UPDATED FOR PHYSICAL VERSION
        key_to_action = {
            pygame.K_w: "FORWARD",
            pygame.K_s: "BACKWARD",
            pygame.K_a: "STEP_LEFT",
            pygame.K_d: "STEP_RIGHT",
            pygame.K_q: "TURN_LEFT",
            pygame.K_e: "TURN_RIGHT",
            pygame.K_SPACE: "ATTACK",  # Changed from INTERACT to ATTACK
            pygame.K_p: "PUNISH",      # New PUNISH action
            pygame.K_r: "ATTACK",      # Alternative key for ATTACK
            pygame.K_RETURN: "END_TURN",  # Enter key to end turn manually
            pygame.K_ESCAPE: "QUIT",
        }

        if key in key_to_action:
            action_name = key_to_action[key]
            if action_name == "QUIT":
                self.running = False
            elif action_name == "END_TURN" and self.turn_based:
                self.end_turn()
            else:
                self.execute_action(action_name)

    def switch_agent(self):
        """Switch to the next available agent."""
        if len(self.agents) <= 1:
            return

        # Find next available agent
        start_index = self.current_agent_index
        while True:
            self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)
            if self.agents[self.current_agent_index].can_act():
                break
            if self.current_agent_index == start_index:
                # No available agents
                self.current_agent_index = start_index
                break

        self.human_player = self.agents[self.current_agent_index]
        print(f"Switched to Agent {self.current_agent_index + 1}")

    def handle_turn_based_gameplay(self):
        """Handle turn-based gameplay logic."""
        # Turn timer is now controlled by key presses, not frames
        # This method is kept for compatibility but doesn't do automatic timing
        pass

    def end_turn(self):
        """End the current agent's turn and move to the next agent."""
        if not self.agents:
            return

        # Mark that this agent has acted this turn
        self.agents_acted_this_turn.add(self.current_agent_index)

        # Reset turn timer and action flag
        self.turn_timer = 0
        self.turn_action_taken = False

        # Check if all agents have acted (complete turn)
        if len(self.agents_acted_this_turn) >= len(self.agents):
            self.full_turn_count += 1
            self.agents_acted_this_turn.clear()
            print(f"=== FULL TURN {self.full_turn_count} COMPLETED ===")

        # Move to next agent
        self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)
        self.human_player = self.agents[self.current_agent_index]

        # Check if next agent is frozen
        if not self.human_player.can_act():
            print(
                f"Turn ended. Agent {self.current_agent_index + 1} is frozen, will skip..."
            )
        else:
            print(f"Turn ended. Now controlling Agent {self.current_agent_index + 1}")

    def execute_action(self, action_name):
        """Execute the specified action for the human player."""
        if not self.human_player or not self.human_player.can_act():
            return

        # Get the action index
        action_index = None
        for idx, name in self.human_player.action_spec.actions.items():
            if name == action_name:
                action_index = idx
                break

        if action_index is None:
            print(f"Unknown action: {action_name}")
            return

        # Execute the action and get reward
        reward = self.human_player.act(self.world, action_index)
        self.score += reward

        # Update individual agent score for the current agent
        self.agent_scores[self.current_agent_index] += reward

        # CRITICAL: Update scores for all agents who received pending rewards
        # This ensures both agents get credit for interactions
        for i, agent in enumerate(self.agents):
            if hasattr(agent, "pending_reward") and agent.pending_reward > 0:
                # This agent received a reward from an interaction
                self.agent_scores[i] += agent.pending_reward
                print(
                    f"Agent {i+1} received interaction reward: {agent.pending_reward:.1f}"
                )
                # Reset the pending reward (it will be processed by the agent's act method)
                agent.pending_reward = 0.0

        # Mark that an action has been taken this turn
        if self.turn_based:
            self.turn_action_taken = True
            # In turn-based mode, automatically end turn after any action
            self.end_turn()

        # Update turn count
        self.turn_count += 1

        # Note: World state is updated every frame in the main loop

    def update_world_state(self):
        """Update the world state by processing transitions."""
        # Update agent states (frozen/respawn logic) - CRITICAL for proper game behavior
        if hasattr(self.world, "env") and hasattr(
            self.world.env, "update_agent_states"
        ):
            self.world.env.update_agent_states()

            # CRITICAL: Find and replace removed agents with Empty entities
            # Scan the entire dynamic layer for agents that should be removed
            for y, x, z in np.ndindex(self.world.map.shape):
                if z == self.world.dynamic_layer:  # Only check dynamic layer
                    location = (y, x, z)
                    entity = self.world.observe(location)

                    # Check if this is an agent that should be removed
                    if isinstance(
                        entity, type(self.agents[0]) if self.agents else False
                    ):
                        # Find which agent this is
                        for agent in self.agents:
                            if entity == agent:
                                # Check if this agent should be removed
                                if (
                                    hasattr(agent, "is_removed")
                                    and agent.is_removed
                                    and hasattr(agent, "respawn_timer")
                                    and agent.respawn_timer > 0
                                ):
                                    # Replace with Empty entity
                                    self.world.add(location, Empty())
                                    print(
                                        f"Replaced removed agent at {location} with Empty entity"
                                    )
                                break

            # Debug: Check agent states (only print changes)
            for i, agent in enumerate(self.agents):
                if (
                    hasattr(agent, "is_removed")
                    and agent.is_removed
                    and not hasattr(agent, "_debug_removed_printed")
                ):
                    print(
                        f"Agent {i+1} is REMOVED (respawn timer: {getattr(agent, 'respawn_timer', 'N/A')})"
                    )
                    agent._debug_removed_printed = True
                elif not hasattr(agent, "is_removed") or not agent.is_removed:
                    if hasattr(agent, "_debug_removed_printed"):
                        delattr(agent, "_debug_removed_printed")

        # Process transitions for all entities except agents (they're handled by environment)
        for y, x, z in np.ndindex(self.world.map.shape):
            location = (y, x, z)
            entity = self.world.observe(location)
            if hasattr(entity, "transition") and not isinstance(
                entity, type(self.agents[0]) if self.agents else False
            ):
                entity.transition(self.world)

    def draw_health_bar(self, x, y, current_health, max_health, width=40, height=6):
        """Draw a health bar."""
        screen_x = x * self.tile_size + (self.tile_size - width) // 2
        screen_y = y * self.tile_size - 10

        # Background (red)
        pygame.draw.rect(self.screen, (255, 0, 0), 
                        (screen_x, screen_y, width, height))
        
        # Health (green)
        health_width = int(width * current_health / max_health)
        if health_width > 0:
            pygame.draw.rect(self.screen, (0, 255, 0), 
                           (screen_x, screen_y, health_width, height))

    def draw(self):
        """Draw the game world using exact same rendering as Sorrel."""
        self.screen.fill((0, 0, 0))  # Black background

        # Render layers exactly like Sorrel does
        self.draw_terrain_layer()
        self.draw_dynamic_layer()
        self.draw_beam_layer()

        # Draw UI
        self.draw_ui()

        pygame.display.flip()

    def draw_terrain_layer(self):
        """Draw the terrain layer using exact sprites."""
        for y in range(self.world.height):
            for x in range(self.world.width):
                location = (y, x, 0)
                entity = self.world.observe(location)

                # Get sprite based on entity type
                if isinstance(entity, Wall):
                    sprite_key = "wall"
                elif isinstance(entity, Sand):
                    sprite_key = "sand"
                elif isinstance(entity, Spawn):
                    sprite_key = "spawn"
                else:
                    sprite_key = "empty"

                # Draw sprite
                if sprite_key in self.sprite_cache:
                    sprite = self.sprite_cache[sprite_key]
                    self.screen.blit(sprite, (x * self.tile_size, y * self.tile_size))
                else:
                    # Fallback: draw colored rectangles
                    rect = pygame.Rect(
                        x * self.tile_size,
                        y * self.tile_size,
                        self.tile_size,
                        self.tile_size,
                    )
                    if isinstance(entity, Wall):
                        pygame.draw.rect(self.screen, (100, 100, 100), rect)
                    elif isinstance(entity, Sand):
                        pygame.draw.rect(self.screen, (194, 178, 128), rect)
                    elif isinstance(entity, Spawn):
                        pygame.draw.rect(self.screen, (255, 255, 0), rect)
                    else:
                        pygame.draw.rect(self.screen, (200, 200, 200), rect)

    def draw_dynamic_layer(self):
        """Draw the dynamic layer using exact sprites."""
        for y in range(self.world.height):
            for x in range(self.world.width):
                location = (y, x, 1)
                entity = self.world.observe(location)

                # Get sprite based on entity type
                if isinstance(entity, StagResource):
                    sprite_key = "stag"
                elif isinstance(entity, HareResource):
                    sprite_key = "hare"
                elif isinstance(entity, StagHuntAgent):
                    # Only draw agent if it's not removed
                    if hasattr(entity, "is_removed") and entity.is_removed:
                        continue  # Skip removed agents
                    # Use orientation-specific sprite
                    orientation_names = {0: "north", 1: "east", 2: "south", 3: "west"}
                    sprite_key = f"agent_{orientation_names[entity.orientation]}"
                else:
                    continue  # Skip empty entities on dynamic layer

                # Draw sprite with alpha blending
                if sprite_key in self.sprite_cache:
                    sprite = self.sprite_cache[sprite_key]
                    # Create a surface for alpha blending
                    sprite_surface = pygame.Surface(
                        (self.tile_size, self.tile_size), pygame.SRCALPHA
                    )
                    sprite_surface.blit(sprite, (0, 0))
                    self.screen.blit(
                        sprite_surface, (x * self.tile_size, y * self.tile_size)
                    )
                else:
                    # Fallback: draw colored rectangles
                    rect = pygame.Rect(
                        x * self.tile_size,
                        y * self.tile_size,
                        self.tile_size,
                        self.tile_size,
                    )
                    if isinstance(entity, StagResource):
                        pygame.draw.circle(
                            self.screen, (255, 0, 0), rect.center, self.tile_size // 3
                        )
                    elif isinstance(entity, HareResource):
                        pygame.draw.rect(self.screen, (0, 255, 0), rect)
                    elif isinstance(entity, StagHuntAgent):
                        # Only draw agent if it's not removed
                        if hasattr(entity, "is_removed") and entity.is_removed:
                            continue  # Skip removed agents
                        # Draw agent as a larger, more visible circle
                        pygame.draw.circle(
                            self.screen, (0, 0, 255), rect.center, self.tile_size // 2
                        )
                        # Draw orientation indicator
                        orientation_vectors = {
                            0: (0, -self.tile_size // 3),  # North (up)
                            1: (self.tile_size // 3, 0),  # East (right)
                            2: (0, self.tile_size // 3),  # South (down)
                            3: (-self.tile_size // 3, 0),  # West (left)
                        }
                        dx, dy = orientation_vectors[entity.orientation]
                        arrow_end = (rect.centerx + dx, rect.centery + dy)
                        pygame.draw.line(
                            self.screen, (255, 255, 255), rect.center, arrow_end, 3
                        )

                # Draw health bars for agents and resources if enabled
                if self.config["display"].get("show_health", True):
                    if isinstance(entity, StagHuntAgent) and not entity.is_removed:
                        self.draw_health_bar(x, y, entity.health, entity.max_health)
                    elif isinstance(entity, (StagResource, HareResource)):
                        self.draw_health_bar(x, y, entity.health, entity.max_health)

    def draw_beam_layer(self):
        """Draw the beam layer using exact sprites."""
        for y in range(self.world.height):
            for x in range(self.world.width):
                location = (y, x, 2)
                entity = self.world.observe(location)

                if isinstance(entity, InteractionBeam):
                    if "beam" in self.sprite_cache:
                        sprite = self.sprite_cache["beam"]
                        # Draw with transparency
                        sprite_surface = pygame.Surface(
                            (self.tile_size, self.tile_size), pygame.SRCALPHA
                        )
                        sprite_surface.blit(sprite, (0, 0))
                        self.screen.blit(
                            sprite_surface, (x * self.tile_size, y * self.tile_size)
                        )
                elif isinstance(entity, AttackBeam):
                    if "attack_beam" in self.sprite_cache:
                        sprite = self.sprite_cache["attack_beam"]
                        sprite_surface = pygame.Surface(
                            (self.tile_size, self.tile_size), pygame.SRCALPHA
                        )
                        sprite_surface.blit(sprite, (0, 0))
                        self.screen.blit(
                            sprite_surface, (x * self.tile_size, y * self.tile_size)
                        )
                elif isinstance(entity, PunishBeam):
                    if "punish_beam" in self.sprite_cache:
                        sprite = self.sprite_cache["punish_beam"]
                        sprite_surface = pygame.Surface(
                            (self.tile_size, self.tile_size), pygame.SRCALPHA
                        )
                        sprite_surface.blit(sprite, (0, 0))
                        self.screen.blit(
                            sprite_surface, (x * self.tile_size, y * self.tile_size)
                        )

    def draw_ui(self):
        """Draw the user interface."""
        ui_y = self.world.height * self.tile_size

        # Background for UI - make it larger to accommodate all scores and turn display
        ui_height = 250 if self.turn_based and len(self.agents) > 1 else 150
        ui_rect = pygame.Rect(0, ui_y, self.screen_width, ui_height)
        pygame.draw.rect(self.screen, (30, 30, 30), ui_rect)

        # Score display
        if self.turn_based and len(self.agents) > 1:
            # Show individual agent scores
            total_score = sum(self.agent_scores)
            total_text = self.font_large.render(
                f"Total Score: {total_score:.1f}", True, (255, 255, 255)
            )
            self.screen.blit(total_text, (10, ui_y + 10))

            # Show individual agent scores in a grid layout
            scores_per_row = 3  # Show 3 scores per row for better visibility
            for i, score in enumerate(self.agent_scores):
                agent_color = (
                    (255, 255, 0) if i == self.current_agent_index else (200, 200, 200)
                )
                agent_score_text = self.font_medium.render(
                    f"Agent {i+1}: {score:.1f}", True, agent_color
                )

                # Calculate position in grid
                row = i // scores_per_row
                col = i % scores_per_row
                x_pos = 10 + col * 120  # 120 pixels between columns
                y_pos = ui_y + 40 + row * 25

                self.screen.blit(agent_score_text, (x_pos, y_pos))
        else:
            # Single agent mode - show regular score
            score_text = self.font_large.render(
                f"Score: {self.score:.1f}", True, (255, 255, 255)
            )
            self.screen.blit(score_text, (10, ui_y + 10))

        turn_y = ui_y + 10
        # Turn count - make it VERY prominent and always visible
        if self.turn_based and len(self.agents) > 1:
            # Draw turn count at the top right for maximum visibility
            turn_text = self.font_large.render(
                f"TURN {self.full_turn_count}", True, (255, 255, 0)
            )  # Bright yellow
            action_text = self.font_medium.render(
                f"Actions: {self.turn_count}", True, (200, 200, 200)
            )

            # Position at top right of screen
            turn_rect = turn_text.get_rect()
            turn_x = self.screen_width - turn_rect.width - 10

            # Draw background for turn count
            bg_rect = pygame.Rect(
                turn_x - 5, turn_y - 5, turn_rect.width + 10, turn_rect.height + 10
            )
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)  # Black background
            pygame.draw.rect(self.screen, (255, 255, 0), bg_rect, 2)  # Yellow border

            self.screen.blit(turn_text, (turn_x, turn_y))
            self.screen.blit(action_text, (turn_x, turn_y + 30))
        else:
            turn_text = self.font_large.render(
                f"Turn: {self.turn_count}", True, (255, 255, 0)
            )
            self.screen.blit(turn_text, (10, ui_y + 10))

        # Agent info and inventory
        if self.human_player:
            inv_stag = self.human_player.inventory.get("stag", 0)
            inv_hare = self.human_player.inventory.get("hare", 0)
            ready = "Ready" if self.human_player.ready else "Not Ready"

            # Get agent position
            agent_pos = self.human_player.location
            orientation_names = {0: "North", 1: "East", 2: "South", 3: "West"}
            orientation_name = orientation_names.get(
                self.human_player.orientation, "Unknown"
            )

            # Calculate agent info Y position based on score display
            agent_info_y = turn_y + 60  # More space for turn count display

            # Show current agent
            agent_text = self.font_medium.render(
                f"Agent {self.current_agent_index + 1}/{len(self.agents)}",
                True,
                (255, 255, 255),
            )
            self.screen.blit(agent_text, (10, agent_info_y))

            # Show health info
            health_text = self.font_medium.render(
                f"Health: {self.human_player.health}/{self.human_player.max_health}",
                True,
                (255, 255, 255),
            )
            self.screen.blit(health_text, (10, agent_info_y + 25))

            # Show turn information
            if self.turn_based:
                # Show action status
                if (
                    hasattr(self.human_player, "is_removed")
                    and self.human_player.is_removed
                ):
                    action_status = "REMOVED - Respawn in progress"
                    action_color = (255, 0, 0)
                elif (
                    hasattr(self.human_player, "is_frozen")
                    and self.human_player.is_frozen
                ):
                    action_status = "FROZEN - Press any key to continue"
                    action_color = (255, 100, 100)
                elif self.turn_action_taken:
                    action_status = "Action Taken - Turn will end"
                    action_color = (255, 0, 0)
                else:
                    action_status = "Press any key to take action"
                    action_color = (0, 255, 0)
                action_text = self.font_medium.render(action_status, True, action_color)
                self.screen.blit(action_text, (10, agent_info_y + 50))

                inv_text = self.font_medium.render(
                    f"Stag: {inv_stag} | Hare: {inv_hare} | {ready}",
                    True,
                    (255, 255, 255),
                )
                self.screen.blit(inv_text, (10, agent_info_y + 75))

                pos_text = self.font_medium.render(
                    f"Position: ({agent_pos[1]}, {agent_pos[0]}) | Facing: {orientation_name}",
                    True,
                    (255, 255, 255),
                )
                self.screen.blit(pos_text, (10, agent_info_y + 100))
            else:
                inv_text = self.font_medium.render(
                    f"Stag: {inv_stag} | Hare: {inv_hare} | {ready}",
                    True,
                    (255, 255, 255),
                )
                self.screen.blit(inv_text, (10, agent_info_y + 50))

                pos_text = self.font_medium.render(
                    f"Position: ({agent_pos[1]}, {agent_pos[0]}) | Facing: {orientation_name}",
                    True,
                    (255, 255, 255),
                )
                self.screen.blit(pos_text, (10, agent_info_y + 75))

        # Physical Version Info
        map_text = self.font_small.render(
            "Stag Hunt Physical - ASCII Map with Health System", True, (200, 200, 200)
        )
        if self.turn_based and self.human_player:
            map_y = agent_info_y + 130
        else:
            map_y = ui_y + 140
        self.screen.blit(map_text, (10, map_y))

    def run(self):
        """Main game loop."""
        print("Stag Hunt Physical ASCII Pygame Started!")
        print("Using exact same sprites and ASCII map as Sorrel framework!")
        print("Enhanced with ATTACK/PUNISH actions and health system!")
        print(f"Number of agents: {len(self.agents)}")
        print("Controls:")
        print("  W/S: Move forward/backward")
        print("  A/D: Step left/right")
        print("  Q/E: Turn left/right")
        print("  SPACE/R: Attack (red beam)")
        print("  P: Punish (yellow beam)")
        if self.turn_based:
            print("  ENTER: End turn manually")
            print("  (Turns end automatically after any action)")
        elif len(self.agents) > 1:
            print("  TAB: Switch between agents")
        print("  ESC: Quit")
        print()

        while self.running:
            self.handle_events()

            # Handle turn-based gameplay
            if self.turn_based:
                self.handle_turn_based_gameplay()

            # Update agent states every frame (critical for frozen/respawn logic)
            self.update_world_state()

            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()
        if self.turn_based and len(self.agents) > 1:
            total_score = sum(self.agent_scores)
            print(f"Game Over! Final Scores:")
            for i, score in enumerate(self.agent_scores):
                print(f"  Agent {i+1}: {score:.1f}")
            print(f"  Total Score: {total_score:.1f}")
        else:
            print(f"Game Over! Final Score: {self.score}")


def main():
    """Main function to run the Physical ASCII pygame version."""
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Stag Hunt Physical ASCII Pygame")
    parser.add_argument(
        "--agents",
        "-a",
        type=int,
        default=1,
        help="Number of agents (default: 1, max: 4)",
    )
    parser.add_argument(
        "--tile-size",
        "-t",
        type=int,
        default=32,
        help="Tile size in pixels (default: 32)",
    )
    parser.add_argument(
        "--fps", "-f", type=int, default=10, help="Game speed in FPS (default: 10)"
    )
    parser.add_argument(
        "--turn-duration",
        "-d",
        type=int,
        default=30,
        help="Turn duration in frames (default: 30)",
    )

    args = parser.parse_args()

    # Validate number of agents
    num_agents = max(1, min(4, args.agents))  # Clamp between 1 and 4
    if args.agents != num_agents:
        print(f"Warning: Number of agents clamped to {num_agents} (max 4)")

    print(f"Starting Stag Hunt Physical with {num_agents} agent(s)")

    # Configuration using ASCII map with physical enhancements
    config = {
        "world": {
            "generation_mode": "ascii_map",
            "ascii_map_file": "docs/stag_hunt_ascii_map_clean.txt",
            "num_agents": num_agents,
            "stag_reward": 1.0,  # Higher reward for stag (requires coordination)
            "hare_reward": 0.1,  # Lower reward for hare (solo achievable)
            "taste_reward": 0.1,  # Legacy parameter
            "stag_health": 12,
            "hare_health": 3,
            "agent_health": 5,
            "health_regeneration_rate": 0.1,
            "reward_sharing_radius": 3,
            "beam_length": 3,
            "beam_radius": 1,
            "attack_cooldown": 3,
            "punish_cooldown": 5,
            "beam_cooldown": 3,  # Legacy parameter
            "respawn_lag": 10,
            "payoff_matrix": [[4, 0], [2, 2]],
            "interaction_reward": 1.0,
            "freeze_duration": 5,
            "respawn_delay": 10,
        },
        "display": {
            "tile_size": args.tile_size,
            "fps": args.fps,
            "turn_duration": args.turn_duration,
            "show_health": True,
        },
    }

    # Create and run the game
    game = StagHuntPhysicalASCIIPygame(config)
    game.run()


if __name__ == "__main__":
    main()
