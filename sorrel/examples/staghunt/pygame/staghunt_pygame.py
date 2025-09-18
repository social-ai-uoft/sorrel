"""
Pygame implementation of the Stag Hunt game.
Allows human players to play the stag hunt game using keyboard controls.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path to import sorrel modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

import pygame
import numpy as np
from sorrel.examples.staghunt.entities import Empty, Sand, Wall, Spawn, StagResource, HareResource, InteractionBeam
from sorrel.examples.staghunt.world import StagHuntWorld
from sorrel.examples.staghunt.agents_v2 import StagHuntAgent
from sorrel.action.action_spec import ActionSpec
from sorrel.examples.staghunt.agents_v2 import StagHuntObservation


class StagHuntPygame:
    """Main pygame class for the Stag Hunt game."""
    
    def __init__(self, config=None):
        # Initialize pygame
        pygame.init()
        
        # Default configuration
        if config is None:
            config = {
                "world": {
                    "height": 11,
                    "width": 11,
                    "num_agents": 1,
                    "resource_density": 0.15,
                    "taste_reward": 0.1,
                    "destroyable_health": 3,
                    "beam_length": 3,
                    "beam_radius": 1,
                    "beam_cooldown": 3,
                    "respawn_lag": 10,
                    "payoff_matrix": [[4, 0], [2, 2]],
                    "interaction_reward": 1.0,
                    "freeze_duration": 5,
                    "respawn_delay": 10,
                },
                "display": {
                    "cell_size": 50,
                    "fps": 10,
                }
            }
        
        self.config = config
        self.cell_size = config["display"]["cell_size"]
        self.fps = config["display"]["fps"]
        
        # Calculate screen dimensions
        self.screen_width = config["world"]["width"] * self.cell_size
        self.screen_height = config["world"]["height"] * self.cell_size + 100  # Extra space for UI
        
        # Create screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Stag Hunt Pygame")
        
        # Colors
        self.colors = {
            'empty': (200, 200, 200),
            'wall': (100, 100, 100),
            'sand': (194, 178, 128),
            'spawn': (255, 255, 0),
            'stag': (255, 0, 0),      # Red for stag
            'hare': (0, 255, 0),      # Green for hare
            'agent': (0, 0, 255),     # Blue for agent
            'beam': (255, 255, 255),  # White for beam
            'background': (50, 50, 50),
            'ui_bg': (30, 30, 30),
            'text': (255, 255, 255)
        }
        
        # Initialize game world
        self.world = StagHuntWorld(config, Empty())
        self.setup_world()
        
        # Initialize human player
        self.human_player = self.create_human_player()
        
        # Game state
        self.running = True
        self.clock = pygame.time.Clock()
        self.score = 0
        self.turn_count = 0
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
    def setup_world(self):
        """Set up the initial world state."""
        # Add walls around the border
        for y in range(self.world.height):
            for x in range(self.world.width):
                if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                    # Add walls around the edge
                    self.world.add((y, x, 0), Wall())
                    self.world.add((y, x, 1), Wall())
                    self.world.add((y, x, 2), Wall())
                else:
                    # Add sand on the terrain layer
                    self.world.add((y, x, 0), Sand(can_convert_to_resource=True, respawn_ready=True))
                    # Add empty space on the dynamic layer
                    self.world.add((y, x, 1), Empty())
                    # Add empty space on the beam layer
                    self.world.add((y, x, 2), Empty())
    
    def create_human_player(self):
        """Create a human player agent."""
        # Create observation spec
        entity_list = [
            "Empty", "Wall", "Spawn", "StagResource", "HareResource",
            "StagHuntAgentNorth", "StagHuntAgentEast", "StagHuntAgentSouth", "StagHuntAgentWest",
            "Sand", "InteractionBeam"
        ]
        observation_spec = StagHuntObservation(
            entity_list,
            full_view=False,
            vision_radius=2
        )
        
        # Create action spec
        action_spec = ActionSpec([
            "NOOP", "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT", 
            "TURN_LEFT", "TURN_RIGHT", "INTERACT"
        ])
        
        # Create a dummy model (not used for human player)
        class DummyModel:
            def take_action(self, state):
                return 0
            def reset(self):
                pass
            @property
            def memory(self):
                return self
            def current_state(self):
                # Return dummy state with correct dimensions
                # The observation spec will determine the actual size
                return np.zeros((4, 278))  # n_frames=5, but we return 4 for stacking
        
        # Create the agent
        agent = StagHuntAgent(
            observation_spec=observation_spec,
            action_spec=action_spec,
            model=DummyModel(),
            interaction_reward=self.config["world"]["interaction_reward"]
        )
        
        # Set the agent kind based on orientation
        agent.update_agent_kind()
        
        # Place agent in the center
        center_y = self.world.height // 2
        center_x = self.world.width // 2
        agent_location = (center_y, center_x, 1)
        self.world.add(agent_location, agent)
        
        return agent
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_keypress(event.key)
    
    def handle_keypress(self, key):
        """Handle keyboard input for player actions."""
        if not self.human_player or not self.human_player.can_act():
            return
        
        # Map keys to actions
        key_to_action = {
            pygame.K_w: "FORWARD",
            pygame.K_s: "BACKWARD",
            pygame.K_a: "STEP_LEFT",
            pygame.K_d: "STEP_RIGHT",
            pygame.K_q: "TURN_LEFT",
            pygame.K_e: "TURN_RIGHT",
            pygame.K_SPACE: "INTERACT",
            pygame.K_ESCAPE: "QUIT"
        }
        
        if key in key_to_action:
            action_name = key_to_action[key]
            if action_name == "QUIT":
                self.running = False
            else:
                self.execute_action(action_name)
    
    def execute_action(self, action_name):
        """Execute the specified action for the human player."""
        if not self.human_player or not self.human_player.can_act():
            return
        
        # Get the action index by finding the key for the action name
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
        
        # Update turn count
        self.turn_count += 1
        
        # Update world state (handle transitions)
        self.update_world_state()
    
    def update_world_state(self):
        """Update the world state by processing transitions."""
        # Process transitions for all entities except the human player
        for y, x, z in np.ndindex(self.world.map.shape):
            location = (y, x, z)
            entity = self.world.observe(location)
            if hasattr(entity, 'transition') and entity != self.human_player:
                entity.transition(self.world)
    
    def draw(self):
        """Draw the game world."""
        self.screen.fill(self.colors['background'])
        
        # Draw the world layers
        self.draw_terrain_layer()
        self.draw_dynamic_layer()
        self.draw_beam_layer()
        
        # Draw UI
        self.draw_ui()
        
        pygame.display.flip()
    
    def draw_terrain_layer(self):
        """Draw the terrain layer (sand, walls, spawn points)."""
        for y in range(self.world.height):
            for x in range(self.world.width):
                location = (y, x, 0)
                entity = self.world.observe(location)
                
                if isinstance(entity, Wall):
                    color = self.colors['wall']
                elif isinstance(entity, Sand):
                    color = self.colors['sand']
                elif isinstance(entity, Spawn):
                    color = self.colors['spawn']
                else:
                    color = self.colors['empty']
                
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
    
    def draw_dynamic_layer(self):
        """Draw the dynamic layer (resources, agents)."""
        for y in range(self.world.height):
            for x in range(self.world.width):
                location = (y, x, 1)
                entity = self.world.observe(location)
                
                if isinstance(entity, StagResource):
                    color = self.colors['stag']
                    # Draw a circle for stag
                    center = (x * self.cell_size + self.cell_size // 2, 
                             y * self.cell_size + self.cell_size // 2)
                    pygame.draw.circle(self.screen, color, center, self.cell_size // 3)
                elif isinstance(entity, HareResource):
                    color = self.colors['hare']
                    # Draw a square for hare
                    rect = pygame.Rect(x * self.cell_size + self.cell_size // 4, 
                                     y * self.cell_size + self.cell_size // 4,
                                     self.cell_size // 2, self.cell_size // 2)
                    pygame.draw.rect(self.screen, color, rect)
                elif isinstance(entity, StagHuntAgent):
                    # Draw agent based on orientation
                    self.draw_agent(x, y, entity.orientation)
    
    def draw_beam_layer(self):
        """Draw the beam layer (interaction beams)."""
        for y in range(self.world.height):
            for x in range(self.world.width):
                location = (y, x, 2)
                entity = self.world.observe(location)
                
                if isinstance(entity, InteractionBeam):
                    # Draw beam as a bright overlay
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                     self.cell_size, self.cell_size)
                    # Semi-transparent white overlay
                    beam_surface = pygame.Surface((self.cell_size, self.cell_size))
                    beam_surface.set_alpha(128)
                    beam_surface.fill(self.colors['beam'])
                    self.screen.blit(beam_surface, (x * self.cell_size, y * self.cell_size))
    
    def draw_agent(self, x, y, orientation):
        """Draw the agent with orientation indicator."""
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        
        # Draw agent body
        pygame.draw.circle(self.screen, self.colors['agent'], (center_x, center_y), self.cell_size // 3)
        
        # Draw orientation indicator (arrow)
        arrow_length = self.cell_size // 4
        orientation_vectors = {
            0: (0, -arrow_length),  # North (up)
            1: (arrow_length, 0),   # East (right)
            2: (0, arrow_length),   # South (down)
            3: (-arrow_length, 0)   # West (left)
        }
        
        dx, dy = orientation_vectors[orientation]
        arrow_end = (center_x + dx, center_y + dy)
        pygame.draw.line(self.screen, self.colors['text'], (center_x, center_y), arrow_end, 3)
    
    def draw_ui(self):
        """Draw the user interface."""
        ui_y = self.world.height * self.cell_size
        
        # Background for UI
        ui_rect = pygame.Rect(0, ui_y, self.screen_width, 100)
        pygame.draw.rect(self.screen, self.colors['ui_bg'], ui_rect)
        
        # Score
        score_text = self.font_large.render(f"Score: {self.score:.1f}", True, self.colors['text'])
        self.screen.blit(score_text, (10, ui_y + 10))
        
        # Turn count
        turn_text = self.font_medium.render(f"Turn: {self.turn_count}", True, self.colors['text'])
        self.screen.blit(turn_text, (10, ui_y + 40))
        
        # Inventory
        if self.human_player:
            inv_stag = self.human_player.inventory.get("stag", 0)
            inv_hare = self.human_player.inventory.get("hare", 0)
            ready = "Ready" if self.human_player.ready else "Not Ready"
            
            inv_text = self.font_medium.render(f"Stag: {inv_stag} | Hare: {inv_hare} | {ready}", True, self.colors['text'])
            self.screen.blit(inv_text, (10, ui_y + 65))
        
        # Controls
        controls = [
            "W: Forward | S: Backward | A: Step Left | D: Step Right",
            "Q: Turn Left | E: Turn Right | SPACE: Interact | ESC: Quit"
        ]
        
        for i, control in enumerate(controls):
            control_text = self.font_small.render(control, True, self.colors['text'])
            self.screen.blit(control_text, (200, ui_y + 10 + i * 20))
    
    def run(self):
        """Main game loop."""
        print("Stag Hunt Pygame Started!")
        print("Collect stag and hare resources, then interact with other agents!")
        print("Controls:")
        print("  W/S: Move forward/backward")
        print("  A/D: Step left/right")
        print("  Q/E: Turn left/right")
        print("  SPACE: Interact (fire beam)")
        print("  ESC: Quit")
        print()
        
        # Spawn initial resources
        self.spawn_initial_resources()
        
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(self.fps)
        
        pygame.quit()
        print(f"Game Over! Final Score: {self.score}")
    
    def spawn_initial_resources(self):
        """Spawn initial resources on the map."""
        num_resources = 5
        spawned = 0
        
        while spawned < num_resources:
            y = np.random.randint(1, self.world.height - 1)
            x = np.random.randint(1, self.world.width - 1)
            location = (y, x, 1)
            
            # Check if location is empty
            entity = self.world.observe(location)
            if isinstance(entity, Empty):
                # Randomly choose stag or hare
                if np.random.random() < 0.3:  # 30% chance for stag
                    resource = StagResource(self.config["world"]["taste_reward"], 
                                          self.config["world"]["destroyable_health"])
                else:  # 70% chance for hare
                    resource = HareResource(self.config["world"]["taste_reward"], 
                                          self.config["world"]["destroyable_health"])
                
                self.world.add(location, resource)
                spawned += 1


def main():
    """Main function to run the pygame version."""
    # Configuration
    config = {
        "world": {
            "height": 11,
            "width": 11,
            "num_agents": 1,
            "resource_density": 0.15,
            "taste_reward": 0.1,
            "destroyable_health": 3,
            "beam_length": 3,
            "beam_radius": 1,
            "beam_cooldown": 3,
            "respawn_lag": 10,
            "payoff_matrix": [[4, 0], [2, 2]],
            "interaction_reward": 1.0,
            "freeze_duration": 5,
            "respawn_delay": 10,
        },
        "display": {
            "cell_size": 50,
            "fps": 10,
        }
    }
    
    # Create and run the game
    game = StagHuntPygame(config)
    game.run()


if __name__ == "__main__":
    main()
