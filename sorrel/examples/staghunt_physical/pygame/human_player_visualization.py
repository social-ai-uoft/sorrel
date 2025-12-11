#!/usr/bin/env python3
"""
Human Player Visualization - A pygame interface for testing the staghunt_physical game.
This script provides a pure visualization interface similar to the human_player_test.ipynb notebook.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pygame
from omegaconf import OmegaConf

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from sorrel.action.action_spec import ActionSpec
from sorrel.examples.staghunt_physical.agents_v2 import (
    StagHuntAgent,
    StagHuntObservation,
)
from sorrel.examples.staghunt_physical.entities import Empty, entity_list
from sorrel.examples.staghunt_physical.env import StagHuntEnv
from sorrel.examples.staghunt_physical.world import StagHuntWorld
from sorrel.models.human_player import HumanObservation, HumanPlayer


class StagHuntHumanVisualization:
    """A pygame-based visualization interface for human player testing.

    Similar to human_player_test.ipynb but as a standalone pygame application.
    """

    def __init__(self, config_path="configs/config_ascii_map.yaml"):
        """Initialize the visualization with the given config."""
        pygame.init()

        # Load configuration
        self.config = OmegaConf.load(config_path)

        # Create world and environment
        self.world = StagHuntWorld(config=self.config, default_entity=Empty())
        self.env = StagHuntEnv(self.world, self.config)

        # Display settings
        self.tile_size = 32
        self.fps = 10
        self.clock = pygame.time.Clock()

        # Calculate display size
        self.display_width = self.world.width * self.tile_size
        self.display_height = self.world.height * self.tile_size

        # Create display
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("Stag Hunt Physical - Human Player Visualization")

        # Colors
        self.colors = {
            "background": (50, 50, 50),
            "wall": (100, 100, 100),
            "sand": (194, 178, 128),
            "spawn": (0, 255, 0),
            "stag": (139, 69, 19),
            "hare": (255, 165, 0),
            "agent_north": (0, 0, 255),
            "agent_east": (255, 0, 0),
            "agent_south": (0, 255, 0),
            "agent_west": (255, 255, 0),
            "empty": (0, 0, 0),
            "attack_beam": (255, 0, 0),
            "punish_beam": (255, 255, 0),
            "text": (255, 255, 255),
        }

        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)

        # Game state
        self.running = True
        self.current_turn = 0
        self.turn_action_taken = False

        # Create human player agent
        self.setup_human_player()

        # Game info
        self.game_info = {
            "turn": 0,
            "agent_health": 5,
            "attack_cooldown": 0,
            "punish_cooldown": 0,
            "total_reward": 0.0,
        }

    def setup_human_player(self):
        """Setup the human player agent similar to the notebook."""
        # Create observation spec
        observation_spec = HumanObservation(
            entity_list=entity_list,
            full_view=True,
            env_dims=(self.world.height, self.world.width),
        )

        # Create action spec
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

        # Create custom human player
        class StagHuntHumanPlayer(HumanPlayer):
            def __init__(self, input_size, action_space, memory_size):
                super().__init__(input_size, action_space, memory_size)
                self.visual_size = (
                    input_size[0]
                    * input_size[1]
                    * input_size[2]
                    * (self.tile_size**2)
                    * self.num_channels
                )
                self.total_input_size = self.visual_size + 3

            def take_action(self, state: np.ndarray):
                """Custom take_action for pygame interface."""
                # This will be handled by pygame events
                return 0  # Default to NOOP

        # Create human player
        self.human_player = StagHuntHumanPlayer(
            input_size=(self.world.height, self.world.width, 3),
            action_space=action_spec.n_actions,
            memory_size=1,
        )

        # Create custom agent
        class StagHuntHumanAgent(StagHuntAgent):
            def get_action(self, state: np.ndarray) -> int:
                return self.model.take_action(state)

            def add_memory(
                self, state: np.ndarray, action: int, reward: float, done: bool
            ) -> None:
                pass

            def can_act(self) -> bool:
                return True

        # Create the agent
        self.agent = StagHuntHumanAgent(
            observation_spec=observation_spec,
            action_spec=action_spec,
            model=self.human_player,
        )

        # Override agents in environment
        self.env.override_agents([self.agent])

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_keypress(event.key)

    def handle_keypress(self, key):
        """Handle keyboard input for player actions."""
        if self.turn_action_taken:
            return

        # Map keys to actions
        key_to_action = {
            pygame.K_w: 1,  # FORWARD
            pygame.K_s: 2,  # BACKWARD
            pygame.K_a: 3,  # STEP_LEFT
            pygame.K_d: 4,  # STEP_RIGHT
            pygame.K_q: 5,  # TURN_LEFT
            pygame.K_e: 6,  # TURN_RIGHT
            pygame.K_r: 7,  # ATTACK
            pygame.K_p: 8,  # PUNISH
            pygame.K_SPACE: 0,  # NOOP
            pygame.K_RETURN: 9,  # END_TURN
        }

        if key in key_to_action:
            action = key_to_action[key]
            if action == 9:  # END_TURN
                self.end_turn()
            else:
                self.perform_action(action)

    def perform_action(self, action):
        """Perform the given action."""
        if not self.agent.can_act():
            return

        # Get current state
        state = self.get_current_state()

        # Perform action
        reward = self.agent.act(self.world, action)

        # Update game info
        self.game_info["total_reward"] += reward
        self.game_info["turn"] = self.world.current_turn
        self.game_info["agent_health"] = getattr(self.agent, "health", 5)
        self.game_info["attack_cooldown"] = getattr(
            self.agent, "attack_cooldown_timer", 0
        )
        self.game_info["punish_cooldown"] = getattr(
            self.agent, "punish_cooldown_timer", 0
        )

        self.turn_action_taken = True

        # Print action result
        action_names = [
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
        print(f"Action: {action_names[action]}, Reward: {reward:.2f}")

    def end_turn(self):
        """End the current turn and advance the game."""
        self.world.current_turn += 1
        self.turn_action_taken = False

        # Update agent cooldowns
        if hasattr(self.agent, "update_cooldown"):
            self.agent.update_cooldown()

        print(f"Turn {self.world.current_turn} completed")

    def get_current_state(self):
        """Get the current game state for the agent."""
        # This is a simplified state - in practice, you'd use the proper observation system
        return np.zeros((self.world.height, self.world.width, 3))

    def draw_entity(self, entity, x, y):
        """Draw an entity at the given screen coordinates."""
        if entity is None:
            return

        entity_type = type(entity).__name__

        # Draw based on entity type
        if entity_type == "Wall":
            color = self.colors["wall"]
        elif entity_type == "Sand":
            color = self.colors["sand"]
        elif entity_type == "Spawn":
            color = self.colors["spawn"]
        elif entity_type == "StagResource":
            color = self.colors["stag"]
        elif entity_type == "HareResource":
            color = self.colors["hare"]
        elif entity_type == "Empty":
            color = self.colors["empty"]
        elif "StagHuntAgent" in entity_type:
            # Color based on orientation
            if "North" in entity_type:
                color = self.colors["agent_north"]
            elif "East" in entity_type:
                color = self.colors["agent_east"]
            elif "South" in entity_type:
                color = self.colors["agent_south"]
            elif "West" in entity_type:
                color = self.colors["agent_west"]
            else:
                color = self.colors["agent_north"]
        else:
            color = self.colors["empty"]

        # Draw entity rectangle
        pygame.draw.rect(self.screen, color, (x, y, self.tile_size, self.tile_size))
        pygame.draw.rect(
            self.screen, (255, 255, 255), (x, y, self.tile_size, self.tile_size), 1
        )

    def draw_health_bar(self, entity, x, y):
        """Draw a health bar for entities with health."""
        if not hasattr(entity, "health") or not hasattr(entity, "max_health"):
            return

        health_ratio = entity.health / entity.max_health
        bar_width = self.tile_size - 4
        bar_height = 4
        bar_x = x + 2
        bar_y = y + self.tile_size - 6

        # Background
        pygame.draw.rect(
            self.screen, (255, 0, 0), (bar_x, bar_y, bar_width, bar_height)
        )
        # Health
        health_width = int(bar_width * health_ratio)
        pygame.draw.rect(
            self.screen, (0, 255, 0), (bar_x, bar_y, health_width, bar_height)
        )

    def draw_world(self):
        """Draw the entire world."""
        self.screen.fill(self.colors["background"])

        # Draw all layers
        for y in range(self.world.height):
            for x in range(self.world.width):
                screen_x = x * self.tile_size
                screen_y = y * self.tile_size

                # Draw terrain layer
                terrain_entity = self.world.observe((y, x, self.world.terrain_layer))
                self.draw_entity(terrain_entity, screen_x, screen_y)

                # Draw dynamic layer
                dynamic_entity = self.world.observe((y, x, self.world.dynamic_layer))
                if dynamic_entity and type(dynamic_entity).__name__ != "Empty":
                    self.draw_entity(dynamic_entity, screen_x, screen_y)
                    self.draw_health_bar(dynamic_entity, screen_x, screen_y)

                # Draw beam layer
                beam_entity = self.world.observe((y, x, self.world.beam_layer))
                if beam_entity and type(beam_entity).__name__ != "Empty":
                    if "Attack" in type(beam_entity).__name__:
                        color = self.colors["attack_beam"]
                    elif "Punish" in type(beam_entity).__name__:
                        color = self.colors["punish_beam"]
                    else:
                        color = (255, 255, 255)

                    # Draw beam as a semi-transparent overlay
                    beam_surface = pygame.Surface((self.tile_size, self.tile_size))
                    beam_surface.set_alpha(128)
                    beam_surface.fill(color)
                    self.screen.blit(beam_surface, (screen_x, screen_y))

    def draw_ui(self):
        """Draw the user interface."""
        # Game info
        info_text = [
            f"Turn: {self.game_info['turn']}",
            f"Health: {self.game_info['agent_health']}",
            f"Attack CD: {self.game_info['attack_cooldown']}",
            f"Punish CD: {self.game_info['punish_cooldown']}",
            f"Reward: {self.game_info['total_reward']:.2f}",
        ]

        y_offset = 10
        for text in info_text:
            surface = self.font.render(text, True, self.colors["text"])
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25

        # Controls
        controls = [
            "Controls:",
            "W - Forward, S - Backward",
            "A - Step Left, D - Step Right",
            "Q - Turn Left, E - Turn Right",
            "R - Attack, P - Punish",
            "SPACE - Noop, ENTER - End Turn",
            "ESC - Quit",
        ]

        x_offset = self.display_width - 200
        y_offset = 10
        for text in controls:
            surface = self.small_font.render(text, True, self.colors["text"])
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += 20

    def run(self):
        """Main game loop."""
        print("Stag Hunt Physical - Human Player Visualization")
        print("Use the controls shown on screen to play the game.")
        print("Position yourself near resources and use R to attack!")

        while self.running:
            self.handle_events()
            self.draw_world()
            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stag Hunt Physical Human Player Visualization"
    )
    parser.add_argument(
        "--config", default="configs/config_ascii_map.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--tile-size", type=int, default=32, help="Size of each tile in pixels"
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")

    args = parser.parse_args()

    # Create and run the visualization
    game = StagHuntHumanVisualization(args.config)
    game.tile_size = args.tile_size
    game.fps = args.fps

    # Recalculate display size
    game.display_width = game.world.width * game.tile_size
    game.display_height = game.world.height * game.tile_size
    game.screen = pygame.display.set_mode((game.display_width, game.display_height))

    try:
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
