"""Pygame implementation of the Treasurehunt game.

Allows human players to play the treasurehunt game using keyboard controls.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import sorrel modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

import numpy as np
import pygame

from sorrel.action.action_spec import ActionSpec
from sorrel.examples.treasurehunt_mp.agents import TreasurehuntAgent
from sorrel.examples.treasurehunt_mp.entities import EmptyEntity, Gem, Sand, Wall
from sorrel.examples.treasurehunt_mp.world import TreasurehuntWorld
from sorrel.observation.observation_spec import OneHotObservationSpec


class TreasurehuntPygame:
    """Main pygame class for the Treasurehunt game."""

    def __init__(self, config=None):
        # Initialize pygame
        pygame.init()

        # Default configuration
        if config is None:
            config = {
                "world": {
                    "height": 10,
                    "width": 10,
                    "gem_value": 10,
                    "spawn_prob": 0.02,
                },
                "display": {
                    "cell_size": 50,
                    "fps": 10,
                },
            }

        self.config = config
        self.cell_size = config["display"]["cell_size"]
        self.fps = config["display"]["fps"]

        # Calculate screen dimensions
        self.screen_width = config["world"]["width"] * self.cell_size
        self.screen_height = config["world"]["height"] * self.cell_size

        # Create screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Treasurehunt Pygame")

        # Colors
        self.colors = {
            "empty": (200, 200, 200),
            "wall": (100, 100, 100),
            "sand": (194, 178, 128),
            "gem": (255, 215, 0),
            "agent": (0, 0, 255),
            "background": (50, 50, 50),
        }

        # Initialize game world
        self.world = TreasurehuntWorld(config, EmptyEntity())
        self.setup_world()

        # Initialize human player
        self.human_player = self.create_human_player()

        # Game state
        self.running = True
        self.clock = pygame.time.Clock()
        self.score = 0

    def setup_world(self):
        """Set up the initial world state."""
        # Add walls around the border
        for y in range(self.world.height):
            for x in range(self.world.width):
                if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                    # Add walls around the edge
                    self.world.add((y, x, 1), Wall())
                else:
                    # Add sand on the bottom layer
                    self.world.add((y, x, 0), Sand())
                    # Add empty space on the top layer
                    self.world.add((y, x, 1), EmptyEntity())

    def create_human_player(self):
        """Create a human player agent."""
        # Create observation spec (not really needed for human player, but required by Agent)
        entity_list = ["EmptyEntity", "Wall", "Sand", "Gem", "TreasurehuntAgent"]
        observation_spec = OneHotObservationSpec(
            entity_list,
            full_view=False,
            vision_radius=1,
            env_dims=(self.world.height, self.world.width, 2),
        )

        # Create action spec
        action_spec = ActionSpec(["up", "down", "left", "right"])

        # Create a dummy model (not used for human player)
        class DummyModel:
            def take_action(self, state):
                return 0

        # Create the agent
        agent = TreasurehuntAgent(
            observation_spec=observation_spec,
            action_spec=action_spec,
            model=DummyModel(),
        )

        # Place agent in a random valid location
        valid_locations = []
        for y in range(1, self.world.height - 1):
            for x in range(1, self.world.width - 1):
                valid_locations.append((y, x, 1))

        if valid_locations:
            start_location = np.random.choice(len(valid_locations))
            agent_location = valid_locations[start_location]
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
        """Handle keyboard input for player movement."""
        if not self.human_player:
            return

        # Map keys to actions
        key_to_action = {
            pygame.K_UP: "up",
            pygame.K_DOWN: "down",
            pygame.K_LEFT: "left",
            pygame.K_RIGHT: "right",
            pygame.K_w: "up",
            pygame.K_s: "down",
            pygame.K_a: "left",
            pygame.K_d: "right",
        }

        if key in key_to_action:
            action_name = key_to_action[key]
            self.move_player(action_name)

    def move_player(self, action_name):
        """Move the human player based on action."""
        if not self.human_player:
            return

        current_location = self.human_player.location
        new_location = current_location

        # Calculate new location based on action
        if action_name == "up":
            new_location = (
                current_location[0] - 1,
                current_location[1],
                current_location[2],
            )
        elif action_name == "down":
            new_location = (
                current_location[0] + 1,
                current_location[1],
                current_location[2],
            )
        elif action_name == "left":
            new_location = (
                current_location[0],
                current_location[1] - 1,
                current_location[2],
            )
        elif action_name == "right":
            new_location = (
                current_location[0],
                current_location[1] + 1,
                current_location[2],
            )

        # Check if new location is valid
        if self.is_valid_location(new_location):
            # Get reward from the target location
            target_object = self.world.observe(new_location)
            reward = getattr(target_object, "value", 0)
            self.score += reward

            # Move the player
            self.world.move(self.human_player, new_location)

            # If we collected a gem, spawn a new one randomly
            if reward > 0:
                self.spawn_random_gem()

    def is_valid_location(self, location):
        """Check if a location is valid for the player to move to."""
        if (
            location[0] < 0
            or location[0] >= self.world.height
            or location[1] < 0
            or location[1] >= self.world.width
        ):
            return False

        # Check if there's a wall at the target location
        target_object = self.world.observe(location)
        if hasattr(target_object, "passable") and not target_object.passable:
            return False

        return True

    def spawn_random_gem(self):
        """Spawn a gem at a random empty location."""
        empty_locations = []
        for y in range(1, self.world.height - 1):
            for x in range(1, self.world.width - 1):
                location = (y, x, 1)
                obj = self.world.observe(location)
                if isinstance(obj, EmptyEntity):
                    empty_locations.append(location)

        if empty_locations:
            gem_location = np.random.choice(len(empty_locations))
            self.world.add(
                empty_locations[gem_location], Gem(self.config["world"]["gem_value"])
            )

    def draw(self):
        """Draw the game world."""
        self.screen.fill(self.colors["background"])

        # Draw the world
        for y in range(self.world.height):
            for x in range(self.world.width):
                # Draw bottom layer (sand)
                sand_location = (y, x, 0)
                sand_obj = self.world.observe(sand_location)
                if isinstance(sand_obj, Sand):
                    rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    )
                    pygame.draw.rect(self.screen, self.colors["sand"], rect)

                # Draw top layer
                top_location = (y, x, 1)
                top_obj = self.world.observe(top_location)

                if isinstance(top_obj, Wall):
                    color = self.colors["wall"]
                elif isinstance(top_obj, Gem):
                    color = self.colors["gem"]
                elif isinstance(top_obj, TreasurehuntAgent):
                    color = self.colors["agent"]
                else:  # EmptyEntity
                    color = self.colors["empty"]

                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, color, rect)

                # Draw border
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Draw instructions
        font_small = pygame.font.Font(None, 24)
        instructions = [
            "Use WASD or Arrow Keys to move",
            "Collect gold gems to score points",
            "Press ESC to quit",
        ]

        for i, instruction in enumerate(instructions):
            text = font_small.render(instruction, True, (255, 255, 255))
            self.screen.blit(text, (10, self.screen_height - 80 + i * 25))

        pygame.display.flip()

    def run(self):
        """Main game loop."""
        print("Treasurehunt Pygame Started!")
        print("Use WASD or Arrow Keys to move around and collect gems!")
        print("Press ESC or close the window to quit.")

        # Spawn initial gems
        for _ in range(3):
            self.spawn_random_gem()

        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()
        print(f"Game Over! Final Score: {self.score}")


def main():
    """Main function to run the pygame version."""
    # Configuration
    config = {
        "world": {
            "height": 12,
            "width": 12,
            "gem_value": 10,
            "spawn_prob": 0.02,
        },
        "display": {
            "cell_size": 50,
            "fps": 10,
        },
    }

    # Create and run the game
    game = TreasurehuntPygame(config)
    game.run()


if __name__ == "__main__":
    main()
