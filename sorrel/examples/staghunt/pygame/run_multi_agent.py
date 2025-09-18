#!/usr/bin/env python3
"""
Multi-agent launcher for Stag Hunt Pygame.
This script provides an easy way to run the game with different agent configurations.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import sorrel modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from staghunt_ascii_pygame import StagHuntASCIIPygame


def main():
    """Main function with different multi-agent configurations."""
    
    print("Stag Hunt Multi-Agent Pygame Launcher")
    print("=" * 40)
    print("1. Single Agent (1 player)")
    print("2. Two Agents (2 players, turn-based)")
    print("3. Three Agents (3 players, turn-based)")
    print("4. Four Agents (4 players, turn-based)")
    print("5. Custom Configuration")
    print()
    
    choice = input("Select number of agents (1-5, or press Enter for single agent): ").strip()
    
    if choice == "2":
        num_agents = 2
        tile_size = 32
        fps = 10
        turn_duration = 30
    elif choice == "3":
        num_agents = 3
        tile_size = 28
        fps = 10
        turn_duration = 25
    elif choice == "4":
        num_agents = 4
        tile_size = 24
        fps = 10
        turn_duration = 20
    elif choice == "5":
        try:
            num_agents = int(input("Enter number of agents (1-4): "))
            num_agents = max(1, min(4, num_agents))
            tile_size = int(input("Enter tile size (16-64): "))
            tile_size = max(16, min(64, tile_size))
            fps = int(input("Enter FPS (5-30): "))
            fps = max(5, min(30, fps))
            if num_agents > 1:
                turn_duration = int(input("Enter turn duration in frames (10-60): "))
                turn_duration = max(10, min(60, turn_duration))
            else:
                turn_duration = 30
        except ValueError:
            print("Invalid input, using default configuration.")
            num_agents = 1
            tile_size = 32
            fps = 10
            turn_duration = 30
    else:  # Default or choice == "1"
        num_agents = 1
        tile_size = 32
        fps = 10
        turn_duration = 30
    
    print(f"\nStarting Stag Hunt with {num_agents} agent(s)...")
    print(f"Tile size: {tile_size}px, FPS: {fps}")
    if num_agents > 1:
        print(f"Turn duration: {turn_duration} frames")
    print()
    
    # Configuration
    config = {
        "world": {
            "generation_mode": "ascii_map",
            "ascii_map_file": "docs/stag_hunt_ascii_map_clean.txt",
            "num_agents": num_agents,
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
            "tile_size": tile_size,
            "fps": fps,
            "turn_duration": turn_duration,
        }
    }
    
    # Create and run the game
    game = StagHuntASCIIPygame(config)
    game.run()


if __name__ == "__main__":
    main()
