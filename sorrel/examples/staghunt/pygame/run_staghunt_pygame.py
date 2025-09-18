#!/usr/bin/env python3
"""
Simple launcher script for the Stag Hunt Pygame.
This script provides an easy way to run the pygame version with different configurations.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import sorrel modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from staghunt_pygame import StagHuntPygame


def main():
    """Main function with different game configurations."""
    
    print("Stag Hunt Pygame Launcher")
    print("=" * 30)
    print("1. Small Arena (8x8)")
    print("2. Medium Arena (11x11) - Default")
    print("3. Large Arena (15x15)")
    print("4. Custom Arena")
    print()
    
    choice = input("Select arena size (1-4, or press Enter for default): ").strip()
    
    if choice == "1":
        config = {
            "world": {
                "height": 8,
                "width": 8,
                "num_agents": 1,
                "resource_density": 0.2,
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
                "cell_size": 60,
                "fps": 10,
            }
        }
    elif choice == "3":
        config = {
            "world": {
                "height": 15,
                "width": 15,
                "num_agents": 1,
                "resource_density": 0.1,
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
                "cell_size": 40,
                "fps": 10,
            }
        }
    elif choice == "4":
        try:
            width = int(input("Enter arena width: "))
            height = int(input("Enter arena height: "))
            cell_size = max(20, min(80, 800 // max(width, height)))
            
            config = {
                "world": {
                    "height": height,
                    "width": width,
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
                    "cell_size": cell_size,
                    "fps": 10,
                }
            }
        except ValueError:
            print("Invalid input, using default configuration.")
            config = None
    else:  # Default or choice == "2"
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
    
    print(f"\nStarting Stag Hunt with {config['world']['width']}x{config['world']['height']} arena...")
    print("Game Rules:")
    print("- Collect stag (red circles) and hare (green squares) resources")
    print("- Stag resources are rarer but more valuable")
    print("- Use interaction beam to engage with other agents")
    print("- Payoff matrix: Stag+Stag=4, Stag+Hare=0, Hare+Hare=2")
    print()
    print("Controls:")
    print("- W/S: Move forward/backward")
    print("- A/D: Step left/right") 
    print("- Q/E: Turn left/right")
    print("- SPACE: Interact (fire beam)")
    print("- ESC: Quit")
    print()
    
    # Create and run the game
    game = StagHuntPygame(config)
    game.run()


if __name__ == "__main__":
    main()
