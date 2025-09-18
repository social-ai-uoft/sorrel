#!/usr/bin/env python3
"""
Simple launcher script for the Treasurehunt Pygame.
This script provides an easy way to run the pygame version with different configurations.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import sorrel modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from treasurehunt_pygame import TreasurehuntPygame


def main():
    """Main function with different game configurations."""
    
    print("Treasurehunt Pygame Launcher")
    print("=" * 30)
    print("1. Small Grid (8x8)")
    print("2. Medium Grid (12x12) - Default")
    print("3. Large Grid (16x16)")
    print("4. Custom Grid")
    print()
    
    choice = input("Select grid size (1-4, or press Enter for default): ").strip()
    
    if choice == "1":
        config = {
            "world": {
                "height": 8,
                "width": 8,
                "gem_value": 10,
                "spawn_prob": 0.02,
            },
            "display": {
                "cell_size": 60,
                "fps": 10,
            }
        }
    elif choice == "3":
        config = {
            "world": {
                "height": 16,
                "width": 16,
                "gem_value": 10,
                "spawn_prob": 0.02,
            },
            "display": {
                "cell_size": 40,
                "fps": 10,
            }
        }
    elif choice == "4":
        try:
            width = int(input("Enter grid width: "))
            height = int(input("Enter grid height: "))
            cell_size = max(20, min(80, 800 // max(width, height)))
            
            config = {
                "world": {
                    "height": height,
                    "width": width,
                    "gem_value": 10,
                    "spawn_prob": 0.02,
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
                "height": 12,
                "width": 12,
                "gem_value": 10,
                "spawn_prob": 0.02,
            },
            "display": {
                "cell_size": 50,
                "fps": 10,
            }
        }
    
    print(f"\nStarting game with {config['world']['width']}x{config['world']['height']} grid...")
    print("Controls: WASD or Arrow Keys to move, ESC to quit")
    print()
    
    # Create and run the game
    game = TreasurehuntPygame(config)
    game.run()


if __name__ == "__main__":
    main()
