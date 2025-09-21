# Treasurehunt Pygame

A pygame implementation of the Treasurehunt game from the Sorrel framework. This allows human players to play the treasurehunt game using keyboard controls.

## Features

- **Human-controlled gameplay**: Use keyboard controls to move around and collect gems
- **Visual representation**: Color-coded grid showing different entities
- **Score tracking**: Keep track of collected gems
- **Configurable grid sizes**: Play on different sized grids
- **Real-time gem spawning**: New gems spawn as you collect them

## Installation

Make sure you have pygame installed:

```bash
pip install pygame
```

## Running the Game

### Quick Start

Run the main game with default settings:

```bash
python treasurehunt_pygame.py
```

### Interactive Launcher

Use the launcher script to choose different grid sizes:

```bash
python run_pygame.py
```

This will give you options for:
- Small Grid (8x8)
- Medium Grid (12x12) - Default
- Large Grid (16x16)
- Custom Grid (you specify the dimensions)

## Controls

- **Movement**: Use WASD keys or Arrow keys
  - W/↑: Move up
  - S/↓: Move down
  - A/←: Move left
  - D/→: Move right
- **Quit**: Press ESC or close the window

## Game Elements

- **Blue Square**: Your player character
- **Gold Squares**: Gems to collect (worth 10 points each)
- **Gray Squares**: Walls (cannot pass through)
- **Beige Background**: Sand (walkable surface)
- **White Squares**: Empty spaces

## Game Rules

1. Move around the grid using keyboard controls
2. Collect gold gems to score points
3. Avoid hitting walls (you can't move through them)
4. New gems will spawn randomly as you collect them
5. Try to get the highest score possible!

## Configuration

You can modify the game configuration by editing the `config` dictionary in the main files:

```python
config = {
    "world": {
        "height": 12,        # Grid height
        "width": 12,         # Grid width
        "gem_value": 10,     # Points per gem
        "spawn_prob": 0.02,  # Gem spawn probability
    },
    "display": {
        "cell_size": 50,     # Size of each grid cell in pixels
        "fps": 10,          # Game speed (frames per second)
    }
}
```

## Integration with Sorrel

This pygame implementation integrates with the existing Sorrel framework:

- Uses the same `TreasurehuntWorld` class
- Uses the same entity classes (`Wall`, `Sand`, `Gem`, `EmptyEntity`)
- Uses the same `TreasurehuntAgent` class (with a dummy model for human input)
- Maintains the same game mechanics and scoring system

## Files

- `treasurehunt_pygame.py`: Main pygame implementation
- `run_pygame.py`: Interactive launcher script
- `README.md`: This documentation file

## Requirements

- Python 3.7+
- pygame 2.0+
- numpy
- sorrel framework (for the game logic)

## Troubleshooting

If you encounter any issues:

1. Make sure pygame is installed: `pip install pygame`
2. Make sure you're running from the correct directory
3. Check that all sorrel dependencies are available
4. Try running with a smaller grid size if performance is slow

Enjoy playing Treasurehunt!
