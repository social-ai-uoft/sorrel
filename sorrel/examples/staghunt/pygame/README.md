# Stag Hunt Pygame

A pygame implementation of the Stag Hunt game from the Sorrel framework. This allows human players to play the stag hunt social dilemma game using keyboard controls.

## Game Overview

Stag Hunt is a classic social dilemma game where players must choose between two strategies:
- **Stag Strategy**: Cooperate with others for higher rewards (4 points if both choose stag)
- **Hare Strategy**: Play it safe alone for guaranteed moderate rewards (2 points if both choose hare)

The game features resource collection, agent orientation, and strategic interactions.

## Features

- **Human-controlled gameplay**: Use keyboard controls to move, turn, and interact
- **Visual representation**: Color-coded grid showing different entities and agent orientation
- **Resource collection**: Collect stag (red circles) and hare (green squares) resources
- **Strategic interactions**: Fire interaction beams to engage with other agents
- **Inventory management**: Track collected resources and readiness status
- **Configurable arena sizes**: Play on different sized arenas
- **Real-time scoring**: Track your performance and strategy choices

## Installation

Make sure you have pygame installed:

```bash
pip install pygame
```

## Running the Game

### Quick Start

Run the main game with default settings:

```bash
python staghunt_pygame.py
```

### Interactive Launcher

Use the launcher script to choose different arena sizes:

```bash
python run_staghunt_pygame.py
```

This will give you options for:
- Small Arena (8x8)
- Medium Arena (11x11) - Default
- Large Arena (15x15)
- Custom Arena (you specify the dimensions)

## Controls

- **W/S**: Move forward/backward relative to your orientation
- **A/D**: Step left/right relative to your orientation
- **Q/E**: Turn left/right (change orientation)
- **SPACE**: Interact (fire interaction beam)
- **ESC**: Quit game

## Game Elements

- **Blue Circle with Arrow**: Your player character (arrow shows orientation)
- **Red Circles**: Stag resources (rare, high value)
- **Green Squares**: Hare resources (common, moderate value)
- **Gray Squares**: Walls (cannot pass through)
- **Beige Background**: Sand terrain (walkable surface)
- **White Overlay**: Interaction beams (temporary)
- **Yellow Squares**: Spawn points

## Game Rules

1. **Movement**: Use W/S to move forward/backward, A/D to step sideways
2. **Orientation**: Use Q/E to turn left/right - this affects movement direction
3. **Resource Collection**: Walk over resources to collect them
4. **Inventory**: Collect both stag and hare resources to become "ready"
5. **Interactions**: Use SPACE to fire an interaction beam when ready
6. **Strategy**: Your strategy is determined by which resource type you have more of
7. **Payoffs**: 
   - Both choose Stag: 4 points each
   - Both choose Hare: 2 points each
   - Mixed choice: 0 points for Stag player, 2 points for Hare player

## Strategy Tips

- **Stag Strategy**: Collect more stag resources than hare resources
- **Hare Strategy**: Collect more hare resources than stag resources
- **Timing**: Wait for the right moment to interact with other agents
- **Positioning**: Use your orientation to aim your interaction beam effectively

## Configuration

You can modify the game configuration by editing the `config` dictionary:

```python
config = {
    "world": {
        "height": 11,                    # Arena height
        "width": 11,                     # Arena width
        "num_agents": 1,                 # Number of agents (currently only 1 human)
        "resource_density": 0.15,        # Probability of resource spawning
        "taste_reward": 0.1,             # Reward for collecting resources
        "destroyable_health": 3,         # Health of resources (zap hits to destroy)
        "beam_length": 3,                # Length of interaction beam
        "beam_radius": 1,                # Radius of interaction beam
        "beam_cooldown": 3,              # Turns before beam can be used again
        "respawn_lag": 10,               # Turns before resources can respawn
        "payoff_matrix": [[4, 0], [2, 2]], # Stag Hunt payoff matrix
        "interaction_reward": 1.0,       # Bonus for participating in interaction
        "freeze_duration": 5,            # Turns agent is frozen after interaction
        "respawn_delay": 10,             # Turns before agent respawns
    },
    "display": {
        "cell_size": 50,                 # Size of each grid cell in pixels
        "fps": 10,                       # Game speed (frames per second)
    }
}
```

## Integration with Sorrel

This pygame implementation integrates with the existing Sorrel framework:

- Uses the same `StagHuntWorld` class
- Uses the same entity classes (`Wall`, `Sand`, `Spawn`, `StagResource`, `HareResource`, etc.)
- Uses the same `StagHuntAgent` class (with a dummy model for human input)
- Maintains the same game mechanics, scoring system, and social dilemma structure
- Supports the same configuration parameters

## Files

- `staghunt_pygame.py`: Main pygame implementation
- `run_staghunt_pygame.py`: Interactive launcher script
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
4. Try running with a smaller arena size if performance is slow

## Game Theory Background

The Stag Hunt is a classic game theory scenario that models the tension between:
- **Risk vs. Reward**: Stag hunting requires cooperation but offers higher rewards
- **Individual vs. Collective**: Personal safety vs. group benefit
- **Trust**: Can you trust others to cooperate?

This implementation allows you to experience these strategic decisions firsthand!

Enjoy playing Stag Hunt!
