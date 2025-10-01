# Stag Hunt Pygame

A comprehensive pygame implementation of the Stag Hunt game from the Sorrel framework. This allows human players to experience the stag hunt social dilemma game using keyboard controls, with both single-agent and multi-agent turn-based gameplay modes.

## Game Overview

Stag Hunt is a classic social dilemma game where players must choose between two strategies:
- **Stag Strategy**: Cooperate with others for higher rewards (4 points if both choose stag)
- **Hare Strategy**: Play it safe alone for guaranteed moderate rewards (2 points if both choose hare)

The game features resource collection, agent orientation, strategic interactions, and proper agent lifecycle management (frozen → removed → respawned).

## Features

### 🎮 **Core Gameplay**
- **Human-controlled gameplay**: Use keyboard controls to move, turn, and interact
- **Multi-agent support**: Play with 1-4 agents in turn-based mode
- **Exact Sorrel replication**: Uses the same sprites, ASCII maps, and game logic as the original framework
- **Turn-based system**: Agents take turns with keyboard-controlled timing

### 🎨 **Visual Features**
- **Exact sprite rendering**: Uses the same PNG sprites as the original Sorrel game
- **ASCII map support**: Loads and displays the exact same ASCII map layouts
- **Agent orientation**: Visual arrows showing which direction agents are facing
- **Real-time UI**: Live score tracking, turn counting, and agent status display

### 🏆 **Game Mechanics**
- **Resource collection**: Collect stag (red circles) and hare (green squares) resources
- **Strategic interactions**: Fire interaction beams to engage with other agents
- **Inventory management**: Track collected resources and readiness status
- **Proper agent lifecycle**: Frozen → Removed → Respawned cycle
- **Dual reward system**: Both agents receive rewards from interactions
- **Configurable parameters**: Customizable arena sizes, tile sizes, FPS, and turn duration

## Installation

Make sure you have pygame installed:

```bash
pip install pygame
```

## Running the Game

### 🚀 **Quick Start - Human Player Visualization (Recommended)**

Run the human player visualization interface (similar to the Jupyter notebook):

```bash
python human_player_visualization.py
```

### 🎮 **Physical ASCII Map Version**

Run the enhanced Physical ASCII map version with ATTACK/PUNISH actions and health system:

```bash
python staghunt_physical_ascii_pygame.py --agents 2
```

**Command line options:**
- `--agents N`: Number of agents (1-4, default: 1)
- `--tile-size N`: Tile size in pixels (16-64, default: 32)
- `--fps N`: Game speed (5-30, default: 10)
- `--turn-duration N`: Turn duration in frames (10-60, default: 30)

### 🎮 **Original ASCII Map Version**

Run the original ASCII map version (without health system):

```bash
python staghunt_ascii_pygame.py --agents 2
```

### 🎮 **Multi-Agent Launcher**

Use the interactive launcher for easy configuration:

```bash
python run_multi_agent.py
```

This provides pre-configured options:
- **2 Agents**: Fast-paced gameplay
- **3 Agents**: Balanced strategy
- **4 Agents**: Complex multi-agent dynamics
- **Custom**: Configure all parameters yourself

### 📊 **Original Versions**

**Basic Pygame (Random Generation):**
```bash
python staghunt_pygame.py
```

**Interactive Launcher (Arena Sizes):**
```bash
python run_staghunt_pygame.py
```

Options:
- Small Arena (8x8)
- Medium Arena (11x11) - Default
- Large Arena (15x15)
- Custom Arena (you specify the dimensions)

## Controls

### 🎮 **Movement & Actions**
- **W/S**: Move forward/backward relative to your orientation
- **A/D**: Step left/right relative to your orientation
- **Q/E**: Turn left/right (change orientation)
- **SPACE/R**: Attack (fire red attack beam)
- **P**: Punish (fire yellow punish beam)
- **ESC**: Quit game

### 🔄 **Multi-Agent Controls**
- **TAB**: Switch between agents (in non-turn-based mode)
- **ENTER**: End turn manually (in turn-based mode)
- **Any Key**: Advance turn when agent is frozen

## Game Elements

### 🎯 **Visual Elements**
- **Blue Circle with Arrow**: Your player character (arrow shows orientation)
- **Red Circles**: Stag resources (rare, high value)
- **Green Squares**: Hare resources (common, moderate value)
- **Gray Squares**: Walls (cannot pass through)
- **Beige Background**: Sand terrain (walkable surface)
- **White Overlay**: Interaction beams (temporary)
- **Yellow Squares**: Spawn points

### 📊 **UI Elements**
- **Turn Counter**: Shows current full turn and individual agent actions
- **Score Display**: Individual agent scores and total score
- **Agent Status**: Current agent, inventory, position, and orientation
- **Action Status**: "Press any key to take action", "FROZEN", "REMOVED", etc.

## Game Rules

### 🎮 **Basic Gameplay**
1. **Movement**: Use W/S to move forward/backward, A/D to step sideways
2. **Orientation**: Use Q/E to turn left/right - this affects movement direction
3. **Resource Collection**: Walk over resources to collect them
4. **Inventory**: Collect both stag and hare resources to become "ready"
5. **Interactions**: Use SPACE to fire an interaction beam when ready
6. **Strategy**: Your strategy is determined by which resource type you have more of

### 🏆 **Scoring System**
- **Taste Rewards**: 0.1 points for each resource collected
- **Interaction Rewards**: Based on payoff matrix
- **Bonus Rewards**: 1.0 point for participating in interactions
- **Payoff Matrix**:
  - Both choose Stag: 4 points each
  - Both choose Hare: 2 points each
  - Mixed choice: 0 points for Stag player, 2 points for Hare player

### 🔄 **Agent Lifecycle (Multi-Agent Mode)**
1. **Active**: Agent can move and interact normally
2. **Frozen**: After interaction, agent cannot act for 5 frames
3. **Removed**: Agent disappears from board for 10 frames
4. **Respawned**: Agent reappears at random spawn point
5. **Active**: Agent can act normally again

## Strategy Tips

### 🎯 **Basic Strategies**
- **Stag Strategy**: Collect more stag resources than hare resources
- **Hare Strategy**: Collect more hare resources than stag resources
- **Timing**: Wait for the right moment to interact with other agents
- **Positioning**: Use your orientation to aim your interaction beam effectively

### 🤝 **Multi-Agent Strategies**
- **Cooperation**: Both agents benefit from stag-stag interactions (4 points each)
- **Risk Management**: Consider if other agents will cooperate or defect
- **Timing**: Coordinate interactions when both agents are ready
- **Resource Balance**: Maintain inventory balance for optimal strategy flexibility

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

## Human Player Visualization

The `human_player_visualization.py` script provides a **pure visualization interface** similar to the Jupyter notebook `human_player_test.ipynb`, but as a standalone pygame application.

### 🎯 **Purpose**
- **Testing interface**: Perfect for testing game mechanics and debugging
- **Pure visualization**: No complex game logic, just visualization and controls
- **Notebook equivalent**: Same functionality as the Jupyter notebook but in pygame
- **Minimal dependencies**: Uses only the core game code without additional features

### 🎮 **Features**
- **Real-time visualization**: See the game world with colored tiles
- **Health bars**: Visual health indicators for agents and resources
- **Action feedback**: Console output for each action performed
- **Turn management**: Manual turn advancement with ENTER key
- **Configurable**: Command-line options for tile size, FPS, and config file

### 🎨 **Visual Elements**
- **Colored tiles**: Different colors for walls, sand, spawn points, resources, agents
- **Health bars**: Green health bars on entities with health
- **Beam effects**: Semi-transparent overlays for attack/punish beams
- **UI overlay**: Game info, controls, and status display

### ⌨️ **Controls**
- **W/S**: Forward/Backward movement
- **A/D**: Step Left/Right
- **Q/E**: Turn Left/Right
- **R**: Attack (reduces resource health)
- **P**: Punish (affects other agents)
- **SPACE**: Noop (do nothing)
- **ENTER**: End turn
- **ESC**: Quit

### 🚀 **Usage**
```bash
# Basic usage
python human_player_visualization.py

# With custom settings
python human_player_visualization.py --tile-size 48 --fps 15 --config configs/config.yaml
```

### 🔧 **Command Line Options**
- `--config`: Path to config file (default: configs/config_ascii_map.yaml)
- `--tile-size`: Size of each tile in pixels (default: 32)
- `--fps`: Frames per second (default: 10)

## Integration with Sorrel

This pygame implementation provides **exact replication** of the Sorrel framework:

### 🔧 **Core Integration**
- **Same World Class**: Uses `StagHuntWorld` with identical configuration
- **Same Entities**: All entity classes (`Wall`, `Sand`, `Spawn`, `StagResource`, `HareResource`, etc.)
- **Same Agent Logic**: Uses `StagHuntAgent` with proper lifecycle management
- **Same Game Mechanics**: Identical scoring, interaction, and respawn systems
- **Same Configuration**: All parameters match the original framework

### 🎨 **Visual Fidelity**
- **Exact Sprites**: Uses the same PNG sprite files as the original
- **ASCII Map Support**: Loads and renders the exact same map layouts
- **Proper Rendering**: Multi-layer compositing matches Sorrel's visualization
- **Agent States**: Frozen, removed, and respawned states work identically

### 🏆 **Gameplay Accuracy**
- **Dual Rewards**: Both agents receive rewards from interactions
- **Turn Management**: Proper turn-based system with agent switching
- **Score Tracking**: Individual and total score tracking
- **Agent Lifecycle**: Complete frozen → removed → respawned cycle

## Files

### 🎮 **Main Game Files**
- `human_player_visualization.py`: **Human player visualization interface** (recommended for testing)
- `staghunt_physical_ascii_pygame.py`: Physical ASCII map version with health system
- `staghunt_ascii_pygame.py`: Original ASCII map version (without health system)
- `staghunt_physical_pygame.py`: Physical version with random generation
- `staghunt_pygame.py`: Basic pygame implementation with random generation
- `run_multi_agent_physical.py`: Multi-agent launcher for physical version
- `run_multi_agent.py`: Multi-agent launcher with pre-configured options
- `run_staghunt_physical_pygame.py`: Physical interactive launcher script
- `run_staghunt_pygame.py`: Original interactive launcher script

### 📁 **Supporting Files**
- `README.md`: This documentation file
- `assets/`: Sprite files (PNG images for visual elements)
- `docs/stag_hunt_ascii_map_clean.txt`: ASCII map layout file

## Requirements

- Python 3.7+
- pygame 2.0+
- numpy
- sorrel framework (for the game logic)

## Troubleshooting

### 🐛 **Common Issues**

**Game won't start:**
1. Make sure pygame is installed: `pip install pygame`
2. Make sure you're running from the correct directory
3. Check that all sorrel dependencies are available
4. Try running with fewer agents if performance is slow

**Visual issues:**
1. **Black screen**: Make sure you're using `staghunt_ascii_pygame.py` for the enhanced version
2. **Missing sprites**: Check that the `assets/` folder contains the PNG files
3. **ASCII map not loading**: Ensure `docs/stag_hunt_ascii_map_clean.txt` exists

**Multi-agent issues:**
1. **Agents not switching**: Use TAB key in non-turn-based mode
2. **Frozen agents stuck**: Press any key to advance turns
3. **Score not updating**: Both agents should receive rewards from interactions

### ⚡ **Performance Tips**
- Use smaller tile sizes (16-32) for better performance
- Lower FPS (5-10) for slower, more controlled gameplay
- Use fewer agents (1-2) for simpler gameplay

## Game Theory Background

The Stag Hunt is a classic game theory scenario that models the tension between:
- **Risk vs. Reward**: Stag hunting requires cooperation but offers higher rewards
- **Individual vs. Collective**: Personal safety vs. group benefit
- **Trust**: Can you trust others to cooperate?

### 🎓 **Educational Value**
This implementation allows you to:
- **Experience strategic decisions** firsthand
- **Test different strategies** in a controlled environment
- **Understand multi-agent dynamics** through turn-based gameplay
- **Observe emergent behaviors** in complex social dilemmas

### 🔬 **Research Applications**
- **Behavioral Economics**: Study human decision-making in social dilemmas
- **Multi-Agent Systems**: Test cooperation and competition strategies
- **Game Theory Education**: Visualize abstract concepts through interactive gameplay
- **AI Training**: Use as a benchmark for reinforcement learning algorithms

---

## 🎉 **Enjoy Playing Stag Hunt!**

Whether you're exploring game theory concepts, testing strategies, or just having fun, this implementation provides an authentic and engaging experience of the classic Stag Hunt social dilemma!
