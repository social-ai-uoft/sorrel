# State Punishment Beta

A modern implementation of the state punishment game using the Sorrel framework. This game explores collective punishment, voting mechanisms, and social dynamics in a multi-agent environment.

## Game Overview

In the State Punishment game, agents navigate a gridworld environment collecting resources while participating in a collective punishment system. Agents can vote to increase or decrease the punishment probability for certain "taboo" resources, creating a dynamic social environment where individual and collective interests interact.

## Key Features

### 1. **Composite Views & Actions Support**
- **Composite Views**: Agents can observe the world from multiple perspectives (their own + other agents)
- **Composite Actions**: Movement and voting can be combined into single actions
- **Simple Mode**: Traditional separate movement and voting actions
- **Configurable**: Enable/disable via command line arguments or configuration

### 2. **Punishment Level Visibility**
- Agents can see the current punishment probability in their observations
- Real-time feedback on how their votes affect the system
- Social harm tracking for each agent

### 3. **Noop Action**
- Agents can choose to do nothing (action 6 in simple mode, action 13 in composite mode)
- Useful for strategic waiting and exploration
- Helps agents learn when not to act

### 4. **Dynamic Punishment System**
- Punishment probability changes based on agent votes
- Different resources have different punishment schedules
- Social harm affects all agents when taboo resources are collected

## Game Mechanics

### Resources
- **Gem**: High value (+5), but causes social harm and can be punished
- **Coin**: High value (+10), no social harm, never punished
- **Bone**: Negative value (-3), high social harm, always punished when taboo

### Actions
#### Simple Mode (7 actions):
- `up`, `down`, `left`, `right`: Movement
- `vote_increase`, `vote_decrease`: Voting
- `noop`: Do nothing

#### Composite Mode (13 actions):
- `up_no_vote`, `down_no_vote`, `left_no_vote`, `right_no_vote`: Movement without voting
- `up_increase`, `down_increase`, `left_increase`, `right_increase`: Movement + vote to increase punishment
- `up_decrease`, `down_decrease`, `left_decrease`, `right_decrease`: Movement + vote to decrease punishment
- `noop`: Do nothing

### Observation Space
- Standard gridworld observation (entity positions, types)
- **Punishment Level**: Current punishment probability (0-1)
- **Social Harm**: Current social harm affecting this agent
- **Vote Activity**: Recent voting patterns (normalized)

## Usage

### Basic Usage
```bash
# Run with default settings (simple mode, 3 agents)
python main.py

# Run with composite views and actions
python main.py --composite-views --composite-actions

# Run with custom number of agents and epochs
python main.py --num-agents 5 --epochs 20000
```

### Command Line Arguments
- `--composite-views`: Enable composite state observations
- `--composite-actions`: Enable composite actions (movement + voting)
- `--num-agents N`: Number of agents in the environment (default: 3)
- `--epochs N`: Number of training epochs (default: 10000)

### Configuration
The game can be configured via `configs/config.yaml` or by modifying the configuration dictionary in `main.py`.

Key parameters:
- `world.init_punishment_prob`: Initial punishment probability (0-1)
- `world.punishment_magnitude`: Magnitude of punishment (negative value)
- `world.change_per_vote`: How much punishment probability changes per vote
- `world.taboo_resources`: List of resources that can be punished
- `world.spawn_prob`: Probability of new resources spawning each turn

## File Structure

```
state_punishment_beta/
├── main.py              # Main experiment script
├── world.py             # StatePunishmentWorld class
├── env.py               # StatePunishmentEnv class  
├── agents.py            # StatePunishmentAgent class
├── entities.py          # Game entities (Gem, Coin, Bone, etc.)
├── state_system.py      # Punishment state management
├── configs/
│   └── config.yaml      # Configuration file
└── README.md            # This file
```

## Key Classes

### StatePunishmentWorld
- Inherits from `Gridworld`
- Manages punishment state system
- Handles resource spawning and social harm
- Tracks punishment level and voting

### StatePunishmentEnv
- Inherits from `Environment[StatePunishmentWorld]`
- Implements agent setup and environment population
- Handles agent transitions and reward calculation
- Manages punishment system updates

### StatePunishmentAgent
- Handles individual agent behavior
- Supports both simple and composite action modes
- Integrates punishment level into observations
- Manages voting actions and resource collection

### StateSystem
- Manages punishment probability and voting
- Handles punishment calculations
- Tracks punishment level changes and history

## Logging and Metrics

The game includes comprehensive logging with:
- Individual agent scores and encounters
- Punishment level dynamics
- Voting behavior patterns
- Social harm distribution
- Resource collection statistics

Metrics are logged to both console and TensorBoard for analysis.

## Differences from Original

This implementation modernizes the original state_punishment game by:

1. **Framework Migration**: Uses Sorrel instead of agentarium
2. **Clean Architecture**: Separates world, environment, and agent logic
3. **Modern Patterns**: Follows established Sorrel patterns from other games
4. **Simplified Configuration**: Dictionary-based config instead of complex YAML
5. **Enhanced Features**: Composite views/actions, noop action, better logging
6. **Code Quality**: Cleaner, more maintainable code structure

## Future Enhancements

Potential improvements for future versions:
- More sophisticated punishment schedules
- Dynamic taboo resource assignment
- Multi-level voting systems
- Communication between agents
- Asymmetric agent roles
- Environmental factors affecting punishment
