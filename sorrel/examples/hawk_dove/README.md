# Hawk-Dove (Chicken) Game Example

This directory contains an implementation of the Hawk-Dove (also known as Chicken) game within the Sorrel environment. 

## Overview
In this gridworld simulation, two agents wander around an environment and randomly encounter `Resource` entities. Agents must choose their behavior toward the resource by taking one of two actions:
- **Hawk**: Play aggressively to take all of the resource.
- **Dove**: Play peacefully and share the resource.

The payoffs are modeled with the classic Hawk-Dove payoff matrix:

### Payoff Matrix
The rewards are determined by the combination of actions on the Resource entity:

| | Hawk | Dove |
|---|---|---|
| **Hawk** | R (Reward) | S (Sucker) |
| **Dove** | T (Temptation) | P (Punishment) |

Default values:
- **T** = 2
- **R** = 1
- **P** = -4
- **S** = 0

## Running the example

```bash
python sorrel/examples/hawk_dove/main.py
```

The script runs the experiment for the default number of epochs (1000) with a maximum of 100 turns per epoch. TensorBoard logs, model checkpoints, and animated GIFs are saved under `data/`.

## Project structure

```
hawk_dove/ 
├─ assets/                    # Sprite images for game entities
├─ data/                      # Generated during runs: logs, checkpoints, GIFs
│   ├─ checkpoints/
│   ├─ gifs/
│   └─ logs/
├─ agents.py                  # Agent implementation
├─ custom_observation_spec.py # Custom observation spec
├─ entities.py                # Entity definitions
├─ env.py                     # Environment wrapper and setup
├─ main.py                    # Experiment entry point
├─ metrics_collector.py       # Metrics collector
└─ world.py                   # Gridworld definition (HawkDoveWorld)
```
