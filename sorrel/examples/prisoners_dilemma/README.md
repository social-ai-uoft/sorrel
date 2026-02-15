# Prisoner's Dilemma Gridworld

This is an example implementation of a gridworld environment inspired by the Prisoner's Dilemma game.

## Overview

In this environment, agents move around a grid and interact with "Exchange" entities.
When two agents "zap" an Exchange entity, they receive rewards based on the Prisoner's Dilemma payoff matrix.

### Actions
- **Move**: Up, Down, Left, Right
- **Cooperate**: Fire a "Cooperate" beam.
- **Defect**: Fire a "Defect" beam.

### Payoff Matrix
The rewards are determined by the combination of actions on the Exchange:

| | Cooperate | Defect |
|---|---|---|
| **Cooperate** | R (Reward) | S (Sucker) |
| **Defect** | T (Temptation) | P (Punishment) |

Default values:
- **T** = 5
- **R** = 3
- **P** = 1
- **S** = 0

## Entities
- **Exchange**: Points where interaction occurs. Requires 2 hits to resolve.
- **Walls**: Obstacles.
- **Sand**: Walkable terrain.

## Running
Use this environment within the Sorrel framework by importing `PrisonersDilemmaWorld` and `PrisonersDilemmaAgent`.
