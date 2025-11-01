"""Ingroup Bias Game implementation for the Sorrel framework.

This package implements the ingroup bias coordination game described by Köster et al. (2025)
as a partially observable Markov game in the Sorrel framework. The game features 8 agents
collecting resources of three different colors (red, green, blue) and engaging in social
interactions based on inventory similarity.

Key features:
- 8 agents with partial observability (11x11 vision window)
- 3 resource types with stochastic spawning
- Inventory-based readiness signaling
- Directed interaction beams with wall collision
- Reward proportional to inventory similarity (dot product)
- Post-interaction freezing and respawning mechanics
- Dual-choice evaluation environment for bias measurement
"""
