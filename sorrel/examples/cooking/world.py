'''CookingWorld implementation for Sorrel example.

This module defines a simple Gridworld for the Cooking environment.
It provides a method to update dynamic stations (e.g., stove timers) each turn.
'''

from __future__ import annotations

from sorrel.worlds.gridworld import Gridworld
from sorrel.examples.cleanup.entities import EmptyEntity, Wall
from sorrel.examples.cooking.entities import Stove, Counter, Plate, Trash, IngredientEntity


class CookingWorld(Gridworld):
    # Define layer indices to match CleanupWorld expectations
    object_layer: int = 0
    agent_layer: int = 1
    """A minimal Cooking kitchen.

    The world is a 3-layer grid (entities, agents). The default entity
    for empty cells is ``EmptyEntity``. Walls are placed around the border.
    """

    def __init__(self, height: int = 5, width: int = 5, layers: int = 1):
        super().__init__(height=height, width=width, layers=layers, default_entity=EmptyEntity())
        # The base Gridworld already creates a full map of EmptyEntity instances.
        self.max_turns = 50

    def update_dynamic_entities(self) -> None:
        """Update entities that have per-turn behaviour (e.g., stoves)."""
        # Find all stove instances and call their transition method.
        for stove in self.get_entities_of_kind('Stove'):
            if isinstance(stove, Stove):
                stove.transition(self)