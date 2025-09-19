"""Entities for the state punishment game."""

import numpy as np
from pathlib import Path
from sorrel.entities import Entity


class EmptyEntity(Entity):
    """Empty entity representing empty space."""
    
    def __init__(self):
        super().__init__()
        self.kind = "EmptyEntity"
        self.passable = True
        self.value = 0
        self.social_harm = 0
        self.sprite = Path(__file__).parent / "./assets/empty.png"
        self.has_transitions = True
    
    def transition(self, world):
        """Randomly spawn resources in empty locations."""
        if hasattr(world, 'spawn_prob') and np.random.random() < world.spawn_prob:
            # Use the world's spawn_entity method to create a new resource
            world.spawn_entity(self.location)


class Wall(Entity):
    """Wall entity that blocks movement."""
    
    def __init__(self):
        super().__init__()
        self.kind = "Wall"
        self.passable = False
        self.value = 0
        self.social_harm = 0
        self.sprite = Path(__file__).parent / "./assets/wall.png"




class A(Entity):
    """Resource A with configurable value and social harm."""
    
    def __init__(self, value: float = 3.0, social_harm: float = 0.5):
        super().__init__()
        self.kind = "A"
        self.passable = True
        self.value = value
        self.social_harm = social_harm
        self.sprite = Path(__file__).parent / "./assets/gem.png"


class B(Entity):
    """Resource B with configurable value and social harm."""
    
    def __init__(self, value: float = 7.0, social_harm: float = 1.0):
        super().__init__()
        self.kind = "B"
        self.passable = True
        self.value = value
        self.social_harm = social_harm
        self.sprite = Path(__file__).parent / "./assets/coin.png"


class C(Entity):
    """Resource C with configurable value and social harm."""
    
    def __init__(self, value: float = 2.0, social_harm: float = 0.3):
        super().__init__()
        self.kind = "C"
        self.passable = True
        self.value = value
        self.social_harm = social_harm
        self.sprite = Path(__file__).parent / "./assets/food.png"


class D(Entity):
    """Resource D with configurable value and social harm."""
    
    def __init__(self, value: float = -2.0, social_harm: float = 1.5):
        super().__init__()
        self.kind = "D"
        self.passable = True
        self.value = value
        self.social_harm = social_harm
        self.sprite = Path(__file__).parent / "./assets/apple.png"


class E(Entity):
    """Resource E with configurable value and social harm."""
    
    def __init__(self, value: float = 1.0, social_harm: float = 0.1):
        super().__init__()
        self.kind = "E"
        self.passable = True
        self.value = value
        self.social_harm = social_harm
        self.sprite = Path(__file__).parent / "./assets/bone.png"
