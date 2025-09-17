"""Entities for the state punishment game."""

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


class Wall(Entity):
    """Wall entity that blocks movement."""
    
    def __init__(self):
        super().__init__()
        self.kind = "Wall"
        self.passable = False
        self.value = 0
        self.social_harm = 0
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Gem(Entity):
    """Gem entity with positive value but potential social harm."""
    
    def __init__(self, value: float = 5.0, social_harm: float = 1.0):
        super().__init__()
        self.kind = "Gem"
        self.passable = True
        self.value = value
        self.social_harm = social_harm
        self.sprite = Path(__file__).parent / "./assets/gem.png"


class Coin(Entity):
    """Coin entity with positive value and no social harm."""
    
    def __init__(self, value: float = 10.0, social_harm: float = 0.0):
        super().__init__()
        self.kind = "Coin"
        self.passable = True
        self.value = value
        self.social_harm = social_harm
        self.sprite = Path(__file__).parent / "./assets/coin.png"


class Bone(Entity):
    """Bone entity with negative value and high social harm."""
    
    def __init__(self, value: float = -3.0, social_harm: float = 2.0):
        super().__init__()
        self.kind = "Bone"
        self.passable = True
        self.value = value
        self.social_harm = social_harm
        self.sprite = Path(__file__).parent / "./assets/bone.png"
