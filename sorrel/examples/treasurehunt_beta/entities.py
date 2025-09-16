"""The entities for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.treasurehunt.world import TreasurehuntWorld

# end imports


class Wall(Entity[TreasurehuntWorld]):
    """An entity that represents a wall in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Sand(Entity[TreasurehuntWorld]):
    """An entity that represents a block of sand in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
        # We technically don't need to make Sand passable here since it's on a different layer from Agent
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"


class Gem(Entity[TreasurehuntWorld]):
    """An entity that represents a gem in the treasurehunt environment."""

    def __init__(self, gem_value):
        super().__init__()
        self.passable = True  # Agents can move onto Gems
        self.value = gem_value
        self.sprite = Path(__file__).parent / "./assets/gem.png"


class Apple(Entity[TreasurehuntWorld]):
    """An entity that represents an apple in the treasurehunt environment."""

    def __init__(self, apple_value):
        super().__init__()
        self.passable = True  # Agents can move onto Apples
        self.value = apple_value
        self.sprite = Path(__file__).parent / "../cleanup/assets/apple.png"


class Coin(Entity[TreasurehuntWorld]):
    """An entity that represents a coin in the treasurehunt environment."""

    def __init__(self, coin_value):
        super().__init__()
        self.passable = True  # Agents can move onto Coins
        self.value = coin_value
        self.sprite = Path(__file__).parent / "./assets/gem.png"  # Reuse gem sprite for coin


class Crystal(Entity[TreasurehuntWorld]):
    """An entity that represents a crystal in the treasurehunt environment."""

    def __init__(self, crystal_value):
        super().__init__()
        self.passable = True  # Agents can move onto Crystals
        self.value = crystal_value
        self.sprite = Path(__file__).parent / "../staghunt/assets/gem.png"  # Use different gem sprite


class Treasure(Entity[TreasurehuntWorld]):
    """An entity that represents a treasure chest in the treasurehunt environment."""

    def __init__(self, treasure_value):
        super().__init__()
        self.passable = True  # Agents can move onto Treasures
        self.value = treasure_value
        self.sprite = Path(__file__).parent / "./assets/hero.png"  # Use hero sprite as placeholder


class EmptyEntity(Entity[TreasurehuntWorld]):
    """An entity that represents an empty space in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces
        self.has_transitions = True  # EmptyEntity can transition into Gems
        self.sprite = Path(__file__).parent / "./assets/empty.png"

    def transition(self, world: TreasurehuntWorld):
        """EmptySpaces can randomly spawn into various resources based on the item spawn
        probabilities dictated in the environment."""
        if (  # NOTE: If the spawn prob is too high, the environment gets overrun
            np.random.random() < world.spawn_prob
        ):
            # Randomly choose which resource to spawn
            resource_type = np.random.choice([
                "gem", "apple", "coin", "crystal", "treasure"
            ])
            
            if resource_type == "gem":
                world.add(self.location, Gem(world.gem_value))
            elif resource_type == "apple":
                world.add(self.location, Apple(world.apple_value))
            elif resource_type == "coin":
                world.add(self.location, Coin(world.coin_value))
            elif resource_type == "crystal":
                world.add(self.location, Crystal(world.crystal_value))
            elif resource_type == "treasure":
                world.add(self.location, Treasure(world.treasure_value))
