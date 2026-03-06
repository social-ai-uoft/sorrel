"""Custom entity classes for the Cooking example.

This module defines a minimal hierarchy of entity types that model kitchen stations and
ingredients.  Only the attributes required by the test suite and the environment logic
are implemented.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

from sorrel.entities import Entity


# ---------------------------------------------------------------------------
# Base station entity
# ---------------------------------------------------------------------------
class StationEntity(Entity[object]):
    """Base class for kitchen stations (stove, counter, plate, trash).

    Subclasses can define additional state such as held items or cooking timers.  All
    stations are unpassable, i.e. agents cannot move onto the same tile.
    """

    def __init__(self):
        super().__init__()
        self.passable = False
        # Placeholder sprite – not used in tests.
        self.sprite = Path(__file__).parent / "assets" / "station.png"
        # ``held`` stores a reference to an ``IngredientEntity`` or ``None``.
        self.held: Optional[IngredientEntity] = None

    @abstractmethod
    def place(self, ingredient: IngredientEntity) -> bool:
        """Place an ingredient on the station if empty.

        Args:
            ingredient (IngredientEntity): The ingredient to place on the stove.

        Returns:
            bool: Whether the ingredient was placed successfully.
        """
        ...

    @abstractmethod
    def take(self) -> Optional[IngredientEntity]:
        """Take an ingredient from the station if possible.

        Returns:
            Entity | None: If possible, an ingredient.
        """
        ...


# ---------------------------------------------------------------------------
# Specific stations
# ---------------------------------------------------------------------------
class Stove(StationEntity):
    """A cooking station that can hold a single ingredient and cook it.

    Attributes:
        cook_time (int): the number of turns required before the ingredient is considered cooked.
        timer (int): the remaining turns before a current item is cooked.
        is_busy (bool): whether the stove is in use.
    """

    def __init__(self, cook_time: int = 1):
        super().__init__()
        self.cook_time = cook_time
        self.timer = 0
        self.is_busy = False
        self.sprite = Path(__file__).parent / "assets" / "stove.png"

    def transition(self, world):
        """Decrement the cooking timer each turn if an ingredient is present."""
        if self.held is not None and self.timer > 0:
            self.timer -= 1
        # When timer reaches zero the ingredient is considered cooked.
        if self.held is not None and self.timer == 0:
            self.held.cooked = True

    def place(self, ingredient: IngredientEntity) -> bool:
        if self.held is None:
            self.held = ingredient
            self.timer = self.cook_time
            self.is_busy = True
            self.sprite = Path(__file__).parent / "assets" / "stove-item.png"
            return True
        return False

    def take(self) -> Optional[IngredientEntity]:
        """Remove the ingredient from the stove (cooked or not)."""
        if self.held is not None:
            ing = self.held
            self.held = None
            self.timer = 0
            self.is_busy = False
            self.sprite = Path(__file__).parent / "assets" / "stove.png"
            return ing
        return None


class Counter(StationEntity):
    """A simple storage counter that can hold a single ingredient."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "assets" / "counter.png"

    def place(self, ingredient: IngredientEntity) -> bool:
        if self.held is None:
            self.held = ingredient
            self.sprite = Path(__file__).parent / "assets" / "counter-item.png"
            return True
        return False

    def take(self) -> Optional[IngredientEntity]:
        if self.held is not None:
            ing = self.held
            self.held = None
            self.sprite = Path(__file__).parent / "assets" / "counter.png"
            return ing
        return None


class Plate(StationEntity):
    """A plate can combine multiple cooked ingredients into a dish.

    Attributes:
        contents:
    """

    def __init__(self):
        super().__init__()
        self.contents: List[IngredientEntity] = []
        self.sprite = Path(__file__).parent / "assets" / "plate.png"

    def place(self, ingredient: IngredientEntity) -> bool:
        # Accept only cooked ingredients (have ``cooked`` attribute set).
        if ingredient.cooked:
            self.contents.append(ingredient)
            self.sprite = Path(__file__).parent / "assets" / "plate-item.png"
            return True
        return False

    def take(self) -> Optional[IngredientEntity]:
        item = None
        if self.contents:
            item = self.contents.pop()
        if not self.contents:
            self.sprite = Path(__file__).parent / "assets" / "plate.png"
        return item

    def take_all(self) -> List[IngredientEntity]:
        items = self.contents.copy()
        self.contents.clear()
        self.sprite = Path(__file__).parent / "assets" / "plate.png"
        return items


class Trash(StationEntity):
    """A trash bin clears any held item when an agent interacts with it."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "assets" / "trash.png"

    def place(self, ingredient: Entity) -> bool:
        # Discard the ingredient – nothing to store.
        return True

    def take(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Ingredient entities
# ---------------------------------------------------------------------------
class IngredientEntity(Entity[object]):
    """Base class for raw ingredients.

    Subclasses should set ``name`` and ``cookable``.  The ``cooked`` flag is
    added by ``Stove.transition`` when cooking completes.
    """

    def __init__(self, name: str, cookable: bool = True):
        super().__init__()
        self.name = name
        self.kind = name
        self.cookable = cookable
        self.cooked = False
        self.passable = True
        self.sprite = Path(__file__).parent / "assets" / "ingredient.png"


# Concrete ingredient examples used in unit tests
class Onion(IngredientEntity):
    def __init__(self):
        super().__init__(name="Onion", cookable=True)


class Tomato(IngredientEntity):
    def __init__(self):
        super().__init__(name="Tomato", cookable=True)


# Re‑use the EmptyEntity from the cleanup example for the default grid cell.
from sorrel.examples.cleanup.entities import EmptyEntity
