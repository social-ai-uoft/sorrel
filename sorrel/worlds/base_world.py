from abc import abstractmethod

from sorrel.entities.entity import Entity

class World:

  @abstractmethod
  def add(self, target_location, entity) -> None:
    ...

  @abstractmethod
  def remove(self, target_location) -> Entity:
    ...

  @abstractmethod
  def move(self, entity, new_location) -> bool:
    ...
  
  @abstractmethod
  def observe(self, target_location) -> Entity | list[Entity]:
    ...