import random

from agentarium.primitives import Entity, GridworldEnv

# --------------------------------------------------- #
# region: Environment Entity classes for Cleanup Task #
# --------------------------------------------------- #

class EmptyEntity(Entity):
  """Empty Entity class for the Cleanup Game."""
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.passable = True
    self.sprite = f'{cfg.root}/examples/cleanup/assets/empty.png'

class Sand(Entity):
  """Sand class for the Cleanup Game."""
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.passable = True
    self.sprite = f'{cfg.root}/examples/cleanup/assets/sand.png'
    # Overwrite: Sand is just a different sprite appearance for
    # the EmptyEntity class, but is otherwise identical.
    self.kind = 'EmptyEntity'

class Wall(Entity):
  """Wall class for the Cleanup Game."""
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.sprite = f'{cfg.root}/examples/cleanup/assets/wall.png'

class River(Entity):
  """River class for the Cleanup game."""
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.has_transitions = True
    self.sprite = f'{cfg.root}/examples/cleanup/assets/water.png'

  def transition(self, env: GridworldEnv):
    # Add pollution with a random probability
    if random.random() < self.cfg.env.pollution_spawn_chance:
      env.add(self.location, Pollution(self.cfg))

class Pollution(Entity):
  """Pollution class for the Cleanup game."""
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.has_transitions = True
    self.sprite = f'{cfg.root}/examples/cleanup/assets/pollution.png'

  def transition(self, env: GridworldEnv):
    # Check the current tile on the beam layer for cleaning beams
    beam_location = self.location[0], self.location[1], env.beam_layer
    
    # If a cleaning beam is on this tile, spawn a river tile
    if env.observe(beam_location).kind == "CleanBeam":
      env.add(self.location, River(self.cfg))


class AppleTree(Entity):
  """Potential apple class for the Cleanup game."""
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.has_transitions = True
    self.sprite = f'{cfg.root}/examples/cleanup/assets/grass.png'

  def transition(self, env: GridworldEnv):
    # If the pollution threshold has not been reached...
    if not env.pollution > self.cfg.env.pollution_threshold:
      # Add apples with a random probability
      if random.random() < self.cfg.env.apple_spawn_chance:
        env.add(self.location, Apple(self.cfg))

class Apple(Entity):
  """Apple class for the Cleanup game."""
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.value = 1 # Reward for eating the apple
    self.sprite = f'{cfg.root}/examples/cleanup/assets/apple_grass.png'

  def transition(self, env: GridworldEnv):
    # Check the current tile on the agent layer for agents
    agent_location = self.location[0], self.location[1], env.agent_layer
    
    # If there is an agent on this tile, spawn an apple tree tile
    if env.observe(agent_location).kind == "CleanupAgent":
      env.add(self.location, AppleTree(self.cfg))

# --------------------------------------------------- #
# endregion                                           #
# --------------------------------------------------- #


