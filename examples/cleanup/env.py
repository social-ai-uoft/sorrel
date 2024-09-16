# --------------------------------- #
# region: Imports                   #
# --------------------------------- #

# Import base packages
import jax.numpy as jnp
import numpy as np
import random

# Import gem-specific packages
from gem.primitives import GridworldEnv
from examples.cleanup.entities import (
    EmptyObject,
    Wall,
    River,
    Pollution,
    AppleTree,
    Apple
)
from examples.cleanup.agents import (
    Agent
)
from examples.trucks.config import Cfg

# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #

class Cleanup(GridworldEnv):
  """Cleanup Environment."""
  def __init__(
    self,
    cfg: Cfg,
    agents: list[Agent]
  ):
    self.cfg = cfg
    self.channels = cfg.env.channels # default: # of entity classes + 1 (agent class) + 2 (beam types)
    self.full_mdp = cfg.env.full_mdp
    self.agents = agents
    self.appearances = self.color_map(self.channels)
    self.object_layer = 0
    self.agent_layer = 1
    self.beam_layer = 2
    super().__init__(cfg.env.height, cfg.env.width, cfg.env.layers, eval(cfg.env.default_object)(cfg, self.appearances['EmptyObject']))
    self.populate()

  def color_map(self, C) -> dict:
    """Color map for visualization."""
    if C == 8:
      colors = {
        'EmptyObject': [0 for _ in range(self.channels)],
        'Agent': [255 if x == 0 else 0 for x in range(self.channels)],
        'Wall': [255 if x == 1 else 0 for x in range(self.channels)],
        'Apple': [255 if x == 2 else 0 for x in range(self.channels)],
        'AppleTree': [255 if x == 3 else 0 for x in range(self.channels)],
        'River': [255 if x == 4 else 0 for x in range(self.channels)],
        'Pollution': [255 if x == 5 else 0 for x in range(self.channels)],
        'CleanBeam': [255 if x == 6 else 0 for x in range(self.channels)],
        'ZapBeam': [255 if x == 7 else 0 for x in range(self.channels)]
      }
    else:
      colors = {
        'EmptyObject': [0.0, 0.0, 0.0],
        'Agent': [150.0, 150.0, 150.0],
        'Wall': [50.0, 50.0, 50.0],
        'Apple': [0.0, 200.0, 0.0],
        'AppleTree': [100.0, 200.0, 0.0],
        'River': [0.0, 0.0, 200.0],
        'Pollution': [0, 100.0, 200.0],
        'CleanBeam': [200.0, 255.0, 200.0],
        'ZapBeam': [255.0, 200.0, 200.0]
      }
    return colors
  
  def populate(self) -> None:
  
    spawn_points = []
        
    # First, create the walls
    for index in np.ndindex(self.world.shape):
      H, W, L = index

      # If the index is the first or last, replace the location with a wall
      if H in [0, self.height - 1] or W in [0, self.width - 1]:
        self.world[index] = Wall(self.cfg, self.appearances["Wall"])
      # Define river, orchard, and potential agent spawn points
      elif L == 0:
        # Top third = river
        if H > 0 and H < (self.height // 3):
          self.world[index] = River(self.cfg, self.appearances["River"])
        # Bottom third = orchard
        elif H > (self.height - 1 - (self.height // 3)) and H < (self.height - 1):
          self.world[index] = AppleTree(self.cfg, self.appearances["AppleTree"])
        # Middle third = potential agent spawn points
        else:
          spawn_points.append(index)

      
    # Place agents randomly based on the spawn points chosen
    loc_index = np.random.choice(len(spawn_points), size = len(self.agents), replace = False)
    locs = [spawn_points[i] for i in loc_index]
    for loc, agent in zip(locs, self.agents):
      loc = tuple(loc)
      self.world[loc] = agent
      agent.location = loc

    

      



      
    


  
