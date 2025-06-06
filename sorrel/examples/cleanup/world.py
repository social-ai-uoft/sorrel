# --------------------------------- #
# region: Imports                   #
# --------------------------------- #

# Import base packages
import numpy as np

# Import sorrel-specific packages
from sorrel.entities import Entity
from sorrel.worlds import Gridworld

# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #


class CleanupWorld(Gridworld):
    """Cleanup world."""

    def __init__(self, config, default_entity: Entity):
        self.config = config
        self.channels = (
            config.agent.agent.obs.channels
        )  # default: # of entity classes + 1 (agent class) + 2 (beam types)
        self.full_mdp = config.env.full_mdp
        self.object_layer = 0
        self.agent_layer = 1
        self.beam_layer = 2
        self.pollution = 0
        super().__init__(
            config.env.height, config.env.width, config.env.layers, default_entity
        )
        self.mode = config.env.mode
        self.max_turns = config.experiment.max_turns
        self.pollution_threshold = config.env.pollution_threshold
        self.pollution_spawn_chance = config.env.pollution_spawn_chance
        self.apple_spawn_chance = config.env.apple_spawn_chance
        self.initial_apples = config.env.initial_apples
        self.turn = 0

    def measure_pollution(self) -> float:
        pollution_tiles = 0
        river_tiles = 0
        for index, x in np.ndenumerate(self.map):
            x: Entity
            if x.kind == "Pollution":
                pollution_tiles += 1
                river_tiles += 1
            elif x.kind == "River":
                river_tiles += 1
        return pollution_tiles / river_tiles
