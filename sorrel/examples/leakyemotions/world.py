"""The environment for the leakyemotions project."""

# begin imports

# Import base types
from sorrel.worlds import Gridworld

# end imports

class LeakyEmotionsWorld(Gridworld):
    """Leakyemotions project."""
    def __init__(self, config, default_entity):
        layers = 3  # walls, grass, agents (can change if needed)
        super().__init__(config.world.height, config.world.width, layers, default_entity)

        self.game_ended = False
        self.spawn_prob = config.world.spawn_prob
        self.num_agents = config.world.agents
        self.num_wolves = config.world.wolves
        self.agents = config.world.agents
        self.max_turns = config.experiment.max_turns

        self.game_score = 0
        
        self.bush_ripeness_total = 0
        self.num_bushes_eaten = 0
        self.dead_agents = []

    def game_over(self):
        if self.num_agents <= 0:
            print("Game Over.")
            self.game_ended = True
            # quit()
        
