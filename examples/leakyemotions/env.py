"""The environment for the leakyemotions project."""

# begin imports
# Import base packages
import numpy as np

# Import experiment specific classes
from examples.leakyemotions.entities import EmptyEntity, Wall, Grass
# Import primitive types
from sorrel.environments import GridworldEnv

# end imports

class LeakyemotionsEnv(GridworldEnv):
    """Leakyemotions project."""
    def __init__(self, height, width, bush_value, spawn_prob, max_turns, agents, bushes):
        layers = 4  # walls, grass, bushes, agents (can change if needed)
        default_entity = EmptyEntity()
        super().__init__(height, width, layers, default_entity)

        self.bush_value = bush_value
        self.spawn_prob = spawn_prob
        self.agents = agents
        self.max_turns = max_turns
        self.bushes = bushes

        self.game_score = 0
        self.populate()

    def populate(self):
        """
        Populate the leakyemotions world by creating walls, then randomly spawning the agents.
        Note that every space is already filled with EmptyEntity as part of super().__init__().
        """
        valid_agent_spawn_locations = []
        valid_bush_spawn_locations = []

        for index in np.ndindex(self.world.shape):
            y, x, z = index
            if y in [0, self.height - 1] or x in [0, self.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.add(index, Wall())
            elif z == 0:  # if location is on the bottom (first) layer, put grass there
                self.add(index, Grass())
            elif z == 1:  # if location is on second layer, then bushes can appear here 
                # valid bush location
                valid_bush_spawn_locations.append(index)
            elif z == 2: # if location is on third layer, agents can appear here (assuming that wolves and leakyemotionagents are on the same layer)
                # valid agent location 
                valid_agent_spawn_locations.append(index)


        # spawn the bushes
        # using np.random.choice, we choose indices in valid_bush_spawn_locations
        bush_locations_indices = np.random.choice(
            len(valid_bush_spawn_locations), size=len(self.bushes), replace=False
        )
        bush_locations = [valid_bush_spawn_locations[i] for i in bush_locations_indices]
        for loc, bush in zip(bush_locations, self.bushes):
            loc = tuple(loc)
            self.add(loc, bush)

        # TODO: need to call bush transition() function in a loop so ripeness keep incrementing until eventual removal

        # spawn the agents
        # using np.random.choice, we choose indices in valid_agent_spawn_locations
        agent_locations_indices = np.random.choice(
            len(valid_agent_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_agent_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.add(loc, agent)


    def reset(self):
        """Reset the environment and all its agents."""
        self.create_world()
        self.game_score = 0
        self.populate()
        for agent in self.agents:
            agent.reset() 
        
