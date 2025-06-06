"""The entities for the leaky emotions project."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.environments import GridworldEnv

# end imports

# class Wolf(Entity):
#     """An entity that represents a wolf in the leakyemotions environment."""
#     def __init__(self, location=None):
#         super().__init__()
#         self.value = 0
#         self.has_transitions = True
#         self.location = None
#         self.sprite = Path(__file__).parent / "./assets/wolfagent.png"
#         self.kind = "Wolf"

#     def chase(self, env: GridworldEnv) -> int:
#         """
#         Chases the nearest agent with a deterministic action policy.

#         Parameters:
#             env: The environment object.

#         Returns:
#             int: The action to take.
#         """
#         # Get all agent targets
#         targets = env.get_entities_of_kind("Agent")

#         # Get locations of all agents
#         target_locations = []
#         for target in targets:
#             target_locations.append(target.location)

#         # Compute distances ~ an array of taxicab distances from the wolf to each agent 
#         distances = self.compute_taxicab_distance(self.location, target_locations)

#         # Choose an agent with the minimum distance to the wolf.
#         min_locs = np.where(distances == distances.min())[0]
#         chosen_agent = targets[np.random.choice(min_locs)]

#         # Compute possible paths
#         ACTIONS = [0, 1, 2, 3]
#         TOO_FAR = 999999999
#         attempted_paths = [self.movement(action) for action in ACTIONS]
#         paths = self.compute_taxicab_distance(chosen_agent.location, attempted_paths)
#         candidate_paths = np.array([paths[action] for action in ACTIONS])

#         # Choose a candidate action that minimizes the taxicab distance
#         candidate_actions = np.where(candidate_paths == candidate_paths.min())[0]
#         chosen_action = np.random.choice(candidate_actions)

#         return chosen_action
    
#     @staticmethod
#     def compute_taxicab_distance(location, targets: list[tuple]) -> np.array:
#         """
#         Computes taxicab distance between one location and a list of other locations.

#         Parameters:
#             targets: A list of locations.
        
#         Returns:
#             np.array: The taxicab distance between the wolf and each agent
#         """

#         distances = []
#         # Get taxicab distance for each agent in the list
#         for target in targets:
#             distance = sum([abs(x - y) for x, y in zip(location, target)])
#             distances.append(distance)

#         return np.array(distances)
    
#     def movement(self, action: int) -> tuple:
#         """
#         Takes an action and returns the location the agent would end up at if it chose that action.

#         Parameters:
#             action (int): Action to take.

#         Returns:
#             tuple: New location after the action in the form (x, y, z).
#         """
#         location = self.location   
#         if action == 0:
#             new_location = (self.location[0] - 1, self.location[1], self.location[2])
#         if action == 1:
#             new_location = (self.location[0] + 1, self.location[1], self.location[2])
#         if action == 2:
#             new_location = (self.location[0], self.location[1] - 1, self.location[2])
#         if action == 3:
#             new_location = (self.location[0], self.location[1] + 1, self.location[2])
        
#         return new_location
    
#     def transition(self, env: GridworldEnv):
#         """Move to the location computed by the chase() function, and decrease LeakyEmotionAgent's score if the wolf overlaps with it."""

#         action = self.chase(env)

#         if action == 0:
#             new_location = (self.location[0] - 1, self.location[1], self.location[2])
#         elif action == 1:
#             new_location = (self.location[0] + 1, self.location[1], self.location[2])
#         elif action == 2:
#             new_location = (self.location[0], self.location[1] - 1, self.location[2])
#         elif action == 3:
#             new_location = (self.location[0], self.location[1] + 1, self.location[2])

#         # decrease entity's value at new_location if it is a rabbit, otherwise do nothing 
#         target_object = env.observe(new_location)
        
#         if target_object.kind == "LeakyEmotionAgent":
#             target_object.value -= 1

#         # try moving to new_location
#         if env.move(self, new_location) == True:
#             self.location = new_location  

class Bush(Entity):
    """An entity that represents a bush in the leakyemotions environment."""   

    def __init__(self, location=None, ripe_num=0):
        super().__init__()
        self.value = 1 
        self.ripeness = ripe_num
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/bush.png"
        self.kind="Bush"
    
    def transition(self, env: GridworldEnv):
        self.ripeness += 1
        if self.ripeness > 14:
            env.remove(self.location)
            env.add(self.location, Grass())
        else:
            self.determine_value()
        return env
    
    def determine_value(self):
        values = [0.1, 0.3, 0.5, 0.9, 2, 3, 5, 5, 5, 3, 2, 0.9, 0.5, 0.3, 0.1]
        self.value = values[self.ripeness] * self.ripeness  # Multiplier function 

class Wall(Entity):
    """An entity that represents a wall in the leakyemotions environment."""
    
    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = Path(__file__).parent / "./assets/wall.png"

class Grass(Entity):
    """An entity that represents a block of grass in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
        # We technically don't need to make Grass passable here since it's on a different layer from Agent
        self.passable = True
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/grass.png"

    def transition(self, env: GridworldEnv):
        """Grass can randomly spawn into Bushes based on the item spawn probabilities dictated in the evironment."""
    
        if (  # NOTE: If the spawn prob is too high, the environment gets overrun
            np.random.random() < env.spawn_prob
        ):
            env.add(self.location, Bush())

class EmptyEntity(Entity):
    """An entity that represents an empty space in the leakyemotions environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces
        self.has_transitions = True  # EmptyEntity can transition into Bushes
        self.sprite = Path(__file__).parent / "./assets/empty.png"
