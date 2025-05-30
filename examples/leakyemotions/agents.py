"The agents for the leaky emotions project."

### Imports
from pathlib import Path
import numpy as np

from sorrel.agents import Agent
from sorrel.models import base_model
from sorrel.environments.gridworld import GridworldEnv
from sorrel.observation import visual_field
###

class LeakyEmotionAgent(Agent):
    """An agent that perceives wolves, bushes, and other agents in the environment."""

    def __init__(self, observation_spec, action_spec, model: base_model.SorrelModel, location: tuple | None):
        super().__init__(observation_spec, action_spec, model, location)
        self.kind = "Agent"
        self.encounters = {}
        self.sprite = Path(__file__).parent / "./assets/leakyemotionagent.png"
    
    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        state = np.zeros_like(np.prod(self.model.input_size))
        action = 0
        reward = 0.0
        done = False
        for i in range(self.model.num_frames):
            self.add_memory(state, action, reward, done)
    
    def pov(self, env: GridworldEnv) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(env, self.location)
        # Flatten the image to get the state
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        prev_states = self.model.memory.current_state(
            stacked_frames=self.model.num_frames - 1
        )
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    def act(self, env: GridworldEnv, action: int) -> float:
        """Act on the environment, returning the reward."""

        # Translate the model output to an action string
        action = self.action_spec.get_readable_action(action)

        new_location = self.location
        if action == "up":
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == "down":
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == "left":
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == "right":
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        target_objects = env.observe_all_layers(new_location)

        reward = 0
        
        for target_object in target_objects:
            reward += target_object.value
            
            if target_object.kind not in self.encounters.keys():
                self.encounters[target_object.kind] = 0
            
            self.encounters[target_object.kind] += 1

            if target_object.kind == "Bush":
                env.num_bushes_eaten = self.encounters[target_object.kind]
                env.bush_ripeness_total += target_object.ripeness

        env.game_score += reward

        # try moving to new_location
        env.move(self, new_location)

        return reward
    
    def is_done(self, env: GridworldEnv) -> bool:
        """Returns whether this Agent is done."""
        return env.turn >= env.max_turns

class Wolf(Agent):
    """An entity that represents a wolf in the leakyemotions environment."""
    def __init__(self, observation_spec, action_spec, model: base_model.SorrelModel, location: tuple | None):
        super().__init__(observation_spec, action_spec, model, location)
        self.value = 0
        self.sprite = Path(__file__).parent / "./assets/wolfagent.png"
        self.kind = "Wolf"

    def get_action(self, state: np.ndarray) -> int:
        """
        Chases the nearest agent with a deterministic action policy. 
        Previously called the chase() function.

        Parameters:
            state: array of all the entities in the environment 

        Returns:
            int: The action to take.
        """
        # Get all agent targets
        targets = []
        
        for _, x in np.ndenumerate(state):
            if x.kind == "Agent":
                targets.append(x)

        # Get locations of all agents
        target_locations = []
        for target in targets:
            target_locations.append(target.location)

        # Compute distances ~ an array of taxicab distances from the wolf to each agent 
        distances = self.compute_taxicab_distance(self.location, target_locations)

        # Choose an agent with the minimum distance to the wolf.
        min_locs = np.where(distances == distances.min())[0]
        chosen_agent = targets[np.random.choice(min_locs)]

        # Compute possible paths
        ACTIONS = [0, 1, 2, 3, 4]
        TOO_FAR = 999999999
        attempted_paths = [self.movement(action) for action in ACTIONS]
        paths = self.compute_taxicab_distance(chosen_agent.location, attempted_paths)
        candidate_paths = np.array([paths[action] for action in ACTIONS])

        # Choose a candidate action that minimizes the taxicab distance
        candidate_actions = np.where(candidate_paths == candidate_paths.min())[0]
        chosen_action = np.random.choice(candidate_actions)

        return chosen_action
    
    def pov(self, env: GridworldEnv) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(env, self.location)
        # Flatten the image to get the state
        return image.reshape(1, -1)
    
    @staticmethod
    def compute_taxicab_distance(location, targets: list[tuple]) -> np.array:
        """
        Computes taxicab distance between one location and a list of other locations.

        Parameters:
            targets: A list of locations.
        
        Returns:
            np.array: The taxicab distance between the wolf and each agent
        """

        distances = []
        # Get taxicab distance for each agent in the list
        for target in targets:
            distance = sum([abs(x - y) for x, y in zip(location, target)])
            distances.append(distance)

        return np.array(distances)
    
    def movement(self, action: int) -> tuple:
        """
        Takes an action and returns the location the agent would end up at if it chose that action.

        Parameters:
            action (int): Action to take.

        Returns:
            tuple: New location after the action in the form (x, y, z).
        """
        location = self.location   
        if action == 0:
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        elif action == 1:
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        elif action == 2:
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        elif action == 3:
            new_location = (self.location[0], self.location[1] + 1, self.location[2])
        elif action == 4:
            new_location = self.location
        
        return new_location
    
    def act(self, env: GridworldEnv, action: int):
        """Move to the location computed by the chase() function, and decrease LeakyEmotionAgent's score if the wolf overlaps with it.
        
        Previously called the hunt() function.
        """
        action = self.get_action(env.world)  ### TODO: Figure out where this function is repeatedly called and add this call somewhere so we get the int for action parameter

        if action == 0:
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        elif action == 1:
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        elif action == 2:
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        elif action == 3:
            new_location = (self.location[0], self.location[1] + 1, self.location[2])
        elif action == 4:
            new_location = self.location

        # decrease entity's value at new_location if it is a rabbit, otherwise do nothing 
        target_object = env.observe(new_location)
        
        if target_object.kind == "LeakyEmotionAgent":
            target_object.value -= 1

        # try moving to new_location
        if env.move(self, new_location) == True:
            self.location = new_location  

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        state = np.zeros_like(np.prod(self.model.input_size))
        action = 0
        reward = 0.0
        done = False
        for i in range(self.model.num_frames):
            self.add_memory(state, action, reward, done)

    def is_done(self, env: GridworldEnv) -> bool:
        """Returns whether this Agent is done."""
        return env.turn >= env.max_turns

# class Wolf(Agent):
#     """
#     Represents a wolf agent in the environment.
#     """

#     def __init__(self, observation_spec, action_spec, model:GridworldEnv, location: tuple | None):
#         """
#         Initializes a Wolf object.

#         Parameters:
#             cfg: Configuration for the wolf.
#             model (_Model): The model used by the wolf.
#             location (tuple): The initial location of the wolf. Parameter is optional, and often set by the env.spawn method.
#         """
#         super().__init__(observation_spec, action_spec, model, location)
#         self.sprite = Path(__file__).parent / "./assets/wolfagent.png"

#     def reset(self) -> None:
#         """Resets the agent by fill in blank images for the memory buffer."""
#         state = np.zeros_like(np.prod(self.model.input_size))
#         action = 0
#         reward = 0.0
#         done = False
#         for i in range(self.model.num_frames):
#             self.add_memory(state, action, reward, done)
    
#     def pov(self, env: GridworldEnv) -> np.ndarray:
#         """Returns the state observed by the agent, from the flattened visual field."""
#         image = self.observation_spec.observe(env, self.location)
#         # Flatten the image to get the state
#         return image.reshape(1, -1)

#     def get_action(self, state: np.ndarray) -> int:
#         """Gets the action from the model, using the stacked states."""
#         prev_states = self.model.memory.current_state(
#             stacked_frames=self.model.num_frames - 1
#         )
#         stacked_states = np.vstack((prev_states, state))

#         model_input = stacked_states.reshape(1, -1)
#         action = self.model.take_action(model_input)
#         return action

#     def act(self, env: GridworldEnv, action: int) -> float:
#         """Act on the environment, returning the reward."""

#         # Translate the model output to an action string
#         action = self.action_spec.get_readable_action(action)

#         new_location = self.location
#         if action == "up":
#             new_location = (self.location[0] - 1, self.location[1], self.location[2])
#         if action == "down":
#             new_location = (self.location[0] + 1, self.location[1], self.location[2])
#         if action == "left":
#             new_location = (self.location[0], self.location[1] - 1, self.location[2])
#         if action == "right":
#             new_location = (self.location[0], self.location[1] + 1, self.location[2])

#         # get reward obtained from object at new_location
#         target_object = env.observe(new_location)
#         reward = target_object.value
#         env.game_score += reward

#         # try moving to new_location
#         env.move(self, new_location)

#         return reward
    
#     def chase(self, env: GridworldEnv) -> int:
#         """
#         Chases the nearest agent with a deterministic action policy.

#         Parameters:
#             env: The environment object.

#         Returns:
#             int: The action to take.
#         """
#         # Get all agent targets
#         targets = env.get_entities_of_kind("agent")

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
#         candidate_paths = np.array([paths[action] if env.world.is_valid_location(attempted_paths[action]) else TOO_FAR for action in ACTIONS])

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
#         # Translate the model output to an action string
#         action = self.action_spec.get_readable_action(action)

#         location = self.location
#         if action == "up":
#             new_location = (self.location[0] - 1, self.location[1], self.location[2])
#         if action == "down":
#             new_location = (self.location[0] + 1, self.location[1], self.location[2])
#         if action == "left":
#             new_location = (self.location[0], self.location[1] - 1, self.location[2])
#         if action == "right":
#             new_location = (self.location[0], self.location[1] + 1, self.location[2])
        
#         return new_location

#     def is_done(self, env: GridworldEnv) -> bool:
#         """Returns whether this Agent is done."""
#         return env.turn >= env.max_turns
    
    