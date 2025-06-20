"The agents for the leaky emotions project."

### Imports
from pathlib import Path
import numpy as np
import random
from math import tanh

from sorrel.agents import Agent
from sorrel.models import base_model
from sorrel.worlds.gridworld import Gridworld
from sorrel.examples.leakyemotions.world import LeakyEmotionsWorld
###

class LeakyEmotionsAgent(Agent[LeakyEmotionsWorld]):
    """An agent that perceives wolves, bushes, and other agents in the environment."""

    def __init__(self, observation_spec, action_spec, model: base_model.BaseModel, location: tuple | None = None):
        super().__init__(observation_spec, action_spec, model, location)
        self.encounters = {}
        self.passable = False
        self.sprite = Path(__file__).parent / "./assets/leakyemotionagent.png"
        self.id = 0
        self.alive = True
        self.emotion = 0.
    
    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        self.alive = True
        return self.model.reset()
    
    def pov(self, world: Gridworld) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(world, self.location)
        # Flatten the image to get the state
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        if not hasattr(self.model, "name"):

            # Update the agent emotion.
            self.update_emotion(state)
            
            # Stack previous frames as needed.
            prev_states = self.model.memory.current_state()
            stacked_states = np.vstack((prev_states, state)) if prev_states else state

            # Take action
            model_input = stacked_states.reshape(1, -1)
            action = self.model.take_action(model_input)
        
        else:
            action = self.model.take_action(state)
        
        return action
    
    def update_emotion(self, state: np.ndarray) -> None:
        """Update the agent's emotion based on its state value approximation.
        
        Args:
            state: The observed input.
        """
        self.emotion = self.model.state_value(state)

    def act(self, world: LeakyEmotionsWorld, action: int) -> float:
        """Act on the environment, returning the reward."""

        # Translate the model output to an action string
        action_name = self.action_spec.get_readable_action(action)

        new_location = self.location
        if action_name == "up":
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action_name == "down":
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action_name == "left":
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action_name == "right":
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        target_objects = world.observe_all_layers(new_location)

        reward = 0
        
        for target_object in target_objects:
            reward += target_object.value
            
            if target_object.kind not in self.encounters.keys():
                self.encounters[target_object.kind] = 0
            
            self.encounters[target_object.kind] += 1

            if target_object.kind == "Bush":
                world.num_bushes_eaten = self.encounters[target_object.kind]
                world.bush_ripeness_total += target_object.ripeness #type: ignore

        # try moving to new_location
        world.move(self, new_location)

        return reward
    
    def is_done(self, world: LeakyEmotionsWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done if self.alive else True

class Wolf(Agent):
    """An entity that represents a wolf in the leakyemotions environment."""
    def __init__(self, observation_spec, action_spec, model: base_model.BaseModel, location: tuple | None = None):
        super().__init__(observation_spec, action_spec, model, location)
        self.value = 0
        self.asleep = False 
        self.time_counter = 0
        self.sprite = Path(__file__).parent / "./assets/wolfagent.png"
        self.passable = False

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
            if x.kind == "LeakyEmotionsAgent":
                targets.append(x)

        # Get locations of all agents
        target_locations = []
        for target in targets:
            target_locations.append(target.location)

        if len(target_locations) > 0:

            # Compute distances ~ an array of taxicab distances from the wolf to each agent 
            distances = self.compute_taxicab_distance(self.location, target_locations)

            # Choose an agent with the minimum distance to the wolf.
            min_locs = np.where(distances == distances.min())[0]
            chosen_agent = targets[np.random.choice(min_locs)]

            # Compute possible paths
            ACTIONS = [0, 1, 2, 3]
            TOO_FAR = 999999999
            attempted_paths = [self.movement(action) for action in ACTIONS]
            paths = self.compute_taxicab_distance(chosen_agent.location, attempted_paths)
            candidate_paths = np.array([paths[action] for action in ACTIONS])

            # Choose a candidate action that minimizes the taxicab distance
            candidate_actions = np.where(candidate_paths == candidate_paths.min())[0]
            chosen_action = np.random.choice(candidate_actions)
        else:
            # Randomly move when no agents exist
            chosen_action = np.random.choice([0, 1, 2, 3])

        return chosen_action
    
    def pov(self, world: Gridworld) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        # Flatten the image to get the state
        return world.map.flatten()
    
    @staticmethod
    def compute_taxicab_distance(location, targets: list[tuple]) -> np.ndarray:
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
        
        new_location = self.location

        if action == 0:
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        elif action == 1:
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        elif action == 2:
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        elif action == 3:
            new_location = (self.location[0], self.location[1] + 1, self.location[2])
        
        return new_location
    
    def act(self, world: LeakyEmotionsWorld, action: int):
        """Move to the location computed by the chase() function, and decrease LeakyEmotionAgent's score if the wolf overlaps with it.
        
        Previously called the hunt() function.
        """
        self.sleep()
        if not self.asleep:
            new_location = self.movement(action)

            # decrease entity's value at new_location if it is a rabbit, otherwise do nothing 
            target_object = world.observe(new_location)
            
            if isinstance(target_object, LeakyEmotionsAgent):
                target_object.alive = False
                if world.num_agents == 0:
                    world.game_over()

                # Final memory for the dead agent, RIP
                target_object_state = target_object.pov(world)
                target_object_would_act = target_object.get_action(target_object_state)

                target_object.add_memory(
                    state=target_object_state, 
                    action=target_object_would_act,
                    reward=-100,
                    done=True
                )

                world.total_reward -= 100

                dead_agent = world.remove(new_location)
                world.dead_agents.append(dead_agent)         

                # try moving to new_location
                world.move(self, new_location)
        return 0.

    def sleep(self):
        '''
        This function serves as the agent's probablistic sleep-wake cycle.
        
            - wolf.asleep = whether the wolf is asleep
            - wolf.sleep_counter = how far the agent is in its cycle
        
        '''
        tmp = random.random()   # random float from 0 to 1, indicates random probability
        
        # Check if agent is asleep
        if self.asleep:
        
            # Wake up if the random value (tmp) is less than the calculated probability of waking up
            # The longer the wolf has slept, the more likely it is to wake up
    
            if tmp < 0.5*(tanh(self.time_counter - 5) + 1):
                # Agent wakes up, reset the time counter
                self.asleep = False
                self.time_counter = 0
            
            else:
                # Otherwise, stay asleep and increment the time counter
                self.time_counter += 1
        
        # Agent is awake
        else:

            # Fall asleep if the random value (tmp) is greater than the calculated probability of sleeping
            # The longer the wolf is awake, the more likely it is to fall asleep
            
            if tmp > 0.5*(tanh(-0.5*self.time_counter + 4) + 1):
                # Agent falls asleep, reset the time counter
                self.asleep = True
                self.time_counter = 0
            
            else:
                # Otherwise, stay awake and increment the time counter
                self.time_counter += 1

    def is_done(self, world: LeakyEmotionsWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
