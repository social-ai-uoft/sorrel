from examples.RPG.entities import Wall, EmptyObject
from examples.trucks.utils import color_map

from ast import literal_eval as make_tuple
from typing import Optional
import torch
import numpy as np

from agentarium.visual_field import visual_field
from agentarium.primitives import GridworldEnv

class Agent:
    def __init__(self, model, cfg):
        self.kind = "Agent"
        self.cfg = cfg      
        self.appearance = make_tuple(cfg.agent.agent.appearance)  # agents are blue
        self.tile_size = make_tuple(cfg.agent.agent.tile_size)
        self.sprite = f'{cfg.root}/examples/RPG/assets/hero.png'
        self.passable = 0  # whether the object blocks movement
        self.value = 0  # agents have no value
        self.health = cfg.agent.agent.health  # for the agents, this is how hungry they are
        self.location = None
        self.action_space = [0, 1, 2, 3]
        self.vision = cfg.agent.agent.vision
        
        # training-related features
        self.model = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        # self.episode_memory = Memory(cfg.agent.agent.memory_size)
        self.num_frames = cfg.agent.agent.num_memories
        self.init_rnn_state = None
        self.encounters = {
            'Gem': 0,
            'Coin': 0,
            'Food': 0,
            'Bone': 0,
            'Wall': 0
        }

    def init_replay(self, env: GridworldEnv) -> None:
        """Fill in blank images for the LSTM."""

        state = np.zeros_like(self.pov(env))
        action = 0  # Action outside the action space
        reward = 0.0
        done = 0.0
        for _ in range(self.num_frames):
            self.model.memory.add(state, action, reward, done)
    
    def add_memory(self, state: np.ndarray, action: int, reward: float, done: bool) -> None:
        """Add an experience to the memory."""
        self.model.memory.add(state, action, reward, float(done))
    
    def add_final_memory(self, state: np.ndarray) -> None:
        self.model.memory.add(state, 0, 0.0, float(True))

    def current_state(self, env: GridworldEnv) -> np.ndarray:
        state = self.pov(env)
        prev_states = self.model.memory.current_state(stacked_frames=self.num_frames-1)
        current_state = np.vstack((prev_states, state))
        return current_state

    def pov(self, env: GridworldEnv) -> np.ndarray:
        """
        Defines the agent's observation function
        """

        # If the environment is a full MDP, get the whole world image
        if env.full_mdp:
            image = visual_field(
                env.world, color_map, channels=env.channels
            )
        # Otherwise, use the agent observation function
        else:
            image = visual_field(
                env.world, color_map, self.location, self.vision, env.channels
            )

        current_state = image.flatten()

        return current_state
        
    def movement(self,
                 action: int
                 ) -> tuple:
        
        '''
        Takes an action and returns a new location
        '''
        if action == 0: # UP
            self.sprite = f'{self.cfg.root}/examples/RPG/assets/hero-back.png'
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == 1: # DOWN
            self.sprite = f'{self.cfg.root}/examples/RPG/assets/hero.png'
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == 2: # LEFT
            self.sprite = f'{self.cfg.root}/examples/RPG/assets/hero-left.png'
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == 3: # RIGHT
            self.sprite = f'{self.cfg.root}/examples/RPG/assets/hero-right.png'
            new_location = (self.location[0], self.location[1] + 1, self.location[2])
        return new_location
    
    def transition(self,
                   env) -> tuple:
        '''
        Changes the world based on the action taken.
        '''

        # Get current state
        state = self.pov(env)
        model_input = torch.from_numpy(self.current_state(env)).view(1, -1)
        reward = 0

        # Take action based on current state
        action = self.model.take_action(model_input)

        # Attempt the transition 
        attempted_location = self.movement(action)
        target_object = env.observe(attempted_location)
        env.move(self, attempted_location)

        # Get the interaction reward
        reward += target_object.value

        # Add to the encounter record
        if str(target_object) in self.encounters.keys():
            self.encounters[str(target_object)] += 1 

        # Get the next state   
        next_state = self.pov(env)

        return state, action, reward, next_state, False
        
    def reset(self, env: GridworldEnv) -> None:
        self.init_replay(env)
        self.encounters = {
            'Gem': 0,
            'Coin': 0,
            'Food': 0,
            'Bone': 0,
            'Wall': 0
        }

def color_map(channels: int) -> dict:
    '''
    Generates a color map for the food truck environment.

    Parameters:
        channels: the number of appearance channels in the environment

    Return:
        A dict of object-color mappings
    '''
    if channels > 5:
        colors = {
            'EmptyObject': [0 for _ in range(channels)],
            'Agent': [255 if x == 0 else 0 for x in range(channels)],
            'Wall': [255 if x == 1 else 0 for x in range(channels)],
            'Gem': [255 if x == 2 else 0 for x in range(channels)],
            'Food': [255 if x == 3 else 0 for x in range(channels)],
            'Coin': [255 if x == 4 else 0 for x in range(channels)],
            'Bone': [255 if x == 5 else 0 for x in range(channels)]
        }
    else:
        colors = {
            'EmptyObject': [255., 255., 255.],
            'Agent': [0., 0., 255.],
            'Wall': [153.0, 51.0, 102.0],
            'Gem': [0., 255., 0.],
            'Coin': [255., 255., 0.],
            'Food': [255., 0., 0.],
            'Bone': [0., 0., 0.]
        }
    return colors