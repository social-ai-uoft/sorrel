from examples.puppet_training.entities import Wall, EmptyObject
# from examples.trucks.utils import color_map

from ast import literal_eval as make_tuple
from typing import Optional
import torch
import numpy as np
import random

from agentarium.visual_field import visual_field
from agentarium.primitives import GridworldEnv

class Agent:
    def __init__(self, model, cfg, ixs):
        self.kind = "Agent"
        self.type = 'agent'
        self.cfg = cfg    
        # print(cfg.agent.agent.appearance)
        if isinstance(cfg.agent.agent.appearance, str):
            self.appearance = make_tuple(cfg.agent.agent.appearance) 
        else:
            self.appearance = cfg.agent.agent.appearance[ixs] # agents are blue # cfg.agent.agent.appearance[ixs]?        
        self.tile_size = make_tuple(cfg.agent.agent.tile_size)
        self.sprite = f'{cfg.root}/examples/puppet_training/assets/hero.png'
        self.passable = 0  # whether the object blocks movement
        self.value = 0  # agents have no value
        self.health = cfg.agent.agent.health  # for the agents, this is how hungry they are
        self.location = None
        self.action_space = [0, 1, 2, 3]
        self.vision = cfg.agent.agent.vision
        self.ixs = ixs
        self.extra_percept_size = cfg.agent.agent.extra_percept_size
        self.is_punished = 0.
        self.to_be_punished = {'Gem':0, 'Bone':0, 'Coin':0}
        self.transgression_record = []
        self.punishment_record = []

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


    def init_replay(self, env: GridworldEnv, state_mode: str) -> None:
        """Fill in blank images for the LSTM."""
        state = np.zeros_like(np.concatenate([self.pov(env), np.zeros(self.extra_percept_size)]))
        # print(state.shape, self.extra_percept_size)
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
                env.world, 
                color_map, 
                channels=env.channels, 
                has_value_map=self.cfg.has_value_map
            )
        # Otherwise, use the agent observation function
        else:
            assert self.kind == "Agent", "Agent must be of kind 'Agent'."
            if self.role == 'partner':
                # confirm partners not at the gate
                assert self.location != (int((env.height-1)/2), env.height, 0), "Partners cannot be at the gate."
            image = visual_field(
                world=env.world, 
                color_map=color_map, 
                location=self.location, 
                vision=self.vision, 
                channels=env.channels, 
                has_value_map=self.cfg.has_value_map,
                env=env
            )

        current_state = image.flatten()

        # append extra percept
     
        return current_state
        

    def movement(self,
                 action: int,
                 mode='simple'
                 ) -> tuple:
        
        '''
        Takes an action and returns a new location
        '''
        if action == 0: # UP
            self.sprite = f'{self.cfg.root}/examples/puppet_training/assets/hero-back.png'
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == 1: # DOWN
            self.sprite = f'{self.cfg.root}/examples/puppet_training/assets/hero.png'
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == 2: # LEFT
            self.sprite = f'{self.cfg.root}/examples/puppet_training/assets/hero-left.png'
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == 3: # RIGHT
            self.sprite = f'{self.cfg.root}/examples/puppet_training/assets/hero-right.png'
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        return new_location

    
    def transition(self,
                   env) -> tuple:
        '''
        Changes the world based on the action taken.

        when transgression is not apparent. 

        currently, all transgression records will be equally punished.
        '''

        # Get current state
        state = self.pov(env)

       
        model_input = torch.from_numpy(self.current_state(
            env=env, 
            )).view(1, -1)

        reward = 0

        # Take action based on current state
        action = self.model.take_action(model_input)
        # Attempt the transition 
        attempted_location = self.movement(action)
        target_object = env.observe(attempted_location)
        env.move(self, attempted_location)

        # Get the reward
        reward += self.value_dict[target_object.kind]
            
        # Add to the encounter record
        if str(target_object) in self.encounters.keys():
            self.encounters[str(target_object)] += 1 

        # Get the next state   
        next_state = self.pov(env)
                    
        return state, action, reward, next_state, False


    def generate_composite_state(self, envs):
        """
        Generate the stacked states. 
        """
        env_states = []
        for env in envs:
            env.full_mdp = False
            env_state = self.pov(env)
            env_states.append(env_state)
        template = np.zeros(env_states[0].shape)
        for i in range(6-len(envs)):       
            env_states.append(template)
        composite_state = np.concatenate(env_states)
        return composite_state

    def reset(self, env: GridworldEnv, state_mode='simple') -> None:
        self.init_replay(env, state_mode)
        self.encounters = {
            'Gem': 0,
            'Coin': 0,
            'Food': 0,
            'Bone': 0,
            'Wall': 0,
            'Agent': 0,
        }
    
    def reset_record(self) -> None:
        self.transgression_record = []
        self.punishment_record = []

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