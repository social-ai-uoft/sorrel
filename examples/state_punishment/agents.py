from examples.state_punishment.entities import Wall, EmptyObject
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
        self.cfg = cfg    
        # print(cfg.agent.agent.appearance)
        if isinstance(cfg.agent.agent.appearance, str):
            self.appearance = make_tuple(cfg.agent.agent.appearance) 
        else:
            self.appearance = cfg.agent.agent.appearance[ixs] # agents are blue # cfg.agent.agent.appearance[ixs]?        
        self.tile_size = make_tuple(cfg.agent.agent.tile_size)
        self.sprite = f'{cfg.root}/examples/state_punishment/assets/hero.png'
        self.passable = 0  # whether the object blocks movement
        self.value = 0  # agents have no value
        self.health = cfg.agent.agent.health  # for the agents, this is how hungry they are
        self.location = None
        self.action_space = [0, 1, 2, 3]
        self.vision = cfg.agent.agent.vision
        self.ixs = ixs
        self.extra_percept_size = cfg.agent.agent.extra_percept_size
        self.is_punished = 0.

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

        state = np.zeros_like(np.concatenate([self.pov(env), np.zeros(self.extra_percept_size)]))
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

    def current_state(self, state_sys, env: GridworldEnv) -> np.ndarray:
        state = self.pov(env)
        state = np.concatenate([state, np.array([state_sys.prob])])
        state = np.concatenate([state, np.array([self.is_punished])])
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
                 action: int,
                 state_sys
                 ) -> tuple:
        
        '''
        Takes an action and returns a new location
        '''
        if action == 0: # UP
            self.sprite = f'{self.cfg.root}/examples/state_punishment/assets/hero-back.png'
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == 1: # DOWN
            self.sprite = f'{self.cfg.root}/examples/state_punishment/assets/hero.png'
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == 2: # LEFT
            self.sprite = f'{self.cfg.root}/examples/state_punishment/assets/hero-left.png'
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == 3: # RIGHT
            self.sprite = f'{self.cfg.root}/examples/state_punishment/assets/hero-right.png'
            new_location = (self.location[0], self.location[1] + 1, self.location[2])
        if action == 4: # vote for state punishment
            self.sprite = self.sprite
            new_location = self.location
            state_sys.prob += state_sys.change_per_vote
            state_sys.prob = np.clip(state_sys.prob, 0, 1)
        if action == 5: # vote against state punishment
            self.sprite = self.sprite
            new_location = self.location
            state_sys.prob -= state_sys.change_per_vote
            state_sys.prob = np.clip(state_sys.prob, 0, 1)

        return new_location
    
    def transition(self,
                   env,
                   state_sys) -> tuple:
        '''
        Changes the world based on the action taken.
        '''

        # Get current state
        state = self.pov(env)
        state = np.concatenate([state, np.array([state_sys.prob])])
        state = np.concatenate([state, np.array([self.is_punished])])
        model_input = torch.from_numpy(self.current_state(state_sys=state_sys, env=env)).view(1, -1)

        
        # print(model_input.size())
        # ll
        # state_punishment_prob_tensor = torch.full((state.shape()[1], state.shape()[2]), state_sys.prob).view(1, -1)
        # model_input = torch.concat([model_input, state_punishment_prob_tensor])
        reward = 0

        # Take action based on current state
        action = self.model.take_action(model_input)

        # Attempt the transition 
        attempted_location = self.movement(action, state_sys)
        target_object = env.observe(attempted_location)
        env.move(self, attempted_location)

        # Get the interaction reward
        reward += target_object.value
        if self.is_punished == 1.:
            reward -= state_sys.magnitude 
        
        self.is_punished = 0. 

        # Get the delayed reward
        reward += env.cache['harm'][self.ixs]
        env.cache['harm'][self.ixs] = 0 
        # If the agent performs a transgression
        if str(target_object) in state_sys.taboo:
            # reward -= state_sys.magnitude * (random.random() < state_sys.prob)
            if random.random() < state_sys.prob_list[str(target_object)]:
                self.is_punished = 1.
            env.cache['harm'] = [env.cache['harm'][k] - target_object.social_harm 
                                        if k != self.ixs else env.cache['harm'][k]
                                        for k in range(len(env.cache['harm']))
                                        ]
            

        # Add to the encounter record
        if str(target_object) in self.encounters.keys():
            self.encounters[str(target_object)] += 1 

        # Get the next state   
        next_state = self.pov(env)
        next_state = np.concatenate([next_state, np.array([state_sys.prob])])
        next_state = np.concatenate([next_state, np.array([self.is_punished])])
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

# def color_map(channels: int) -> dict:
#     '''
#     Generates a color map for the food truck environment.

#     Parameters:
#         channels: the number of appearance channels in the environment

#     Return:
#         A dict of object-color mappings
#     '''
#     if channels > 5:
#         colors = {
#             # 'EmptyObject': [0 for _ in range(channels)],
#             'EmptyObject': [1. if x == 0 else 0 for x in range(channels)],
#             'Wall': [1. if x == 1 else 0 for x in range(channels)],
#             'Gem': None,
#             'Coin': None
#             # 'Agent': [255 if x == 0 else 0 for x in range(channels)],
#             # 'Wall': [255 if x == 1 else 0 for x in range(channels)],
#             # 'Gem': [255 if x == 2 else 0 for x in range(channels)],
#             # 'Food': [255 if x == 3 else 0 for x in range(channels)],
#             # 'Coin': [255 if x == 4 else 0 for x in range(channels)],
#             # 'Evil_coin': [255 if x == 4 else 0 for x in range(channels)],
#             # 'Bone': [255 if x == 5 else 0 for x in range(channels)]
#         }
#     else:
#         colors = {
#             'EmptyObject': [255., 255., 255.],
#             'Agent': [0., 0., 255.],
#             'Wall': [153.0, 51.0, 102.0],
#             'Gem': [0., 255., 0.],
#             'Coin': [255., 255., 0.],
#             'Evil_Coin': [255., 255., 0.],
#             'Food': [255., 0., 0.],
#             'Bone': [0., 0., 0.]
#         }
#     return colors



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