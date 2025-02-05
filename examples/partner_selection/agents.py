from examples.partner_selection.entities import Wall, EmptyObject
# from examples.trucks.utils import color_map

from ast import literal_eval as make_tuple
from typing import Optional
import torch
import numpy as np
import random
from scipy.special import softmax

from agentarium.visual_field import visual_field
from agentarium.primitives import GridworldEnv

class Agent:
    def __init__(self, model, cfg, ixs):
        self.kind = "Agent"
        self.type = 'agent'
        self.cfg = cfg    
        # print(cfg.agent.agent.appearance)
        self.appearance = None
        # if isinstance(cfg.agent.agent.appearance, str):
        #     self.appearance = make_tuple(cfg.agent.agent.appearance) 
        # else:
        #     self.appearance = cfg.agent.agent.appearance[ixs] # agents are blue # cfg.agent.agent.appearance[ixs]?        
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
        self.to_be_punished = {'Gem':0, 'Bone':0, 'Coin':0}
        self.preferences = None
        self.variability = None
        self.social_task_memory = {'gt':[], 'pred':[]}

        # training-related features
        self.task_model = None  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.partner_choice_model = model
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

    def current_state(self, env: GridworldEnv) -> np.ndarray:
        state = self.pov(env)
        # state = np.concatenate([state, np.array([self.is_punished])])
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
                 ) -> tuple:
        
        '''
        Takes an action and returns a new location
        '''
        if action == 0: # UP
            self.sprite = f'{self.cfg.root}/examples/partner_selection/assets/hero-back.png'
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == 1: # DOWN
            self.sprite = f'{self.cfg.root}/examples/partner_selection/assets/hero.png'
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == 2: # LEFT
            self.sprite = f'{self.cfg.root}/examples/partner_selection/assets/hero-left.png'
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == 3: # RIGHT
            self.sprite = f'{self.cfg.root}/examples/partner_selection/assets/hero-right.png'
            new_location = (self.location[0], self.location[1] + 1, self.location[2])
        if action == 4: # zap
            new_location = (self.location[0], self.location[1], self.location[2])
        return new_location
    

    
    def simple_transition(
            self,
            partner,
            env
            ) -> tuple:
        '''
        Changes the world based on the action taken.
        '''

        # Get current state
        state = self.pov(env)
        if self.model_type != 'PPO':
            model_input = torch.from_numpy(self.current_state(env=env)).view(1, -1)
        else:
            model_input = torch.from_numpy(state)
            
        reward = 0

        # Take action based on current state
        if self.model_type == 'PPO':
            action, action_prob = self.task_model.take_action(model_input)
        else:
            action = self.task_model.take_action(model_input)
            action_prob = None 

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
        if self.model_type == 'PPO':
            return state, action, reward, next_state, False, action_prob
        else:
            return state, action, reward, next_state, False
        
    
    def frozen_network_foraging(
                self,
                partner,
                env,
                max_turns):
            '''
            Run the agent and the partner in an enviornment for testing. 
            '''
            env.reset()

            env.cache['harm'] = [0 for _ in range(len(env.agents))]

            done = 0 
            turn = 0
            losses = 0
            game_points = [0 for _ in range(len(env.agents))]
            self_points = 0
            self_id = self.appearance

            while not done:

                turn = turn + 1

                entities = env.get_entities_for_transition()
                # Entity transition
                for entity in entities:
                    entity.transition(env)

                # Agent transition
                for agent in [self, partner]:

                    (state,
                    action,
                    reward,
                    next_state,
                    done_
                    ) = agent.simple_transition(env)

                    if turn >= max_turns or done_:
                        done = 1

                    exp = (1, (state, action, reward, next_state, done))
                    # agent.episode_memory.append(exp)

                    game_points[agent.ixs] += reward
                    if agent.appearance == self_id:
                        self_points += reward

            return self_points 
    
    
    def interaction_task(self,
                         partner,
                         cfg,
                         env
                         ):
        '''
        Conduct the interaction task with the selected partner.
        '''
        reward = 0 
        for _ in range(cfg.interaction_task.n_trials):
            if cfg.interaction_task.mode == 'prediction':
                prediction = self.task_model(torch.from_numpy(partner.appearance).float())
                partner_choice_in_task = np.argmax(partner.preferences)
                # print(prediction, partner_choice_in_task)
                if int(torch.max(prediction, dim=0)[1]) == int(partner_choice_in_task):
                    reward += 1
                self.social_task_memory['pred'].append(prediction)
                self.social_task_memory['gt'].append(torch.tensor(partner_choice_in_task))
            elif cfg.interaction_task.mode == 'grid':
                # agents = [self, partner]
                # entities = create_entities(cfg)
                # env = partner_selection(cfg, agents, entities)
                reward += self.frozen_network_foraging(partner, env, cfg.task_max_turns)
        return reward 
    
    
    def transition(self,
                   env,
                   partner_choices,
                   is_focal,
                   cfg, 
                   mode='prediction') -> tuple:
        '''
        Changes the world based on the action taken.
        '''

        # Get current state
        state = env.state(self)
        # model_input = torch.from_numpy(self.current_state(env=env)).view(1, -1)
        if self.model_type == 'PPO':
            model_input = torch.from_numpy(state)

        reward = 0

        # Take action based on current state
        if self.model_type == 'PPO':
            # action, action_prob = self.model.take_action(model_input, whether_to_predict=False, steps=2)
            action, action_prob = self.partner_choice_model.take_action(model_input)
        else:
            action = self.partner_choice_model.take_action(model_input)
            action_prob = None

        # Attempt the transition 
        selected_partner = partner_choices[action]
        selected_partner_ixs = selected_partner.ixs

        # # Get the interaction rewards
        # if is_focal:
        #     reward += self.interaction_task(selected_partner, mode, 5, env)

        if self.model_type == 'PPO':
            return state, action, selected_partner, False, action_prob, selected_partner_ixs
        else:
            return state, action, selected_partner, False, selected_partner_ixs
        

        
    def reset(self, env: GridworldEnv) -> None:
        self.init_replay(env)
        self.encounters = {
            'Gem': 0,
            'Coin': 0,
            'Food': 0,
            'Bone': 0,
            'Wall': 0
        }

    def update_preference(self):
        randomness = (random.random() - 0.5) * self.variability
        self.preferences = np.array(self.preferences) - np.array([randomness, -1 * randomness])
        self.preferences = np.clip(self.preferences, 0., 1.)


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
            # 'Agent': [255 if x == 0 else 0 for x in range(channels)],
            'Wall': [255. if x == 1 else 0 for x in range(channels)],
            'Gem': [255. if x == 2 else 0 for x in range(channels)],
            'Food': [255. if x == 4 else 0 for x in range(channels)],
            'Coin': [255. if x == 3 else 0 for x in range(channels)],
            'Bone': [255. if x == 5 else 0 for x in range(channels)]
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