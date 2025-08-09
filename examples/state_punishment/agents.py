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
        self.type = 'agent'
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
        self.to_be_punished = {'Gem':0, 'Bone':0, 'Coin':0}
        self.transgression_record = {resource: [] for resource in cfg.state_sys.resources}
        self.punishment_record = {resource: [] for resource in cfg.state_sys.resources}

        # training-related features
        self.model = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        # self.episode_memory = Memory(cfg.agent.agent.memory_size)
        self.num_frames = cfg.agent.agent.num_memories
        self.init_rnn_state = None
        self.encounters = {
            entity_name: 0 for entity_name in cfg.env.entity_names
        }


    def init_replay(self, env: GridworldEnv, state_mode: str) -> None:
        """Fill in blank images for the LSTM."""
        if state_mode == 'simple':
            state = np.zeros_like(np.concatenate([self.pov(env), np.zeros(self.extra_percept_size)]))
        else:
            grid_state = self.generate_composite_state([env for _ in range(6)])
            state = np.zeros_like(np.concatenate([grid_state, np.zeros(self.extra_percept_size)]))
        action = 0  # Action outside the action space
        reward = 0.0
        done = 0.0
        for _ in range(self.num_frames-1):
            self.model.memory.add(state, action, reward, done)
    

    def add_memory(self, state: np.ndarray, action: int, reward: float, done: bool) -> None:
        """Add an experience to the memory."""
        self.model.memory.add(state, action, reward, float(done))
    

    def add_final_memory(self, state: np.ndarray) -> None:
        self.model.memory.add(state, 0, 0.0, float(True))


    def current_state(self, state_sys, env: GridworldEnv, envs, state_mode='simple') -> np.ndarray:
        if state_mode:
            state = self.generate_composite_state(envs)
        else:
            state = self.pov(env)
        to_be_punished = sum(self.to_be_punished.values())
        state = np.concatenate([state, np.array([state_sys.prob*255])])
        state = np.concatenate([state, np.array([to_be_punished*255])])
        if self.num_frames > 1:
            prev_states = self.model.memory.current_state(stacked_frames=self.num_frames-1)
            current_state = np.vstack((prev_states, state))
        else:
            current_state = state
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
                 state_sys,
                 mode='simple'
                 ) -> tuple:
        
        '''
        Takes an action and returns a new location
        '''
        # if allow only one vote
        vote_is_valid = False
        if self.ixs == 0:
            vote_is_valid = True

        if mode == 'simple':
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
                if vote_is_valid:
                    state_sys.prob += state_sys.change_per_vote
                    state_sys.prob = np.clip(state_sys.prob, 0, 1)
                    state_sys.level += 1
                    state_sys.level = int(np.clip(state_sys.level, 0, state_sys.max_level))
            if action == 5: # vote against state punishment
                self.sprite = self.sprite
                new_location = self.location
                if vote_is_valid:
                    state_sys.prob -= state_sys.change_per_vote
                    state_sys.prob = np.clip(state_sys.prob, 0, 1)
                    state_sys.level -= 1
                    state_sys.level = int(np.clip(state_sys.level, 0, state_sys.max_level))

        elif mode == 'composite':
            if action%4 == 0: # UP
                self.sprite = f'{self.cfg.root}/examples/state_punishment/assets/hero-back.png'
                new_location = (self.location[0] - 1, self.location[1], self.location[2])
            if action%4 == 1: # DOWN
                self.sprite = f'{self.cfg.root}/examples/state_punishment/assets/hero.png'
                new_location = (self.location[0] + 1, self.location[1], self.location[2])
            if action%4 == 2: # LEFT
                self.sprite = f'{self.cfg.root}/examples/state_punishment/assets/hero-left.png'
                new_location = (self.location[0], self.location[1] - 1, self.location[2])
            if action%4 == 3: # RIGHT
                self.sprite = f'{self.cfg.root}/examples/state_punishment/assets/hero-right.png'
                new_location = (self.location[0], self.location[1] + 1, self.location[2])
            
            if action//4 == 0:
                state_sys.prob += state_sys.change_per_vote
                state_sys.prob = np.clip(state_sys.prob, 0, 1)
            elif action//4 == 1:
                state_sys.prob -= state_sys.change_per_vote
                state_sys.prob = np.clip(state_sys.prob, 0, 1)

        return new_location

    
    def transition(self,
                   env,
                   state_sys,
                   mode, 
                   action_mode='simple',
                   state_is_composite=False,
                   envs=None,
                   is_eval=False) -> tuple:
        '''
        Changes the world based on the action taken.

        when transgression is not apparent. 

        currently, all transgression records will be equally punished.
        '''

        # Get current state
        # print(state_is_composite)
        if state_is_composite:
            state = self.generate_composite_state(envs)
        else:
            state = self.pov(env)

        to_be_punished = sum(self.to_be_punished.values())
       
        state = np.concatenate([state, np.array([state_sys.prob*255])])
        state = np.concatenate([state, np.array([to_be_punished*255])])
        model_input = torch.from_numpy(self.current_state(
            state_sys=state_sys, 
            env=env, 
            envs=envs, 
            state_mode=state_is_composite
            )).view(1, -1)

        reward = 0

        # Take action based on current state
        if is_eval:
            action, action_values = self.model.take_action(model_input, eval=True)
        else:
            action = self.model.take_action(model_input)
        if self.cfg.random:
            action = random.randint(0,self.cfg.model.iqn.parameters.action_size-1)
        # Attempt the transition 
        attempted_location = self.movement(action, state_sys, action_mode)
        target_object = env.observe(attempted_location)
        env.move(self, attempted_location)

        # Get the interaction reward
        reward += target_object.value
        # if to_be_punished >= 1.: ##TODO: change to to_be_punished 
        #     self.punishment_record.append(1)
        #     # reward -= state_sys.magnitude * to_be_punished
        #     # TODO: if self.to_be_punished: 
        # else:
        #     self.punishment_record.append(0)
        
        if mode == 'certain':
            self.to_be_punished = {'Gem':0, 'Bone':0, 'Coin':0}

        # Get the delayed reward
        # if env.cache['harm'][self.ixs] < 0:
        #     print(self.ixs)
        #     ll
        reward += env.cache['harm'][self.ixs]
        env.cache['harm'][self.ixs] = 0

        # If the agent performs a transgression
        if str(target_object) in state_sys.potential_taboo:

            if str(target_object) in state_sys.taboo:
                self.transgression_record[str(target_object)].append(1)
            # else:
                # self.transgression_record.append(0)

            if mode == 'certain':
                # prob = state_sys.punishment_schedule_func(str(target_object))
                #TODO: merge the conditions/simplify
                if state_sys.resource_punishment_is_ambiguous:
                    punishment_prob = state_sys.punishment_schedule_func(str(target_object))
                    val_of_state_punishment = state_sys.magnitude * (random.random() < punishment_prob)
                    reward -= val_of_state_punishment
                else:
                    if state_sys.only_punish_taboo:
                        if str(target_object) in state_sys.taboo:
                            punishment_prob = state_sys.punishment_schedule_func(str(target_object))
                            val_of_state_punishment = state_sys.magnitude * (random.random() < punishment_prob)
                            reward -= val_of_state_punishment
                        else:
                            val_of_state_punishment = 0
                    # use a fixed list of probabilities to determine punishment
                    else:
                        val_of_state_punishment = state_sys.magnitude * (random.random() 
                                                        < state_sys.prob_list[str(target_object)]*state_sys.prob) # instant punishment
                        reward -= val_of_state_punishment
                # record being punished or not
                self.punishment_record[str(target_object)].append(1*(val_of_state_punishment > 0))

            cache_harm = [env.cache['harm'][k] - target_object.social_harm 
                                        if k != self.ixs else env.cache['harm'][k]
                                        for k in range(len(env.cache['harm']))
                                        ]
            
            # assign harm value to each individual world
            if state_is_composite:
                for env_ixs, env_ in enumerate(envs):
                    env_.cache['harm'][env_ixs] = cache_harm[env_ixs]
            else:
                env.cache['harm'] = cache_harm
            

        # Add to the encounter record
        if str(target_object) in self.encounters.keys():
            self.encounters[str(target_object)] += 1 

        # Get the next state   
        if state_is_composite:
            next_state = self.generate_composite_state(envs)
        else:
            next_state = self.pov(env)
        next_state = np.concatenate([next_state, np.array([state_sys.prob*255])])
        next_state = np.concatenate([next_state, np.array([to_be_punished*255])])  ## TODO: how to edit the to_be_punished state within agents
        
        # reset to_be_punished 
        if mode == 'ambiguous':
            self.to_be_punished = {'gem':0, 'bone':0, 'coin':0}
        if not is_eval:
            return state, action, reward, next_state, False
        else:
            return state, action, reward, next_state, False, action_values


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

    # def transition(self,
    #                env,
    #                state_sys) -> tuple:
    #     '''
    #     Changes the world based on the action taken.

    #     when transgression is apparent. 
    #     '''

    #     # Get current state
    #     state = self.pov(env)
    #     state = np.concatenate([state, np.array([state_sys.prob])])
    #     state = np.concatenate([state, np.array([self.is_punished])])
    #     model_input = torch.from_numpy(self.current_state(state_sys=state_sys, env=env)).view(1, -1)

        
    #     # print(model_input.size())
    #     # ll
    #     # state_punishment_prob_tensor = torch.full((state.shape()[1], state.shape()[2]), state_sys.prob).view(1, -1)
    #     # model_input = torch.concat([model_input, state_punishment_prob_tensor])
    #     reward = 0

    #     # Take action based on current state
    #     action = self.model.take_action(model_input)

    #     # Attempt the transition 
    #     attempted_location = self.movement(action, state_sys)
    #     target_object = env.observe(attempted_location)
    #     env.move(self, attempted_location)

    #     # Get the interaction reward
    #     reward += target_object.value
    #     if self.is_punished == 1.: ##TODO: change to to_be_punished 
    #         reward -= state_sys.magnitude 
    #         # TODO: if self.to_be_punished: 
        
    #     self.is_punished = 0. 

    #     # Get the delayed reward
    #     reward += env.cache['harm'][self.ixs]
    #     env.cache['harm'][self.ixs] = 0 
    #     # If the agent performs a transgression
    #     if str(target_object) in state_sys.taboo:
    #         # reward -= state_sys.magnitude * (random.random() < state_sys.prob)
    #         if random.random() < state_sys.prob_list[str(target_object)]:
    #             self.is_punished = 1.
    #         env.cache['harm'] = [env.cache['harm'][k] - target_object.social_harm 
    #                                     if k != self.ixs else env.cache['harm'][k]
    #                                     for k in range(len(env.cache['harm']))
    #                                     ]
            

    #     # Add to the encounter record
    #     if str(target_object) in self.encounters.keys():
    #         self.encounters[str(target_object)] += 1 

    #     # Get the next state   
    #     next_state = self.pov(env)
    #     next_state = np.concatenate([next_state, np.array([state_sys.prob])])
    #     next_state = np.concatenate([next_state, np.array([self.is_punished])])  ## TODO: how to edit the to_be_punished state within agents
        
    #     # reset to_be_punished 
    #     # self.to_be_punished = {'gem':0, 'bone':0, 'coin':0}

    #     return state, action, reward, next_state, False


    def reset(self, env: GridworldEnv, state_mode='simple') -> None:
        self.init_replay(env, state_mode)
        self.encounters = {
            entity_name: 0 for entity_name in self.cfg.env.entity_names
        }
    
    def reset_record(self) -> None:
        self.transgression_record = {resource: [] for resource in self.cfg.state_sys.resources}
        self.punishment_record = {resource: [] for resource in self.cfg.state_sys.resources}

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
            'Bone': [255 if x == 5 else 0 for x in range(channels)],
            'A': [255 if x == 6 else 0 for x in range(channels)],
            'B': [255 if x == 7 else 0 for x in range(channels)],
            'C': [255 if x == 8 else 0 for x in range(channels)],
            'D': [255 if x == 9 else 0 for x in range(channels)],
            'E': [255 if x == 10 else 0 for x in range(channels)],
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