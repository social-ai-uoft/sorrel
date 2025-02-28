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
        self.base_preferences = None
        self.variability = None
        self.base_variability = None
        self.social_task_memory = {'gt':[], 'pred':[]}
        self.delay_reward = 0
        self.trainable = cfg.agent.agent.trainable
        self.frozen = cfg.agent.agent.frozen
        self.action_size = cfg.agent.agent.action_size

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
            try:
                self.model.memory.add(state, action, reward, done)
            except:
                self.partner_choice_model.memory.add(state, action, reward, float(done))
     
    
    def add_memory(self, state: np.ndarray, action: int, reward: float, done: bool) -> None:
        """Add an experience to the memory."""
        try:
            self.model.memory.add(state, action, reward, float(done))
        except:
            self.partner_choice_model.memory.add(state, action, reward, float(done))
    
    def add_final_memory(self, state: np.ndarray) -> None:
        try:
            self.model.memory.add(state, 0, 0.0, float(True))
        except:
            self.partner_choice_model.memory.add(state, 0, 0.0, float(True))

    def current_state(self, env: GridworldEnv, partner_selection=True) -> np.ndarray:
        if partner_selection:
            state = env.state(self)
        else:
            state = self.pov(env)
        # state = np.concatenate([state, np.array([self.is_punished])])
        try:
            prev_states = self.model.memory.current_state(stacked_frames=self.num_frames-1)
        except:
            prev_states = self.partner_choice_model.memory.current_state(stacked_frames=self.num_frames-1)
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
            model_input = torch.from_numpy(self.current_state(env=env, partner_selection=False)).view(1, -1)
        else:
            model_input = torch.from_numpy(state).double()
            
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
                # partner_choice_in_task = random.choices([v for v in range(len(partner.preferences))], 
                #                                         partner.preferences, k=1)[0]
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
        partner.delay_reward = reward
        # print(partner.ixs, partner.delay_reward)
        return reward 
    
    def SB_task(
            self, 
            action, 
            partner,
            cfg, 
            env
            ):
        '''
        Conduct the Stravinsky and Bach task with the selected partner.
        '''
        # generate partner choice
        if cfg.partner_free_choice:
            state = env.state(self, cfg)
            if self.model_type != 'PPO':
                model_input = torch.from_numpy(
                    self.current_state(env=env, partner_selection=True)
                    ).view(1, -1)
            if self.model_type == 'PPO':
                model_input = torch.from_numpy(state).double()
            partner_action, partner_action_prob = self.partner_choice_model.take_action(model_input)
            parnter_learning_tuples = {'state': state, 
                                       'action': partner_action, 
                                       'action_prob': partner_action_prob}
        elif cfg.partner_free_choice_beforehand:
            partner_action = partner.cached_action
            parnter_learning_tuples = None
        else:
            partner_action = random.choices([0, 1], partner.preferences, k=1)[0]
            parnter_learning_tuples = None

        # determine the outcomes 
        if action <= 3:
            self_SB_choice = action % 2
            # print('pair', self_SB_choice, partner_action)
            if self_SB_choice == partner_action:
                if cfg.study == 1.5:
                    r = self.preferences[self_SB_choice] > 0.1
                elif cfg.study == 1:
                    r = 1 * (self.preferences[self_SB_choice] > 0.)
                reward = r
                partner_reward = r
            else:
                reward = 0
                partner_reward = 0
        else: 
            reward = 0
            partner_reward = 0 
        # reward = self.preferences[self_SB_choice] > 0
        # print(self.ixs, reward, self_SB_choice, self.preferences)
        # partner_reward = -1
        return reward, partner_reward, \
            self.preferences[self_SB_choice] > 0, \
                self_SB_choice == partner_action, \
                    parnter_learning_tuples

    def selection_task_action(self, action, is_focal):
        """
        Define the action dynamics of the partner selection model
        """
        if not self.frozen:
            if action == 2:
                self.variability -= self.base_variability*0.2 # 0.02
                self.variability = min(max(0., self.variability), 1.)
            elif action == 3:
                self.variability += self.base_variability*0.2
                self.variability = min(max(0., self.variability), 1.)
    
    def change_preferences(self, action):
        """
        Change the prefernces values based on the actions taken
        """
        v = 0.02
        if hasattr(self, 'role'):
            if self.role == 'partner':
                assert action in [0, 1, 2], ValueError('Action not in action space')
                if not self.frozen:
                    if action == 0:
                        self.preferences[0] -= v
                        self.preferences[1] += v
                    elif action == 1:
                        self.preferences[0] += v
                        self.preferences[1] -= v
                    for i in range(len(self.preferences)):
                        self.preferences[i] = min(max(0., self.preferences[i]), 1.)
              
        else:
            assert action in range(6), ValueError("Action not in action space")
            if not self.frozen:
                if action == 4:
                        self.preferences[0] -= v
                        self.preferences[1] += v
                elif action == 5:
                    self.preferences[0] += v
                    self.preferences[1] -= v
                for i in range(len(self.preferences)):
                    self.preferences[i] = min(max(0., self.preferences[i]), 1.)
            
            
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
        state = env.state(self, cfg)
        if self.model_type != 'PPO':
            model_input = torch.from_numpy(self.current_state(env=env, partner_selection=True)).view(1, -1)
        if self.model_type == 'PPO':
            model_input = torch.from_numpy(state).double()

        reward = 0

        # Take action based on current state
        if self.model_type == 'PPO':
            if hasattr(self, 'lstm'):
                if self.lstm:
                    self.hidden_in = self.hidden_out
                    # action, action_prob = self.model.take_action(model_input, whether_to_predict=False, steps=2)
                    action, action_prob, hidden = self.partner_choice_model.take_action(model_input, self.hidden_in)
                    self.hidden_out = hidden
            else:
                
                action, action_prob = self.partner_choice_model.take_action(model_input)
                # set the strategy
                if cfg.hardcoded:
                    if self.ixs != 0:
                        action = 2

        else:
            if hasattr(self, 'lstm'):
                if self.lstm:
                    self.hidden_in = self.hidden_out
                    action, hidden = self.partner_choice_model.take_action(model_input, hidden=self.hidden_in)
                    self.hidden_out = hidden
            else:
                action = self.partner_choice_model.take_action(model_input)
            action_prob = None

        # set random actions
        if cfg.study == 3:
            if cfg.random_selection:
                if action in [0,1]:
                    action = random.randint(0,1)
                    
        elif cfg.study == 2:
            if self.ixs == 0 and cfg.random_selection:
                action = random.randint(0,1)

        elif cfg.study == 1.5:
            if cfg.random_selection:
                if action <= 3:
                    pref_act = action % 2
                    action = 2*random.randint(0,1) + pref_act
        
        elif cfg.study == 1:
            if self.ixs == 0 and cfg.random_selection:
                if action <= 3:
                    pref_act = action % 2
                    action = 2*random.randint(0,1) + pref_act

        # execute the selection model action
        if cfg.experiment.is_SB_task:
            self.change_preferences(action)
        else:
            self.selection_task_action(action, is_focal)

        # Select the partner
        if cfg.experiment.is_SB_task:
            if action <= 3: 
                selected_partner = partner_choices[int(action//2)]
                selected_partner_ixs = selected_partner.ixs
            else:
                selected_partner = None
                selected_partner_ixs = None
        else:
            if int(action) <= 1:
                selected_partner = partner_choices[action]
                selected_partner_ixs = selected_partner.ixs
            else:
                selected_partner = None
                selected_partner_ixs = None

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

    def update_preference(self, mode='categorical'):
        if mode == 'additive':
            randomness = (random.random() - 0.5) * self.variability * 10
            self.preferences = np.array(self.base_preferences) - np.array([randomness, -1 * randomness])
            self.preferences = np.clip(self.preferences, 0., 1.)
        elif mode == 'categorical':
            random_num = random.random()
            num_categories = len(self.preferences)
            if self.variability > random_num:
                probs = np.zeros(num_categories)
                probs[np.random.randint(num_categories)] = 1.
                self.preferences = probs


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