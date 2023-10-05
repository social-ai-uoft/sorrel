'''
pov
transition
have transition keep track of number of gems collected
'''

from examples.RPG.entities import Wall, EmptyObject

from ast import literal_eval as make_tuple
from collections import deque
import torch

# TODO: 
# make sure dead agent should change model to None
# probably need to add die into particular circumstances in transition
# touch base with Eric about adding in new pov code

class Agent:
    def __init__(self, model, cfg):
        self.type = "Agent"
        self.cfg = cfg      
        self.appearance = make_tuple(cfg.agent.agent.appearance)  # agents are blue
        self.tile_size = make_tuple(cfg.agent.agent.tile_size)
        self.sprite = 'examples/RPG/assets/hero.png'
        self.passable = 0  # whether the object blocks movement
        self.value = 0  # agents have no value
        self.health = cfg.agent.agent.health  # for the agents, this is how hungry they are
        self.location = None
        self.action_type = "neural_network"
        
        # training-related features
        self.model = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.episode_memory = deque([], maxlen=100)  # we should read in these maxlens
        self.num_memories = cfg.agent.agent.num_memories
        self.init_rnn_state = None
        self.visual_depth = cfg.agent.agent.visual_depth 
        self.pov_size = cfg.agent.agent.pov_size
    
    def init_replay(self): 
        """
        Fills in blank images for the LSTM before game play.
        Impicitly defines the number of sequences that the LSTM will be trained on.
        """
        image = torch.zeros(1, self.num_memories, self.visual_depth, self.pov_size, self.pov_size).float()
        priority = torch.tensor(0.1)
        blank = torch.tensor(0.0)
        exp = (priority, (image, blank, blank, image, blank))
        self.episode_memory.append(exp)
    
    def reset(self):
        '''
        resets agent init_rnn_state, replay, and reward
        '''
        self.init_rnn_state = None
        self.init_replay()
        self.reward = 0

    def movement(self, action):
        """
        Takes an action and returns a new location
        """
        if action == 0:
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == 1:
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == 2:
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == 3:
            new_location = (self.location[0], self.location[1] + 1, self.location[2])
        return new_location
    
    def pov(self, env):
        from utils import make_pov_image
        # pdb.set_trace()

        previous_state = self.episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

        state_now = torch.tensor([])
 
        # img = create_pov_image(env, 0, self)
        # transform = T.Compose([T.PILToTensor()])
        # input = transform(img).unsqueeze(0).permute(0, 3, 1, 2).float()

        img = make_pov_image(env, self)
        input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        state_now = torch.cat((state_now, input.unsqueeze(0)), dim=2)

        # hack: add empty inventory var
        # inventory_var = torch.tensor([])
        # tmp = (current_state[:, -1, -1, :, :] * 0) + 0
        # inventory_var = torch.cat((inventory_var, tmp), dim=0)
        # inventory_var = inventory_var.unsqueeze(0).unsqueeze(0)
        # state_now = torch.cat((state_now, inventory_var), dim=2)

        current_state[:, -1, :, :, :] = state_now

        return current_state
    
    def transition(self, env):
        """
        Changes the world based on the action taken
        """
        reward = 0
        state = self.pov(env)
        action, self.init_rnn_state = self.model.take_action(state, self.init_rnn_state)

        attempted_location = self.movement(action)
        target_object = env.observe(attempted_location)
        env.move(self, attempted_location)
        reward = target_object.value

        next_state = self.pov(env)
        self.reward += reward

        return state, action, reward, next_state

    # def transition(self, env, models, action, location):
    #     """
    #     Changes the world based on the action taken
    #     """
    #     done = 0
    #     reward = 0
    #     attempted_location = self.movement(action)
    #     holdObject = env.world[location]

    #     if env.world[attempted_location].passable == 1:
    #         env.world[location] = EmptyObject()
    #         reward = env.world[attempted_location].value
    #         env.world[attempted_location] = holdObject
    #         new_loc = attempted_location

    #     else:
    #         if isinstance(
    #             env.world[attempted_location], Wall
    #         ):  # Replacing comparison with string 'kind'
    #             reward = -0.1

    #     next_state = env.pov(new_loc)
    #     self.reward += reward

    #     return env.world, reward, next_state, done, new_loc