from examples.RPG.entities import Wall, EmptyObject, Collectable

from ast import literal_eval as make_tuple
from collections import deque
import torch

# TODO: 
# make sure dead agent should change model to None
# probably need to add die into particular circumstances in transition
# touch base with Eric about adding in new pov code

class Agent:
    def __init__(self, model, cfg):
        # basic features shared with other entities
        self.type = "agent"
                
        self.appearance = make_tuple(cfg.agent.agent.appearance)  # agents are blue
        self.tile_size = make_tuple(cfg.agent.agent.tile_size)
        self.sprite = 'examples/RPG/assets/hero.png'
        
        self.passable = 0  # whether the object blocks movement
        self.value = 0  # agents have no value
        self.health = cfg.agent.agent.health  # for the agents, this is how hungry they are
        self.death = False
        self.location = None
        self.trainable = 1  # whether there is a network to be optimized
        self.has_transitions = True
        self.action_type = "neural_network"
        
        # training-related features
        self.model = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.episode_memory = deque([], maxlen=100)  # we should read in these maxlens
        self.num_memories = cfg.agent.agent.num_memories
        self.init_rnn_state = None
        self.visual_depth = cfg.agent.agent.visual_depth  # agents can see three radius around them
        self.pov_size = cfg.agent.agent.pov_size

    def init_replay(self): 
        """
        Fills in blank images for the LSTM before game play.
        Impicitly defines the number of sequences that the LSTM will be trained on.
        """
        image = torch.zeros(1, self.num_memories, self.visual_depth, self.tile_size[0] * (2 * self.visual_depth + 1), self.tile_size[1] * (2 * self.visual_depth + 1)).float()
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

    def die(
        self, models, world, attempted_locaton_1, attempted_locaton_2, extra_reward=True
    ):
        """
        Replaces the last memory with a memory that has a reward of -25 and the image of its
        death. This is to encourage the agent to not die.
        TODO: this is failing at the moment. Need to fix.
        """
        lastexp = world[attempted_locaton_1, attempted_locaton_2, 0].episode_memory[-1]
        world[attempted_locaton_1, attempted_locaton_2, 0].episode_memory[-1] = (
            lastexp[0],
            lastexp[1],
            -25,
            lastexp[3],
            1,
        )

        # TODO: Below is very clunky and a more principles solution needs to be found

        models[
            world[attempted_locaton_1, attempted_locaton_2, 0].policy
        ].transfer_memories(
            world, (attempted_locaton_1, attempted_locaton_2, 0), extra_reward=True
        )

        # this can only be used it seems if all agents have a different id
        self.death = True  # label the agents death
        self.reward = 0
        self.appearance = [130.0, 130.0, 130.0]  # dead agents are grey
        self.model = None
        self.trainable = 0  # whether there is a network to be optimized
        self.action_type = "static"
        self.visual_depth = 4
        self.episode_memory = deque([], maxlen=5)
        self.has_transitions = False

    def pov(self, env):
        """
        env: instance of environment that agent perceives
        TODO: refactor all the code so that this is here
        """
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
        # pdb.set_trace()

        # hack: add empty inventory var
        inventory_var = torch.tensor([])
        tmp = (current_state[:, -1, -1, :, :] * 0) + 0
        inventory_var = torch.cat((inventory_var, tmp), dim=0)
        inventory_var = inventory_var.unsqueeze(0).unsqueeze(0)
        state_now = torch.cat((state_now, inventory_var), dim=2)

        # if len(inventory) > 0:
        #     """
        #     Loops through each additional piece of information and places into one layer
        #     """
        #     inventory_var = torch.tensor([])
        #     for item in range(len(inventory)):
        #         tmp = (current_state[:, -1, -1, :, :] * 0) + inventory[item]
        #         inventory_var = torch.cat((inventory_var, tmp), dim=0)
        #     inventory_var = inventory_var.unsqueeze(0).unsqueeze(0)
        #     state_now = torch.cat((state_now, inventory_var), dim=2)

        current_state[:, -1, :, :, :] = state_now
        # current_state = state_now

        return current_state

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

    def transition(self, env):
        """
        Changes the world based on the action taken
        """
        reward = 0
        state = self.pov(env)
        action, self.init_rnn_state = self.model.take_action(state, self.init_rnn_state)

        attempted_location = self.movement(action)
        target_object = env.observe(attempted_location)

        # if type entity, collect value

        # if isinstance(target_object, Collectable):
        #     reward = target_object.value
        # # ____ should be covered by move____    
        # if target_object.passable == 1:
        #     reward = target_object.value

        # elif isinstance(target_object, Wall):  # Replacing comparison with string 'kind'
        #         reward = -0.1

        # movement logic - will move unless can't,
        # gathers reward/penalty of target object
        env.move(self, attempted_location)
        reward = target_object.value

        next_state = self.pov(env)
        self.reward += reward

        return state, action, reward, next_state
        