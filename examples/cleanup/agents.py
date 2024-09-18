import torch
from typing import Optional, Sequence

import numpy as np

from examples.trucks.agents import Memory
from gem.models.ann import ANN
from gem.primitives import Object, GridworldEnv, Location, Vector
from gem.utils import visual_field, visual_field_multilayer
from gem.models.grid_cells import positional_embedding

# ------------------- #
# region: Agent class #
# ------------------- #

class Agent(Object):
    """Cleanup agent."""

    def __init__(self, cfg, appearance, model):

        super().__init__(appearance)

        self.cfg = cfg
        self.vision = cfg.agent.agent.vision
        self.direction = 2  # 90 degree rotation: default at 180 degrees (facing down)
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.has_transitions = True
        self.sprite = f'{self.cfg.root}/examples/cleanup/assets/hero.png'

        # training-related features
        self.action_type = "neural_network"
        self.model = model
        self.episode_memory = Memory(cfg.agent.agent.memory_size)
        self.num_memories = cfg.agent.agent.memory_size
        self.init_rnn_state = None

        # logging features
        self.outcome_record = {"harvest": 0, "zap": 0, "get_zapped": 0, "clean": 0}

    def sprite_loc(self) -> None:
        """Determine the agent's sprite based on the location."""
        sprite_directions = [
            f'{self.cfg.root}/examples/cleanup/assets/hero-back.png', # up
            f'{self.cfg.root}/examples/cleanup/assets/hero-right.png', # right
            f'{self.cfg.root}/examples/cleanup/assets/hero.png', # down
            f'{self.cfg.root}/examples/cleanup/assets/hero-left.png' # left
        ]
        self.sprite = sprite_directions[self.direction]

    def init_replay(self) -> None:
        """Fill in blank images for the LSTM."""

        priority = torch.tensor(0.1)
        num_frames = self.model.num_frames
        if self.cfg.env.full_mdp:
            state = torch.zeros(1, num_frames, *self.model.state_size).float()
        else:
            # Number of one-hot code channels
            C = len(self.appearance)
            H = W = self.vision * 2 + 1
            state = torch.zeros(1, num_frames, C, H, W).float()

        action = torch.tensor(7.0)  # Action outside the action space
        reward = torch.tensor(0.0)
        done = torch.tensor(0.0)
        exp = (priority, (state, action, reward, state, done))
        self.episode_memory.append(exp)

    def act(self, action: int) -> tuple[int, ...]:
        """Act on the environment.
        
        Params:
            action: (int) An integer indicating the action to take.
            
        Return:
            tuple[int, ...] A location tuple indicating the updated
            location of the agent.
        """

        # Default location
        next_location = self.location

        if action == 0:  # NOOP
            pass

        if action == 1:  # FORWARD
            forward_vector = Vector(1, 0, direction=self.direction)
            cur_location = Location(*self.location)
            next_location = (cur_location + forward_vector).to_tuple()

        if action == 2:  # BACK
            backward_vector = Vector(-1, 0, direction=self.direction)
            cur_location = Location(*self.location)
            next_location = (cur_location + backward_vector).to_tuple()

        if action == 3:  # TURN CLOCKWISE
            # Add 90 degrees; modulo 4 to ensure range of [0, 1, 2, 3]
            self.direction = (self.direction + 1) % 4

        if action == 4:  # TURN COUNTERCLOCKWISE
            self.direction = (self.direction - 1) % 4

        self.sprite_loc()

        return next_location
    
    def spawn_beam(self, env: GridworldEnv, action):
        """Generate a beam extending cfg.agent.agent.beam_radius pixels
        out in front of the agent."""

        # Get the forward movement tile
        forward_vector = Vector(1, 0, direction=self.direction)
        right_vector = Vector(0, 1, direction=self.direction)
        left_vector = Vector(0, -1, direction=self.direction)

        # Candidate beam locations
        beam_locs = [
            Location(*self.location) + (forward_vector * 1), # 1 pixel ahead
            Location(*self.location) + (forward_vector * 2), # 2 pixel ahead
        ]

        # Check beam layer for walls...


    def pov(self, env) -> torch.Tensor:
        """
        Defines the agent's observation function
        """

        # If the environment is a full MDP, get the whole world image
        if env.full_mdp:
            image = visual_field_multilayer(
                env.world, env.color_map, channels=env.channels
            )
        # Otherwise, use the agent observation function
        else:
            image = visual_field_multilayer(
                env.world, env.color_map, self.location, self.vision, env.channels
            )

        current_state = torch.tensor(image).unsqueeze(0)

        return current_state

    def transition(self, env: GridworldEnv, state, action):
        """Changes the world based on action taken."""
        reward = 0

        # Attempt the transition
        attempted_location = self.act(action)

        # Get the candidate reward objects
        reward_locations = [
            (attempted_location[0], attempted_location[1], i)
            for i in range(env.world.shape[2])
        ]
        reward_objects = [env.observe(loc) for loc in reward_locations]

        # Complete the transition
        env.move(self, attempted_location)

        # Get the interaction reward
        for obj in reward_objects:
            reward += obj.value

        # Get the next state
        location_code = positional_embedding(self.location, env, 3, 3)
        next_state = np.concatenate([self.pov(env).flatten(), location_code]).reshape(
            1, -1
        )

        return reward, next_state, False

    def reset(self) -> None:
        self.episode_memory.clear()
        # self.init_replay()

# ------------------- #
# endregion           #
# ------------------- #

# ------------------- #
# region: Beams       #
# ------------------- #

class Beam(Object):
    """Generic beam class for agent beams."""
    def __init__(self, cfg, appearance):
        super().__init__(appearance)
        self.cfg = cfg
        self.sprite = f'{cfg.root}/examples/cleanup/assets/beam.png'

class CleanBeam(Beam):
    def __init__(self, cfg, appearance):
        super().__init__(cfg, appearance)

class ZapBeam(Beam):
    def __init__(self, cfg, appearance):
        super().__init__(cfg, appearance)
        self.sprite = f'{cfg.root}/examples/cleanup/assets/zap.png'                

# ------------------- #
# endregion           #
# ------------------- #

"""
-------------------
old functions below
-------------------
"""


def pov_old(self, env) -> torch.Tensor:
    """
    Defines the agent's observation function
    """
    # Get the previous state
    previous_state = self.episode_memory.get_last_memory("states")

    # Get the frames from the previous state
    current_state = previous_state.clone()

    current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

    # If the environment is a full MDP, get the whole world image
    if env.full_mdp:
        image = visual_field_multilayer(env.world, env.color_map, channels=env.channels)
    # Otherwise, use the agent observation function
    else:
        image = visual_field_multilayer(
            env.world, env.color_map, self.location, self.vision, env.channels
        )

    # Update the latest state to the observation
    state_now = torch.tensor(image).unsqueeze(0)
    current_state[:, -1, :, :, :] = state_now

    return current_state


def init_replay(self) -> None:
    """Fill in blank images for the LSTM."""

    priority = torch.tensor(0.1)
    num_frames = self.model.num_frames
    if self.cfg.env.full_mdp:
        state = torch.zeros(1, num_frames, *self.model.state_size).float()
    else:
        # Number of one-hot code channels
        C = len(self.appearance)
        H = W = self.vision * 2 + 1
        state = torch.zeros(1, num_frames, C, H, W).float()

    action = torch.tensor(7.0)  # Action outside the action space
    reward = torch.tensor(0.0)
    done = torch.tensor(0.0)
    exp = (priority, (state, action, reward, state, done))
    self.episode_memory.append(exp)
