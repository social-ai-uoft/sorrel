from matplotlib import pyplot as plt
import numpy as np
from IPython.display import clear_output

import sys, os 
sys.path.append(os.path.abspath("."))
from agentarium.models.DDQN import ClaasyReplayBuffer as Buffer

class ModelHumanPlayer:

    def __init__(self, action_space, state_size, memory_size):
        self.name = "human"
        self.action_space = np.arange(action_space)
        self.state_size = state_size
        self.memory_size = memory_size
        self.num_frames = memory_size
        self.memory = self.memory = Buffer(
            capacity=memory_size,
            obs_shape=(np.array(self.state_size).prod(),)
        )
        self.show = False

    def take_action(self, state):
        
        if self.show:
            clear_output(wait = True)
            for i in range(self.memory_size):
                frame = state[:, i, :, :, :].squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
                plt.subplot(1, self.memory_size, i+1)
                plt.imshow(frame)
            plt.show()
        
        done = 0
        while done == 0:
            action_ = input("Select Action: ")
            if action_ in ["w", "a", "s", "d"]:
                if action_ == "w":
                    action = 0
                elif action_ == "s":
                    action = 1
                elif action_ == "a":
                    action = 2
                elif action_ == "d":
                    action = 3
            elif action_ in [str(act) for act in self.action_space]:
                action = int(action_)
            else:
                print("Please try again. Possible actions are below.")
                print(self.action_space)
                # we can have iinputType above also be joystick, or other controller
            if action is not None:
                if action in self.action_space:
                    done = 1

        return action


