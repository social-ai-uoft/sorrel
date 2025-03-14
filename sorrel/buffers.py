from __future__ import annotations

# --------------- #
# region: Imports #
# --------------- #

# Standard Python library imports for data structures and randomness
import numpy as np
import torch

from typing import Sequence

# --------------- #
# endregion       #
# --------------- #
    

class Buffer:

    def __init__(self, capacity: int, obs_shape: Sequence[int]):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.buffer = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0
    
    def add(self, obs, action, reward, done):
        """
        Add an experience to the replay buffer.

        Args:
            obs (np.ndarray): The observation/state.
            action (int): The action taken.
            reward (float): The reward received.
            done (bool): Whether the episode terminated after this step.
        """
        self.buffer[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, stacked_frames: int = 1):
        """
        Sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.
            stacked_frames (int): The number of frames to stack together.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                A tuple containing the states, actions, rewards, next states, dones, and 
                invalid (meaning stacked frmaes cross episode boundary).
        """
        indices = np.random.choice(max(1, self.size - stacked_frames - 1),  batch_size, replace=False)
        indices = indices[:, np.newaxis]
        indices = (indices + np.arange(stacked_frames))

        states = torch.from_numpy(self.buffer[indices]).view(batch_size, -1)
        next_states = torch.from_numpy(self.buffer[indices + 1]).view(batch_size, -1)
        actions = torch.from_numpy(self.actions[indices[:, -1]]).view(batch_size, -1)
        rewards  = torch.from_numpy(self.rewards[indices[:, -1]]).view(batch_size, -1)
        dones = torch.from_numpy(self.dones[indices[:, -1]]).view(batch_size, -1)
        valid = torch.from_numpy(1. - np.any(self.dones[indices[:, :-1]], axis=-1)).view(batch_size, -1)

        return states, actions, rewards, next_states, dones, valid
    
    def clear(self):
        """Zero out the arrays."""
        self.buffer = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def getidx(self):
        return self.idx
    
    def current_state(self, stacked_frames=1):
        if self.idx < stacked_frames:
            diff = self.idx - stacked_frames
            return np.concatenate((self.buffer[diff%len(self):], self.buffer[:self.idx]))
        return self.buffer[self.idx-stacked_frames:self.idx]
    
    def __repr__(self):
        return f"Buffer(capacity={self.capacity}, obs_shape={self.obs_shape})"
    
    def __str__(self):
        return repr(self)
    
    def __len__(self):
        return self.capacity