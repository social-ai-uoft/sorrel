"""Shared memory replay buffer for multiprocessing.

This class extends Buffer to use shared memory, making it a drop-in replacement
that works exactly like the original Buffer but can be shared across processes.
"""

import multiprocessing as mp
import os
from multiprocessing import shared_memory
from typing import Sequence

import numpy as np

from sorrel.buffers import Buffer


class SharedReplayBuffer(Buffer):
    """Shared memory replay buffer for multiprocessing.
    
    This class extends Buffer to use shared memory for multiprocessing.
    It behaves exactly like the original Buffer class, with the only difference
    being that the underlying arrays are stored in shared memory.
    
    All methods work exactly as in the original Buffer class.
    """
    
    def __init__(self, capacity: int, obs_shape: Sequence[int], n_frames: int = 1, 
                 create: bool = True, shm_names: dict = None, idx: mp.Value = None, 
                 size: mp.Value = None):
        """Initialize shared replay buffer.
        
        Args:
            capacity: Buffer capacity
            obs_shape: Observation shape (same as original Buffer)
            n_frames: Number of frames to stack (same as original Buffer)
            create: If True, create new shared memory. If False, attach to existing.
            shm_names: Dictionary with shared memory names (for attaching)
            idx: Shared mp.Value for index (if attaching to existing buffer)
            size: Shared mp.Value for size (if attaching to existing buffer)
        """
        # Store capacity, obs_shape, n_frames (same as original)
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.n_frames = n_frames
        
        # Calculate buffer sizes
        state_size = int(np.prod(obs_shape)) * capacity
        action_size = capacity
        reward_size = capacity
        done_size = capacity
        
        if create:
            # Generate unique names for shared memory
            pid = os.getpid()
            self.shm_name_states = f"shm_states_{pid}_{id(self)}"
            self.shm_name_actions = f"shm_actions_{pid}_{id(self)}"
            self.shm_name_rewards = f"shm_rewards_{pid}_{id(self)}"
            self.shm_name_dones = f"shm_dones_{pid}_{id(self)}"
            
            # Create shared memory blocks
            try:
                self.shm_states = shared_memory.SharedMemory(
                    create=True, 
                    size=state_size * 4,  # float32 = 4 bytes
                    name=self.shm_name_states
                )
                self.shm_actions = shared_memory.SharedMemory(
                    create=True,
                    size=action_size * 8,  # int64 = 8 bytes
                    name=self.shm_name_actions
                )
                self.shm_rewards = shared_memory.SharedMemory(
                    create=True,
                    size=reward_size * 4,  # float32 = 4 bytes
                    name=self.shm_name_rewards
                )
                self.shm_dones = shared_memory.SharedMemory(
                    create=True,
                    size=done_size * 4,  # float32 = 4 bytes
                    name=self.shm_name_dones
                )
            except FileExistsError:
                # Clean up if already exists
                self._cleanup_existing()
                raise
            
            # Create numpy arrays backed by shared memory (same shape as original)
            self.states = np.ndarray(
                (capacity, *obs_shape), 
                dtype=np.float32,
                buffer=self.shm_states.buf
            )
            self.actions = np.ndarray(
                capacity, 
                dtype=np.int64,
                buffer=self.shm_actions.buf
            )
            self.rewards = np.ndarray(
                capacity,
                dtype=np.float32,
                buffer=self.shm_rewards.buf
            )
            self.dones = np.ndarray(
                capacity,
                dtype=np.float32,
                buffer=self.shm_dones.buf
            )
            
            # Initialize arrays to zero (same as original)
            self.states.fill(0)
            self.actions.fill(0)
            self.rewards.fill(0)
            self.dones.fill(0)
            
            # Atomic indices using multiprocessing.Value (shared across processes)
            self._idx = mp.Value('i', 0)
            self._size = mp.Value('i', 0)
            
            # Store names for passing to other processes
            self.shm_names = {
                'states': self.shm_name_states,
                'actions': self.shm_name_actions,
                'rewards': self.shm_name_rewards,
                'dones': self.shm_name_dones,
            }
        else:
            # Attach to existing shared memory
            if shm_names is None:
                raise ValueError("shm_names required when create=False")
            if idx is None or size is None:
                raise ValueError("idx and size required when create=False")
            
            self.shm_name_states = shm_names['states']
            self.shm_name_actions = shm_names['actions']
            self.shm_name_rewards = shm_names['rewards']
            self.shm_name_dones = shm_names['dones']
            
            self.shm_states = shared_memory.SharedMemory(name=self.shm_name_states)
            self.shm_actions = shared_memory.SharedMemory(name=self.shm_name_actions)
            self.shm_rewards = shared_memory.SharedMemory(name=self.shm_name_rewards)
            self.shm_dones = shared_memory.SharedMemory(name=self.shm_name_dones)
            
            # Create numpy arrays from existing shared memory
            self.states = np.ndarray(
                (capacity, *obs_shape),
                dtype=np.float32,
                buffer=self.shm_states.buf
            )
            self.actions = np.ndarray(
                capacity,
                dtype=np.int64,
                buffer=self.shm_actions.buf
            )
            self.rewards = np.ndarray(
                capacity,
                dtype=np.float32,
                buffer=self.shm_rewards.buf
            )
            self.dones = np.ndarray(
                capacity,
                dtype=np.float32,
                buffer=self.shm_dones.buf
            )
            
            # Use provided shared indices
            self._idx = idx
            self._size = size
    
    def __getstate__(self):
        """Custom pickle support for multiprocessing.
        
        When pickled, we need to store the information needed to recreate
        the buffer in the subprocess by attaching to existing shared memory.
        """
        state = {
            'capacity': self.capacity,
            'obs_shape': self.obs_shape,
            'n_frames': self.n_frames,
            'shm_names': self.shm_names,
            '_idx': self._idx,
            '_size': self._size,
        }
        return state
    
    def __setstate__(self, state):
        """Custom unpickle support for multiprocessing.
        
        When unpickled in subprocess, attach to existing shared memory
        using the stored names.
        """
        self.capacity = state['capacity']
        self.obs_shape = state['obs_shape']
        self.n_frames = state['n_frames']
        self.shm_names = state['shm_names']
        self._idx = state['_idx']
        self._size = state['_size']
        
        # Attach to existing shared memory
        self.shm_name_states = self.shm_names['states']
        self.shm_name_actions = self.shm_names['actions']
        self.shm_name_rewards = self.shm_names['rewards']
        self.shm_name_dones = self.shm_names['dones']
        
        self.shm_states = shared_memory.SharedMemory(name=self.shm_name_states)
        self.shm_actions = shared_memory.SharedMemory(name=self.shm_name_actions)
        self.shm_rewards = shared_memory.SharedMemory(name=self.shm_name_rewards)
        self.shm_dones = shared_memory.SharedMemory(name=self.shm_name_dones)
        
        # Create numpy arrays from existing shared memory
        self.states = np.ndarray(
            (self.capacity, *self.obs_shape),
            dtype=np.float32,
            buffer=self.shm_states.buf
        )
        self.actions = np.ndarray(
            self.capacity,
            dtype=np.int64,
            buffer=self.shm_actions.buf
        )
        self.rewards = np.ndarray(
            self.capacity,
            dtype=np.float32,
            buffer=self.shm_rewards.buf
        )
        self.dones = np.ndarray(
            self.capacity,
            dtype=np.float32,
            buffer=self.shm_dones.buf
        )
    
    @property
    def idx(self):
        """Get current index (works like original Buffer.idx)."""
        return self._idx.value
    
    @idx.setter
    def idx(self, value):
        """Set current index (works like original Buffer.idx)."""
        with self._idx.get_lock():
            self._idx.value = value
    
    @property
    def size(self):
        """Get current size (works like original Buffer.size)."""
        return self._size.value
    
    @size.setter
    def size(self, value):
        """Set current size (works like original Buffer.size)."""
        with self._size.get_lock():
            self._size.value = value
    
    def _cleanup_existing(self):
        """Clean up existing shared memory if it exists."""
        for name in [self.shm_name_states, self.shm_name_actions, 
                     self.shm_name_rewards, self.shm_name_dones]:
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
    
    def add(self, obs, action, reward, done):
        """Add an experience to the replay buffer.
        
        This method works exactly like the original Buffer.add().
        The only difference is thread-safe access to idx and size.
        
        Args:
            obs (np.ndarray): The observation/state.
            action (int): The action taken.
            reward (float): The reward received.
            done (bool): Whether the episode terminated after this step.
        """
        # Combine both operations in a single lock to reduce overhead
        with self._idx.get_lock():
            current_idx = self._idx.value
            self._idx.value = (current_idx + 1) % self.capacity
            # Also update size in the same lock (faster than two separate locks)
            old_size = self._size.value
            self._size.value = min(old_size + 1, self.capacity)
        
        # Write to arrays (same as original) - no lock needed for array writes
        self.states[current_idx] = obs
        self.actions[current_idx] = action
        self.rewards[current_idx] = reward
        self.dones[current_idx] = done
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences from the replay buffer.
        
        This method works exactly like the original Buffer.sample().
        The only difference is thread-safe access to size.
        
        Args:
            batch_size (int): The number of experiences to sample.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing the states, actions, rewards, next states, dones, and
                invalid (meaning stacked frames cross episode boundary).
        """
        # Get current size (lock-free read - size can be slightly stale but that's okay)
        # We only need approximate size for the check, exact size isn't critical
        current_size = self._size.value
        
        # Check if we have enough samples (same as original Buffer logic)
        available_samples = max(1, current_size - self.n_frames - 1)
        if available_samples < batch_size:
            # Not enough samples yet - return None (learner will wait)
            return None
        
        # Same logic as original Buffer.sample()
        indices = np.random.choice(
            available_samples, batch_size, replace=False
        )
        indices = indices[:, np.newaxis]
        indices = indices + np.arange(self.n_frames)
        
        states = self.states[indices].reshape(batch_size, -1)
        next_states = self.states[indices + 1].reshape(batch_size, -1)
        actions = self.actions[indices[:, -1]].reshape(batch_size, -1)
        rewards = self.rewards[indices[:, -1]].reshape(batch_size, -1)
        dones = self.dones[indices[:, -1]].reshape(batch_size, -1)
        valid = (1.0 - np.any(self.dones[indices[:, :-1]], axis=-1)).reshape(
            batch_size, -1
        )
        
        return states, actions, rewards, next_states, dones, valid
    
    def add_empty(self):
        """Advancing the id by `self.n_frames`, adding empty frames to the replay buffer.
        
        This method works exactly like the original Buffer.add_empty().
        """
        with self._idx.get_lock():
            self._idx.value = (self._idx.value + self.n_frames - 1) % self.capacity
    
    def clear(self):
        """Zero out the arrays.
        
        This method works exactly like the original Buffer.clear().
        """
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        with self._idx.get_lock():
            self._idx.value = 0
        with self._size.get_lock():
            self._size.value = 0
    
    def getidx(self):
        """Get the current index.
        
        Returns:
            int: The current index
        """
        return self.idx
    
    def current_state(self) -> np.ndarray:
        """Get the current state.
        
        This method works exactly like the original Buffer.current_state().
        
        Returns:
            np.ndarray: An array with the last `self.n_frames` observations stacked together as the current state.
        """
        current_idx = self.idx
        if current_idx < (self.n_frames - 1):
            diff = current_idx - (self.n_frames - 1)
            return np.concatenate(
                (self.states[diff % self.capacity :], self.states[: current_idx])
            )
        return self.states[current_idx - (self.n_frames - 1) : current_idx]
    
    def __len__(self):
        """Get current buffer size."""
        return self.size
    
    def __getitem__(self, idx):
        """Get item by index (same as original)."""
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.dones[idx])
    
    def cleanup(self):
        """Clean up shared memory resources.
        
        This should be called when the buffer is no longer needed.
        Only the process that created the shared memory should call unlink().
        """
        try:
            # Close shared memory handles
            if hasattr(self, 'shm_states'):
                self.shm_states.close()
            if hasattr(self, 'shm_actions'):
                self.shm_actions.close()
            if hasattr(self, 'shm_rewards'):
                self.shm_rewards.close()
            if hasattr(self, 'shm_dones'):
                self.shm_dones.close()
            
            # Unlink shared memory (remove from system)
            # Only unlink if we created it (not if we attached to existing)
            if hasattr(self, 'shm_name_states') and hasattr(self, 'shm_states'):
                try:
                    self.shm_states.unlink()
                except (FileNotFoundError, PermissionError):
                    pass  # Already unlinked or not owned by this process
            if hasattr(self, 'shm_name_actions') and hasattr(self, 'shm_actions'):
                try:
                    self.shm_actions.unlink()
                except (FileNotFoundError, PermissionError):
                    pass
            if hasattr(self, 'shm_name_rewards') and hasattr(self, 'shm_rewards'):
                try:
                    self.shm_rewards.unlink()
                except (FileNotFoundError, PermissionError):
                    pass
            if hasattr(self, 'shm_name_dones') and hasattr(self, 'shm_dones'):
                try:
                    self.shm_dones.unlink()
                except (FileNotFoundError, PermissionError):
                    pass
        except Exception as e:
            # Silently handle any cleanup errors
            pass


