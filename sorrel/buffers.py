from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

# Large constant for virtual index modulus (Buffer compatibility)
# Using a large value to avoid premature wrapping of virtual index
_VIRTUAL_INDEX_MODULUS = 1_000_000_000  # 10^9


class Buffer:
    """Buffer class for recording and storing agent actions.

    Attributes:
        capacity (int): The size of the replay buffer. Experiences are overwritten when the numnber of memories exceeds capacity.
        obs_shape (Sequence[int]): The shape of the observations. Used to structure the state buffer.
        states (np.ndarray): The state array.
        actions (np.ndarray): The action array.
        rewards (np.ndarray): The reward array.
        dones (np.ndarray): The done array.
        idx (int): The current position of the buffer.
        size (int): The current size of the array.
        n_frames (int): The number of frames to stack when sampling or creating empty frames between games.
    """

    def __init__(self, capacity: int, obs_shape: Sequence[int], n_frames: int = 1):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.n_frames = n_frames

    def add(self, obs, action, reward, done):
        """Add an experience to the replay buffer.

        Args:
            obs (np.ndarray): The observation/state.
            action (int): The action taken.
            reward (float): The reward received.
            done (bool): Whether the episode terminated after this step.
        """
        self.states[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_empty(self):
        """Advancing the id by `self.n_frames`, adding empty frames to the replay
        buffer."""
        self.idx = (self.idx + self.n_frames - 1) % self.capacity

    def sample(self, batch_size: int):
        """Sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing the states, actions, rewards, next states, dones, and
                invalid (meaning stacked frmaes cross episode boundary).
        """
        indices = np.random.choice(
            max(1, self.size - self.n_frames - 1), batch_size, replace=False
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

    def clear(self):
        """Zero out the arrays."""
        self.states = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def getidx(self):
        """Get the current index.

        Returns:
            int: The current index
        """
        return self.idx

    def current_state(self) -> np.ndarray:
        """Get the current state.

        Returns:
            np.ndarray: An array with the last `self.n_frames` observations stacked together as the current state.
        """

        if self.idx < (self.n_frames - 1):
            diff = self.idx - (self.n_frames - 1)
            return np.concatenate(
                (self.states[diff % self.capacity :], self.states[: self.idx])
            )
        return self.states[self.idx - (self.n_frames - 1) : self.idx]

    def __repr__(self):
        return f"Buffer(capacity={self.capacity}, obs_shape={self.obs_shape})"

    def __str__(self):
        return repr(self)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.dones[idx])


class EpisodeBuffer:
    """Replay buffer that stores entire episodes and supports both CPC and IQN style sampling.

    The ``EpisodeBuffer`` keeps a list of episodes where each episode is a sequence of
    (state, action, reward, done) tuples.  It exposes two sampling methods:

    * ``sample`` (alias for ``sample_iqn``) returns a batch of transitions suitable for value-based methods
      such as DQN/IQN.  Transitions are sampled uniformly from the buffer and
      optionally support stacking ``n_frames`` consecutive observations.  A
      ``valid`` mask is returned that is zero when the stacked frames cross an
      episode boundary.

    * ``sample_cpc`` returns a batch of transitions augmented with a future goal.
      For each sampled time step ``t`` within an episode, a future time step
      ``k > t`` is drawn (uniformly by default or weighted by a discount
      factor).  The observation at ``k`` is used as the goal for contrastive
      predictive coding.  This sampling scheme can be used for CPC/TD‑InfoNCE
      objectives where the agent must predict future states.

    Unlike the ``Buffer`` class which stores a ring of fixed capacity and
    overwrites old data, ``EpisodeBuffer`` organises its data by episode and
    automatically removes the oldest episode when the maximum number of episodes
    is reached.

    This class is designed to be a drop-in replacement for ``Buffer`` in most
    cases, while providing additional functionality for episode-aware sampling.

    Attributes:
        capacity (int): maximum number of episodes to store.  When the
            buffer exceeds this size, the oldest episode is automatically dropped.
            Default is 10 episodes.
        max_episode_length (int): maximum number of transitions per episode.
            Prevents unbounded memory growth if episodes never terminate.
            When exceeded, episode is automatically terminated. Default is 10000.
        obs_shape (Sequence[int]): shape of a single observation.
        n_frames (int): number of consecutive frames to stack when sampling.
        episodes (list): list of episodes, each episode is a dict with
            'states', 'actions', 'rewards' and 'dones' keys mapping to lists.
        num_transitions (int): total number of transitions currently stored.
            Bounded by capacity * max_episode_length (at most 10 * 10000 = 100k transitions).
        size (int): alias for num_transitions (for Buffer compatibility).
        idx (int): virtual index tracking position (for Buffer compatibility).
    """

    def __init__(self, capacity: int = 10, obs_shape: Sequence[int] = None, n_frames: int = 1, max_episode_length: int = 10000):
        """
        Initialize EpisodeBuffer.
        
        Args:
            capacity: Maximum number of episodes to store (default: 10).
                     When capacity is reached, oldest episodes are automatically dropped.
                     Must be > 0.
            obs_shape: Shape of a single observation (required).
            n_frames: Number of consecutive frames to stack when sampling (default: 1).
            max_episode_length: Maximum number of transitions per episode (default: 10000).
                              Prevents unbounded memory growth if episodes never terminate.
                              When exceeded, episode is automatically terminated and a new one starts.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if max_episode_length <= 0:
            raise ValueError(f"max_episode_length must be > 0, got {max_episode_length}")
        self.capacity = capacity
        self.max_episode_length = max_episode_length
        if obs_shape is None:
            raise ValueError("obs_shape is required for EpisodeBuffer")
        self.obs_shape = tuple(obs_shape)
        self.n_frames = n_frames
        # Use list for episodes (supports slicing natively)
        self.episodes: list[dict[str, list]] = []
        self.num_transitions: int = 0
        # Virtual index for Buffer compatibility (tracks position in flat transition space)
        self._idx: int = 0

    @property
    def size(self) -> int:
        """Alias for num_transitions (Buffer compatibility)."""
        return self.num_transitions

    @property
    def idx(self) -> int:
        """Virtual index tracking position (Buffer compatibility)."""
        return self._idx

    def _start_new_episode(self) -> None:
        """Create a new episode container and append to episodes list.
        
        If the list is at capacity, the oldest episode will be dropped
        and num_transitions will be updated accordingly.
        """
        # If list is at capacity, drop the oldest episode
        if len(self.episodes) >= self.capacity:
            # Remove oldest episode and update num_transitions
            oldest_ep = self.episodes.pop(0)
            transitions_to_remove = len(oldest_ep['states'])
            self.num_transitions -= transitions_to_remove
            # Update virtual index
            self._idx = max(0, self._idx - transitions_to_remove)
        
        # Append new episode
        self.episodes.append({
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': []
        })

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool) -> None:
        """Add a single transition to the buffer.

        Args:
            obs: observation for the current step.
            action: action taken at the current step.
            reward: reward received after taking the action.
            done: boolean flag indicating if the episode terminated after this step.
        """
        # If there are no episodes yet or the last episode ended, start a new episode
        if not self.episodes or (self.episodes and self.episodes[-1]['dones'] and self.episodes[-1]['dones'][-1]):
            self._start_new_episode()
        
        ep = self.episodes[-1]
        current_ep_length = len(ep['states'])
        
        # Prevent unbounded memory growth: if episode exceeds max length, terminate it
        # Check BEFORE adding to ensure we never exceed max_episode_length
        if current_ep_length >= self.max_episode_length:
            # Mark current episode as done and start a new one
            if current_ep_length > 0:
                ep['dones'][-1] = True  # Mark last transition as done
            self._start_new_episode()
            ep = self.episodes[-1]
        
        # Add transition to current episode
        ep['states'].append(obs.astype(np.float32))
        ep['actions'].append(int(action))
        ep['rewards'].append(float(reward))
        ep['dones'].append(bool(done))
        self.num_transitions += 1
        # Update virtual index for Buffer compatibility
        # This is only for compatibility - the actual buffer size is tracked by num_transitions
        self._idx = (self._idx + 1) % _VIRTUAL_INDEX_MODULUS

    def add_empty(self) -> None:
        """Add empty frames to the buffer (Buffer compatibility).

        For EpisodeBuffer, this advances the virtual index by n_frames.
        Note: This doesn't actually add empty frames since EpisodeBuffer
        is episode-structured, but maintains compatibility with Buffer interface.
        """
        # Use same modulus as add() for consistency
        self._idx = (self._idx + self.n_frames - 1) % _VIRTUAL_INDEX_MODULUS

    def __len__(self) -> int:
        return self.num_transitions

    def _get_episode_and_index(self, flat_index: int) -> tuple[int, int]:
        """Convert a flat transition index into episode index and within-episode index.

        Args:
            flat_index: integer between 0 and ``self.num_transitions - 1``.

        Returns:
            A tuple ``(ep_idx, step_idx)`` where ``ep_idx`` is the index of the
            episode in ``self.episodes`` and ``step_idx`` is the index of the
            transition within that episode.
        """
        count = 0
        for e_idx, ep in enumerate(self.episodes):
            ep_len = len(ep['states'])
            if flat_index < count + ep_len:
                return e_idx, flat_index - count
            count += ep_len
        raise IndexError(f"Flat index {flat_index} out of range")

    def sample(self, batch_size: int):
        """Sample a batch of experiences from the buffer (Buffer compatibility).

        This is an alias for ``sample_iqn()`` to maintain compatibility with
        the ``Buffer`` interface.

        Args:
            batch_size: number of transitions to sample.

        Returns:
            A tuple ``(states, actions, rewards, next_states, dones, valid)`` where
            ``states`` and ``next_states`` have shape ``(batch_size, n_frames * obs_dim)``.
        """
        return self.sample_iqn(batch_size)

    def sample_iqn(self, batch_size: int):
        """Sample a batch of transitions uniformly for IQN/DQN training.

        This method draws transitions uniformly from all stored transitions and
        constructs stacked states and next states with ``n_frames`` frames.
        It returns a tuple similar to :meth:`Buffer.sample` with an
        additional ``valid`` mask indicating whether the stacked frames
        cross an episode boundary.

        Args:
            batch_size: number of transitions to sample.

        Returns:
            A tuple ``(states, actions, rewards, next_states, dones, valid)`` where
            ``states`` and ``next_states`` have shape ``(batch_size, n_frames * obs_dim)``.
        """
        if self.num_transitions < self.n_frames + 1:
            raise ValueError("Not enough transitions in the buffer to sample")
        # Flatten all transitions indices
        max_valid_start = self.num_transitions - self.n_frames - 1
        indices = np.random.choice(max_valid_start, batch_size, replace=False)
        # Containers for batch
        states_batch = np.zeros((batch_size, self.n_frames * int(np.prod(self.obs_shape))), dtype=np.float32)
        next_states_batch = np.zeros_like(states_batch)
        actions_batch = np.zeros((batch_size, 1), dtype=np.int64)
        rewards_batch = np.zeros((batch_size, 1), dtype=np.float32)
        dones_batch = np.zeros((batch_size, 1), dtype=np.float32)
        valid_batch = np.ones((batch_size, 1), dtype=np.float32)

        # Iterate over sampled indices and build stacked observations
        for i, flat_idx in enumerate(indices):
            # Determine episode and step index for the start of the stack
            ep_idx, step_idx = self._get_episode_and_index(flat_idx)
            # Build stacked state and next_state
            frames = []
            next_frames = []
            valid_flag = True
            for f in range(self.n_frames):
                # Get frame at t+f
                ep_i, step_i = self._get_episode_and_index(flat_idx + f)
                frame_obs = self.episodes[ep_i]['states'][step_i]
                frames.append(frame_obs)
                # Check if done before final frame
                if f < self.n_frames - 1:
                    # done at this step invalidates future stacking
                    if self.episodes[ep_i]['dones'][step_i]:
                        valid_flag = False
                # Build next frame for t+f+1
                ep_j, step_j = self._get_episode_and_index(flat_idx + f + 1)
                next_frame_obs = self.episodes[ep_j]['states'][step_j]
                next_frames.append(next_frame_obs)
            # Flatten stacked frames
            states_batch[i] = np.concatenate(frames).reshape(-1)
            next_states_batch[i] = np.concatenate(next_frames).reshape(-1)
            # Assign action, reward, done at last frame
            ep_last, step_last = self._get_episode_and_index(flat_idx + self.n_frames - 1)
            actions_batch[i, 0] = self.episodes[ep_last]['actions'][step_last]
            rewards_batch[i, 0] = self.episodes[ep_last]['rewards'][step_last]
            dones_batch[i, 0] = float(self.episodes[ep_last]['dones'][step_last])
            valid_batch[i, 0] = 1.0 if valid_flag else 0.0

        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, valid_batch

    def sample_cpc(self, batch_size: int, discount: float = 1.0):
        """Sample a batch of transitions with future goals for CPC training.

        For each sampled transition at time ``t``, a future time step ``k > t``
        within the same episode is sampled (uniformly when ``discount=1.0`` or
        biased towards closer futures with ``discount < 1.0``).  The goal is
        the observation at step ``k``.  Stacked states and next states are
        constructed similarly to :meth:`sample_iqn`.

        Args:
            batch_size: number of transitions to sample.
            discount: geometric discount factor for sampling the future goal.
                When ``discount < 1.0``, nearer futures are more likely.

        Returns:
            A tuple ``(states, actions, rewards, next_states, dones, goals)`` where
            ``goals`` has shape ``(batch_size, n_frames * obs_dim)`` when
            ``n_frames`` frames are stacked.
        """
        if self.num_transitions < self.n_frames + 2:
            raise ValueError("Not enough transitions to sample CPC batches")
        states_batch = np.zeros((batch_size, self.n_frames * int(np.prod(self.obs_shape))), dtype=np.float32)
        next_states_batch = np.zeros_like(states_batch)
        goals_batch = np.zeros_like(states_batch)
        actions_batch = np.zeros((batch_size, 1), dtype=np.int64)
        rewards_batch = np.zeros((batch_size, 1), dtype=np.float32)
        dones_batch = np.zeros((batch_size, 1), dtype=np.float32)

        for i in range(batch_size):
            # Sample an episode that has at least 2 transitions
            valid_eps = [e for e in self.episodes if len(e['states']) > 1]
            if not valid_eps:
                raise ValueError("No episode with enough steps for CPC sampling")
            ep = valid_eps[np.random.randint(len(valid_eps))]
            ep_len = len(ep['states'])
            # Choose starting index such that there is room for n_frames and one future step
            max_start = ep_len - (self.n_frames + 1)
            if max_start < 0:
                continue  # Shouldn't happen due to valid_eps condition
            start = np.random.randint(0, max_start + 1)
            # Determine future index > start
            future_candidates = np.arange(start + 1, ep_len)
            if discount < 1.0:
                # Compute discounted weights; distances from start
                distances = future_candidates - start
                weights = discount ** distances
                # Normalize to probability distribution
                prob = weights / weights.sum()
                future_idx = np.random.choice(future_candidates, p=prob)
            else:
                future_idx = np.random.choice(future_candidates)
            # Build stacked state and next_state from the episode
            frames = ep['states'][start:start + self.n_frames]
            next_frames = ep['states'][start + 1:start + 1 + self.n_frames]
            # When n_frames > 1 and the episode ended before full stack, pad with last frame
            if len(frames) < self.n_frames:
                pad = [frames[-1]] * (self.n_frames - len(frames))
                frames = frames + pad
            if len(next_frames) < self.n_frames:
                pad = [next_frames[-1]] * (self.n_frames - len(next_frames))
                next_frames = next_frames + pad
            states_batch[i] = np.concatenate(frames).reshape(-1)
            next_states_batch[i] = np.concatenate(next_frames).reshape(-1)
            # Assign action, reward, done at the last frame of the stack
            last_idx = start + self.n_frames - 1
            actions_batch[i, 0] = ep['actions'][last_idx]
            rewards_batch[i, 0] = ep['rewards'][last_idx]
            dones_batch[i, 0] = float(ep['dones'][last_idx])
            # Goal is the stacked observation at the future index; we stack n_frames frames
            goal_frames = ep['states'][future_idx:future_idx + self.n_frames]
            if len(goal_frames) < self.n_frames:
                pad = [goal_frames[-1]] * (self.n_frames - len(goal_frames))
                goal_frames = goal_frames + pad
            goals_batch[i] = np.concatenate(goal_frames).reshape(-1)
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, goals_batch

    def clear(self) -> None:
        """Clear all stored episodes and reset counters."""
        self.episodes.clear()
        self.num_transitions = 0
        self._idx = 0

    def getidx(self) -> int:
        """Get the current virtual index (Buffer compatibility).

        Returns:
            int: The current virtual index
        """
        return self._idx

    def current_state(self) -> np.ndarray:
        """Get the current state (Buffer compatibility).

        Returns the last ``n_frames`` observations from the current episode
        (or last episode if current episode is empty), stacked together.

        Returns:
            np.ndarray: An array with shape (n_frames, obs_dim) containing the last ``n_frames`` observations.
                       This matches Buffer.current_state() behavior for compatibility.
        """
        obs_dim = int(np.prod(self.obs_shape))
        
        if not self.episodes:
            # No episodes yet, return zeros with correct shape
            return np.zeros((self.n_frames, obs_dim), dtype=np.float32)
        
        # Get the last episode (current episode)
        ep = self.episodes[-1]
        if not ep['states']:
            # Current episode is empty, try previous episode
            if len(self.episodes) > 1:
                ep = self.episodes[-2]
            else:
                return np.zeros((self.n_frames, obs_dim), dtype=np.float32)
        
        # Get last n_frames from the episode
        states = ep['states']
        if len(states) >= self.n_frames:
            frames = states[-self.n_frames:]
        else:
            # Pad with the last frame if not enough frames
            if len(states) > 0:
                frames = states + [states[-1]] * (self.n_frames - len(states))
            else:
                # No states at all, return zeros
                return np.zeros((self.n_frames, obs_dim), dtype=np.float32)
        
        # Stack frames into 2D array: (n_frames, obs_dim)
        # Each frame should be shape (obs_dim,), so we can stack them
        frames_array = np.array(frames)  # Shape: (n_frames, obs_dim) if frames are 1D
        
        # Ensure correct shape
        if frames_array.ndim == 1:
            # Single frame case (shouldn't happen with n_frames > 1, but handle gracefully)
            frames_array = frames_array.reshape(1, -1)
            # Pad to n_frames
            if frames_array.shape[0] < self.n_frames:
                padding = np.tile(frames_array[-1:], (self.n_frames - frames_array.shape[0], 1))
                frames_array = np.vstack([frames_array, padding])
        elif frames_array.ndim == 2 and frames_array.shape[0] < self.n_frames:
            # Not enough frames, pad with last frame
            padding = np.tile(frames_array[-1:], (self.n_frames - frames_array.shape[0], 1))
            frames_array = np.vstack([frames_array, padding])
        
        return frames_array

    def __getitem__(self, idx: int):
        """Get a transition by flat index (Buffer compatibility).

        Args:
            idx: flat index into the buffer (0 to num_transitions - 1).

        Returns:
            A tuple ``(state, action, reward, done)`` for the transition at index ``idx``.
        """
        if idx < 0 or idx >= self.num_transitions:
            raise IndexError(f"Index {idx} out of range [0, {self.num_transitions})")
        ep_idx, step_idx = self._get_episode_and_index(idx)
        ep = self.episodes[ep_idx]
        return (
            ep['states'][step_idx],
            ep['actions'][step_idx],
            ep['rewards'][step_idx],
            ep['dones'][step_idx]
        )

    def __repr__(self) -> str:
        return f"EpisodeBuffer(capacity={self.capacity}, obs_shape={self.obs_shape}, n_frames={self.n_frames})"

    def __str__(self) -> str:
        return repr(self)
    
    def sample_sequences(self, batch_size: int, seq_len: int):
        """
        Sample sequences of fixed length from stored episodes.
        
        This method samples `batch_size` sequences, each of length `seq_len`,
        ensuring that sequences don't cross episode boundaries. If not enough
        valid sequences are available, returns None.
        
        Args:
            batch_size: Number of sequences to sample
            seq_len: Length of each sequence
            
        Returns:
            Tuple of numpy arrays (states, actions, rewards, dones) or None if
            insufficient data. Each array has shape (batch_size, seq_len, ...).
        """
        if self.num_transitions < seq_len:
            return None
        
        # Find all valid sequence start positions (within episodes, with enough room)
        valid_starts = []
        for ep_idx, ep in enumerate(self.episodes):
            ep_len = len(ep['states'])
            # Can start sequences from index 0 to ep_len - seq_len
            for start in range(max(0, ep_len - seq_len + 1)):
                valid_starts.append((ep_idx, start))
        
        if len(valid_starts) < batch_size:
            return None
        
        # Sample batch_size sequence starts
        selected = np.random.choice(len(valid_starts), batch_size, replace=False)
        
        # Prepare output arrays
        obs_dim = int(np.prod(self.obs_shape))
        states_seq = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
        actions_seq = np.zeros((batch_size, seq_len), dtype=np.int64)
        rewards_seq = np.zeros((batch_size, seq_len), dtype=np.float32)
        dones_seq = np.zeros((batch_size, seq_len), dtype=np.float32)
        
        # Fill arrays with sampled sequences
        for i, idx in enumerate(selected):
            ep_idx, start = valid_starts[idx]
            ep = self.episodes[ep_idx]
            
            # Extract sequence from episode
            end = start + seq_len
            states_seq[i] = np.array(ep['states'][start:end]).reshape(seq_len, obs_dim)
            actions_seq[i] = np.array(ep['actions'][start:end])
            rewards_seq[i] = np.array(ep['rewards'][start:end])
            dones_seq[i] = np.array(ep['dones'][start:end], dtype=np.float32)
        
        return states_seq, actions_seq, rewards_seq, dones_seq