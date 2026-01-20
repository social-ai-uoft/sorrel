"""Implicit Quantile Network Implementation.

The IQN learns an estimate of the entire distribution of possible rewards (Q-values) for taking
some action.

Source code is based on Dittert, Sebastian. "Implicit Quantile Networks (IQN) for Distributional
Reinforcement Learning and Extensions." https://github.com/BY571/IQN. (2020).

Structure:

IQN
 - calc_cos: calculate the cos values
 - forward: input pass through linear layer, get modified by cos values, pass through NOISY linear layer, and calculate output based on value and advantage
 - get_qvalues: set action probabilities as the mean of the quantiles

iRainbowModel (contains two IQN networks; one for local and one for target)
 - take_action: standard epsilon greedy action selection
 - train_step: train the model using quantile huber loss from IQN
 - soft_update: set weights of target network to be a mixture of weights from local and target network
"""

# ------------------------ #
# region: Imports          #
# ------------------------ #

# Import base packages
import random
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from sorrel.buffers import Buffer

# Import sorrel-specific packages
from sorrel.models.pytorch.layers import NoisyLinear
from sorrel.models.pytorch.pytorch_base import DoublePyTorchModel

# ------------------------ #
# endregion                #
# ------------------------ #

# ------------------------ #
# region: IQN              #
# ------------------------ #


class IQN(nn.Module):
    """The IQN Q-network."""

    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        seed: int,
        n_quantiles: int,
        n_frames: int = 5,
        device: str | torch.device = "cpu",
        use_factored_actions: bool = False,
        action_dims: Sequence[int] | None = None,
        factored_target_variant: str = "A",
    ) -> None:

        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = np.array(input_size)
        self.state_dim = len(self.input_shape)
        self.action_space = action_space
        self.n_quantiles = n_quantiles
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = (
            torch.FloatTensor([np.pi * i for i in range(1, self.n_cos + 1)])
            .view(1, 1, self.n_cos)
            .to(device)
        )
        self.device = device

        # Factored action space parameters
        self.use_factored_actions = use_factored_actions
        if use_factored_actions:
            if action_dims is None:
                raise ValueError("action_dims must be provided when use_factored_actions=True")
            self.action_dims = tuple(action_dims)
            self.n_action_dims = len(action_dims)
            # Validate that prod(action_dims) == action_space
            if np.prod(action_dims) != action_space:
                raise ValueError(
                    f"prod(action_dims)={np.prod(action_dims)} must equal action_space={action_space}"
                )
            self.factored_target_variant = factored_target_variant
            if factored_target_variant not in ["A", "B"]:
                raise ValueError(f"factored_target_variant must be 'A' or 'B', got '{factored_target_variant}'")
        else:
            self.action_dims = None
            self.n_action_dims = 0
            self.factored_target_variant = "A"

        # Network architecture
        self.head1 = nn.Linear(n_frames * self.input_shape.prod(), layer_size)

        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = NoisyLinear(layer_size, layer_size)
        self.cos_layer_out = layer_size

        # Original architecture (always created for backward compatibility)
        self.advantage = NoisyLinear(layer_size, action_space)
        self.value = NoisyLinear(layer_size, 1)
        
        # Factored architecture (only created when use_factored_actions=True)
        if use_factored_actions:
            self.quantile_heads = nn.ModuleList([
                NoisyLinear(layer_size, n_d) for n_d in action_dims
            ])
        else:
            self.quantile_heads = None

    def calc_cos(
        self, batch_size: int, n_tau: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculating the cosine values depending on the number of tau samples.

        Args:
            batch_size (int): The batch size.
            n_tau (int): The number of tau samples.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The cosine values and tau samples.
        """
        taus = (
            torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device)
        )  # (batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def forward(
        self, input: torch.Tensor, n_tau: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantile Calculation depending on the number of tau.

        Returns:
            tuple: quantiles, torch.Tensor (size: (batch_size, n_tau, action_space)); taus, torch.Tensor (size) ((batch_size, n_tau, 1))
        """
        # REMOVED: as suggested by Claude and GPT, input is not an image, so no need to add noise or normalize
        # Add noise to the input
        # eps = 0.01
        # noise = torch.rand_like(input) * eps
        # input = input / 255.0
        # input = input + noise

        # Flatten the input from [1, N, 7, 9, 9] to [1, N * 7 * 9 * 9]
        # batch_size, timesteps, C, H, W = input.size()
        # c_out = input.view(batch_size * timesteps, C, H, W)
        # r_in = c_out.view(batch_size, -1)

        batch_size = input.size()[0]
        r_in = input.view(batch_size, -1)

        # Pass input through linear layer and activation function ([1, 250])
        x = self.head1(r_in)
        x = torch.relu(x)

        # Calculate cos values
        cos, taus = self.calc_cos(
            batch_size, n_tau
        )  # cos.shape = (batch, n_tau, layer_size)
        cos = cos.view(batch_size * n_tau, self.n_cos)  # (1 * 12, 64)

        # Pass cos through linear layer and activation function
        cos = self.cos_embedding(cos)
        cos = torch.relu(cos)
        cos_x = cos.view(
            batch_size, n_tau, self.cos_layer_out
        )  # cos_x.shape = (batch, n_tau, layer_size)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * n_tau, self.cos_layer_out)

        # Pass input through NOISY linear layer and activation function ([1, 250])
        x = self.ff_1(x)
        x = torch.relu(x)

        if self.use_factored_actions:
            # Branching architecture: each head outputs quantiles for its action dimension
            quantiles_list = []
            for d, head in enumerate(self.quantile_heads):
                quantiles_d = head(x)  # (batch*n_tau, n_d)
                quantiles_d = quantiles_d.view(batch_size, n_tau, self.action_dims[d])
                quantiles_list.append(quantiles_d)
            return quantiles_list, taus
        else:
            # Original architecture
            # Calculate output based on value and advantage
            advantage = self.advantage(x)
            value = self.value(x)
            out = value + advantage - advantage.mean(dim=1, keepdim=True)

            return out.view(batch_size, n_tau, self.action_space), taus

    def get_qvalues(self, inputs, is_eval=False):
        if is_eval:
            n_tau = 256
        else:
            n_tau = self.n_quantiles
        
        forward_output = self.forward(inputs, n_tau)
        quantiles, _ = forward_output
        
        if self.use_factored_actions:
            # Return list of Q-value tensors, one for each branch
            qvalues_list = [q.mean(dim=1) for q in quantiles]  # Mean over quantiles
            return qvalues_list
        else:
            # Original behavior: single Q-value tensor
            actions = quantiles.mean(dim=1)
            return actions


# ------------------------ #
# endregion                #
# ------------------------ #

# ------------------------ #
# region: iRainbow         #
# ------------------------ #


class iRainbowModel(DoublePyTorchModel):
    """A combination of IQN with Rainbow, which itself combines priority experience
    replay, dueling DDQN, distributional DQN, noisy DQN, and multi-step return."""

    def __init__(
        # Base ANN parameters
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        epsilon_min: float,
        device: str | torch.device,
        seed: int,
        # iRainbow parameters
        n_frames: int,
        n_step: int,
        sync_freq: int,
        model_update_freq: int,
        batch_size: int,
        memory_size: int,
        LR: float,
        TAU: float,
        GAMMA: float,
        n_quantiles: int,
        # Factored action space parameters
        use_factored_actions: bool = False,
        action_dims: Sequence[int] | None = None,
        factored_target_variant: str = "A",
    ):
        """Initialize an iRainbow model.

        Args:
            input_size (Sequence[int]): The dimension of each state.
            action_space (int): The number of possible actions.
            layer_size (int): The size of the hidden layer.
            epsilon (float): Epsilon-greedy action value.
            device (str | torch.device): Device used for the compute.
            seed (int): Random seed value for replication.
            n_frames (int): Number of timesteps for the state input.
            batch_size (int): The zize of the training batch.
            memory_size (int): The size of the replay memory.
            GAMMA (float): Discount factor
            LR (float): Learning rate
            TAU (float): Network weight soft update rate
            n_quantiles (int): Number of quantiles
        """

        # Initialize base ANN parameters
        super().__init__(
            input_size, action_space, layer_size, epsilon, epsilon_min, device, seed
        )

        # iRainbow-specific parameters
        self.n_frames = n_frames
        self.TAU = TAU
        self.n_quantiles = n_quantiles
        self.GAMMA = GAMMA
        self.batch_size = batch_size
        self.n_step = n_step
        self.sync_freq = sync_freq
        self.model_update_freq = model_update_freq

        # Factored action space parameters (already in function signature)

        # IQN-Network
        self.qnetwork_local = IQN(
            input_size,
            action_space,
            layer_size,
            seed,
            n_quantiles,
            n_frames,
            device=device,
            use_factored_actions=use_factored_actions,
            action_dims=action_dims,
            factored_target_variant=factored_target_variant,
        ).to(device)
        self.qnetwork_target = IQN(
            input_size,
            action_space,
            layer_size,
            seed,
            n_quantiles,
            n_frames,
            device=device,
            use_factored_actions=use_factored_actions,
            action_dims=action_dims,
            factored_target_variant=factored_target_variant,
        ).to(device)

        # Aliases for saving to disk
        self.models = {"local": self.qnetwork_local, "target": self.qnetwork_target}
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Memory buffer
        self.memory = Buffer(
            capacity=memory_size,
            obs_shape=(np.array(self.input_size).prod(),),
            n_frames=n_frames,
        )

    def __str__(self):
        return f"iRainbowModel(input_size={np.array(self.input_size).prod() * self.n_frames},action_space={self.action_space})"

    def take_action(self, state: np.ndarray) -> int:
        """Returns actions for given state as per current policy.

        Args:
            state (np.ndarray): current state

        Returns:
            int: The action to take.
        """
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            torch_state = torch.from_numpy(state)
            torch_state = torch_state.float().to(self.device)
            # Add batch dimension if needed
            if torch_state.dim() == 1:
                torch_state = torch_state.unsqueeze(0)

            self.qnetwork_local.eval()
            with torch.no_grad():
                if self.qnetwork_local.use_factored_actions:
                    qvalues_list = self.qnetwork_local.get_qvalues(torch_state, is_eval=True)
                    actions = tuple(torch.argmax(q, dim=-1).cpu().numpy()[0] for q in qvalues_list)
                    # Convert to single index for backward compatibility
                    # Encoding: action = move_action * n_vote + vote_action
                    # For action_dims = [5, 3]: move_action ∈ {0,1,2,3,4} (4=noop), vote_action ∈ {0,1,2}
                    if len(actions) == 2:
                        move_idx, vote_idx = actions
                        single_action = move_idx * self.qnetwork_local.action_dims[1] + vote_idx
                        return int(single_action)
                    # General case for D > 2
                    single_action = actions[0]
                    for d in range(1, len(actions)):
                        single_action = single_action * self.qnetwork_local.action_dims[d] + actions[d]
                    return int(single_action)
                else:
                    action_values = self.qnetwork_local.get_qvalues(torch_state, is_eval=True)
                    action = np.argmax(action_values.cpu().data.numpy(), axis=1)
                    return action[0]
            self.qnetwork_local.train()
        else:
            if self.qnetwork_local.use_factored_actions:
                # Random action for each branch
                actions = tuple(random.choice(range(n_d)) for n_d in self.qnetwork_local.action_dims)
                # Convert to single index
                if len(actions) == 2:
                    move_idx, vote_idx = actions
                    single_action = move_idx * self.qnetwork_local.action_dims[1] + vote_idx
                    return int(single_action)
                # General case for D > 2
                single_action = actions[0]
                for d in range(1, len(actions)):
                    single_action = single_action * self.qnetwork_local.action_dims[d] + actions[d]
                return int(single_action)
            else:
                action = random.choices(np.arange(self.action_space), k=1)
                return action[0]
    
    def get_all_qvalues(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions in the given state.
        
        Args:
            state: The current state as numpy array
            
        Returns:
            numpy array of Q-values for all actions
        """
        torch_state = torch.from_numpy(state)
        torch_state = torch_state.float().to(self.device)
        # Add batch dimension if needed
        if torch_state.dim() == 1:
            torch_state = torch_state.unsqueeze(0)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.get_qvalues(torch_state, is_eval=True)
        self.qnetwork_local.train()
        
        if self.qnetwork_local.use_factored_actions:
            # For factored actions, compute Q-values for all combinations
            # action_values is a list of Q-value tensors, one per branch
            # We need to compute Q(s, a) = Q_move(s, a_move) + Q_vote(s, a_vote) for all combinations
            qvalues_move = action_values[0].cpu().data.numpy()[0]  # (5,)
            qvalues_vote = action_values[1].cpu().data.numpy()[0]  # (3,)
            
            # Compute Q-values for all 15 combinations
            all_qvalues = np.zeros(15)
            for move_idx in range(5):
                for vote_idx in range(3):
                    action_idx = move_idx * 3 + vote_idx
                    all_qvalues[action_idx] = qvalues_move[move_idx] + qvalues_vote[vote_idx]
            
            return all_qvalues
        else:
            return action_values.cpu().data.numpy()[0]  # Return as 1D array

    def train_step(self, custom_gamma: float = None) -> np.ndarray:
        """Update value parameters using given batch of experience tuples.

        .. note:: The training loop CANNOT be named `train()` or `training()` as this conflicts with `nn.Module` superclass functions.

        Args:
            custom_gamma: Optional custom discount factor. If None, uses self.GAMMA.
                For vote model, pass γ^X (macro discount). For move model, pass None.

        Returns:
            float: The loss output.
        """
        loss = torch.tensor(0.0)
        self.optimizer.zero_grad()

        # Use custom_gamma if provided, otherwise use self.GAMMA
        # Note: IQN uses n-step returns, so discount is (gamma)**n_step
        discount_factor = custom_gamma if custom_gamma is not None else self.GAMMA

        # Check if we have enough experiences to sample a batch
        # The sampleable population is reduced by n_frames + 1 due to frame stacking requirements
        sampleable_size = max(1, len(self.memory) - self.n_frames - 1)
        if sampleable_size >= self.batch_size:
            # Sample minibatch
            states, actions, rewards, next_states, dones, valid = self.memory.sample(
                batch_size=self.batch_size
            )

            # Convert to torch tensors and move to device
            states = torch.from_numpy(states).float().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)
            actions = torch.from_numpy(actions).long().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            dones = torch.from_numpy(dones).float().to(self.device)
            valid = torch.from_numpy(valid).float().to(self.device)

            if self.qnetwork_local.use_factored_actions:
                # Factored action space training
                D = self.qnetwork_local.n_action_dims
                action_dims = self.qnetwork_local.action_dims
                
                # Sample quantiles
                taus_cur = torch.rand(self.batch_size, self.n_quantiles, 1).to(self.device)
                
                # Get greedy next actions for each branch
                # Note: forward() returns quantiles, not qvalues. We compute qvalues by taking mean over quantiles.
                quantiles_next_list, _ = self.qnetwork_local.forward(next_states, self.n_quantiles)
                qvalues_next_list = [q.mean(dim=1) for q in quantiles_next_list]  # Mean over quantiles to get Q-values
                a_star_list = [torch.argmax(q, dim=-1) for q in qvalues_next_list]
                
                # Compute target quantiles
                target_quantiles_list, _ = self.qnetwork_target.forward(next_states, self.n_quantiles)
                
                if self.qnetwork_local.factored_target_variant == "A":
                    # Variant A: Shared target
                    target_sum = torch.zeros(self.batch_size, self.n_quantiles, 1).to(self.device)
                    for d in range(D):
                        # Gather quantiles for greedy action
                        a_star_d = a_star_list[d].unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
                        target_q_d = target_quantiles_list[d].gather(
                            2, a_star_d.expand(self.batch_size, self.n_quantiles, 1)
                        )  # (batch, n_quantiles, 1)
                        target_sum += target_q_d
                    y = rewards.unsqueeze(-1) + (
                        discount_factor**self.n_step * (target_sum / D) * (1.0 - dones.unsqueeze(-1))
                    )
                    
                    # Loss for each branch toward shared target
                    loss = 0.0
                    for d in range(D):
                        # Get current action indices for branch d
                        actions_d = self._extract_action_component(actions, d)
                        # Ensure actions_d is 1D [batch_size]
                        if actions_d.dim() > 1:
                            actions_d = actions_d.squeeze()
                        # Get expected quantiles
                        quantiles_expected_list, _ = self.qnetwork_local.forward(states, self.n_quantiles)
                        # Reshape actions_d to [batch_size, 1, 1] then expand to [batch_size, n_quantiles, 1]
                        actions_d_indices = actions_d.unsqueeze(-1).unsqueeze(1).expand(self.batch_size, self.n_quantiles, 1)
                        quantiles_expected_d = quantiles_expected_list[d].gather(2, actions_d_indices)
                        # Quantile regression loss
                        td_error = y - quantiles_expected_d
                        huber_l = calculate_huber_loss(td_error, 1.0) * valid.unsqueeze(-1)
                        quantil_l = abs(taus_cur - (td_error.detach() < 0).float()) * huber_l / 1.0
                        loss += quantil_l.mean()
                    loss = loss / D
                else:
                    # Variant B: Separate targets
                    loss = 0.0
                    for d in range(D):
                        a_star_d = a_star_list[d].unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
                        target_q_d = target_quantiles_list[d].gather(
                            2, a_star_d.expand(self.batch_size, self.n_quantiles, 1)
                        )  # (batch, n_quantiles, 1)
                        y_d = rewards.unsqueeze(-1) + (
                            discount_factor**self.n_step * target_q_d * (1.0 - dones.unsqueeze(-1))
                        )
                        # Get current action and compute loss
                        actions_d = self._extract_action_component(actions, d)
                        # Ensure actions_d is 1D [batch_size]
                        if actions_d.dim() > 1:
                            actions_d = actions_d.squeeze()
                        quantiles_expected_list, _ = self.qnetwork_local.forward(states, self.n_quantiles)
                        # Reshape actions_d to [batch_size, 1, 1] then expand to [batch_size, n_quantiles, 1]
                        actions_d_indices = actions_d.unsqueeze(-1).unsqueeze(1).expand(self.batch_size, self.n_quantiles, 1)
                        quantiles_expected_d = quantiles_expected_list[d].gather(2, actions_d_indices)
                        td_error = y_d - quantiles_expected_d
                        huber_l = calculate_huber_loss(td_error, 1.0) * valid.unsqueeze(-1)
                        quantil_l = abs(taus_cur - (td_error.detach() < 0).float()) * huber_l / 1.0
                        loss += quantil_l.mean()
                    loss = loss / D
            else:
                # Original single-action-space training
                # REPLACED: as suggested by Gemini, use local network to select action and target network to evaluate it
                q_values_next_local, _ = self.qnetwork_local(next_states, self.n_quantiles)
                action_indx = torch.argmax(
                    q_values_next_local.mean(dim=1), dim=1, keepdim=True
                )
                Q_targets_next, _ = self.qnetwork_target(next_states, self.n_quantiles)
                Q_targets_next = Q_targets_next.gather(
                    2,
                    action_indx.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1),
                ).transpose(1, 2)

                # Compute Q targets for current states
                Q_targets = rewards.unsqueeze(-1) + (
                    discount_factor**self.n_step
                    * Q_targets_next
                    * (1.0 - dones.unsqueeze(-1))
                )

                # Get expected Q values from local model
                Q_expected, taus = self.qnetwork_local(states, self.n_quantiles)
                Q_expected: torch.Tensor = Q_expected.gather(
                    2, actions.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1)
                )

                # Quantile Huber loss
                td_error: torch.Tensor = Q_targets - Q_expected
                assert td_error.shape == (
                    self.batch_size,
                    self.n_quantiles,
                    self.n_quantiles,
                ), "wrong td error shape"
                huber_l = calculate_huber_loss(td_error, 1.0)
                # Zero out loss on invalid actions (when you clip past the end of an episode)
                huber_l = huber_l * valid.unsqueeze(-1)

                quantil_l: torch.Tensor = (
                    abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0
                )

                loss = quantil_l.mean()
            
            
            # Minimize the loss
            loss.backward()
            clip_grad_norm_(self.qnetwork_local.parameters(), 1)
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update()

        return loss.detach().cpu().numpy()

    def soft_update(self) -> None:
        """Soft update model parameters.

        `θ_target = τ*θ_local + (1 - τ)*θ_target`
        """
        for target_param, local_param in zip(
            self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
        ):
            target_param.data.copy_(
                self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data
            )

    def start_epoch_action(self, **kwargs) -> None:
        """Model actions before agent takes an action.

        Args:
            **kwargs: All local variables are passed into the model
        """
        # Add empty frames to the replay buffer
        self.memory.add_empty()
        # If it's time to sync, load the local network weights to the target network.
        if kwargs["epoch"] % self.sync_freq == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def end_epoch_action(self, **kwargs) -> None:
        """Model actions computed after each agent takes an action.

        Args:
            **kwargs: All local variables are passed into the model
        """
        # if kwargs["epoch"] > 200 and kwargs["epoch"] % self.model_update_freq == 0:
        #     kwargs["loss"] = self.train_step()
        #     if "game_vars" in kwargs:
        #         kwargs["game_vars"].losses.append(kwargs["loss"])
        #     else:
        #         kwargs["losses"] += kwargs["loss"]
    
    def _extract_action_component(self, actions: torch.Tensor, component_idx: int) -> torch.Tensor:
        """Extract action component from single action index.
        
        Decoding: Given action index a and action_dims = [n_0, n_1, n_2, ...]
        For component d:
            if d == 0: a_0 = a // (n_1 * n_2 * ...)
            elif d == D-1: a_D-1 = a % n_D-1
            else: a_d = (a // (n_d+1 * n_d+2 * ...)) % n_d
        """
        action_dims = self.qnetwork_local.action_dims
        D = len(action_dims)
        
        if component_idx == 0:
            # First component
            divisor = np.prod(action_dims[1:])
            return actions // divisor
        elif component_idx == D - 1:
            # Last component
            return actions % action_dims[-1]
        else:
            # Middle components
            divisor_after = np.prod(action_dims[component_idx+1:])
            component = (actions // divisor_after) % action_dims[component_idx]
            return component


# ------------------------ #
# endregion                #
# ------------------------ #


def calculate_huber_loss(td_errors: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Calculate elementwise Huber loss.

    Args:
        td_errors (torch.Tensor): The temporal difference errors.
        k (float): The kappa parameter.

    Returns:
        torch.Tensor: The Huber loss value.
    """
    loss = torch.where(
        td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k)
    )
    return loss
