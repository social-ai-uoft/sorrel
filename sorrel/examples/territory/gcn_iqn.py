import random
from typing import Sequence
from sorrel.buffers import Buffer
import numpy as np
import torch
from sorrel.models.pytorch import PyTorchIQN
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from sorrel.models.pytorch.iqn import IQN, calculate_huber_loss
from sorrel.models.pytorch.pytorch_base import DoublePyTorchModel
from sorrel.observation.visual_field import visual_field
from sorrel.observation.observation_spec import ObservationSpec
from sorrel.worlds.gridworld import Gridworld

class GCNiRainbowModel(DoublePyTorchModel):
    def __init__(
        # Base ANN parameters
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        device: str | torch.device,
        seed: int,
        # iRainbow parameters
        n_frames: int,
        n_step: int = 3,
        sync_freq: int = 200,
        model_update_freq: int = 4,
        batch_size: int = 64,
        memory_size: int = 1024,
        LR: float = 0.001,
        TAU: float = 0.001,
        GAMMA: float = 0.99,
        n_quantiles: int = 12,
        hidden_dim: int = 64,
        embed_dim: int = 64
    ):
        # Initialize base ANN parameters
        super().__init__(input_size, action_space, layer_size, epsilon, device, seed)

        # iRainbow-specific parameters
        self.n_frames = n_frames
        self.TAU = TAU
        self.n_quantiles = n_quantiles
        self.GAMMA = GAMMA
        self.batch_size = batch_size
        self.n_step = n_step
        self.sync_freq = sync_freq
        self.model_update_freq = model_update_freq

        self.conv1 = GCNConv(input_size, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)

        # IQN-Network
        self.qnetwork_local = IQN(
            (1, embed_dim),
            action_space,
            layer_size,
            seed,
            n_quantiles,
            n_frames,
            device=device,
        ).to(device)
        self.qnetwork_target = IQN(
            (1, embed_dim),
            action_space,
            layer_size,
            seed,
            n_quantiles,
            n_frames,
            device=device,
        ).to(device)

        self.models = {"local": self.qnetwork_local, "target": self.qnetwork_target}
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Memory buffer
        self.memory = GCNBuffer(
            capacity=memory_size,
            obs_shape=(np.array(self.input_size).prod(),),
            n_frames=n_frames,
            states_shape=(memory_size, 64),
        )

    def __str__(self):
        return f"GCNiRainbowModel(input_size={np.array(self.input_size).prod() * self.n_frames},action_space={self.action_space})"

    def take_action(self, x, edge_index, prev_states, return_state = False) -> int:
        x = torch.from_numpy(x).float().to(self.device)
        edge_index = torch.from_numpy(edge_index).long().to(self.device)

        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))

        state = torch.mean(h, dim=0, keepdim=True)

        if isinstance(prev_states, np.ndarray):
            prev_states = torch.from_numpy(prev_states).float().to(self.device)

        prev_states = prev_states.reshape(1, -1)       # (1, k)
        state = state.reshape(1, -1)                   # (1, D)

        s = torch.cat((prev_states, state), dim=1)     # concat along feature dim

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            torch_state = s
            torch_state = torch_state.float().to(self.device)

            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_qvalues(torch_state)  # .mean(0)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            return action[0] if not return_state else action[0], state.cpu().data.numpy()
        else:
            action = random.choices(np.arange(self.action_space), k=1)
            return action[0] if not return_state else action[0], state.cpu().data.numpy()

    def train_step(self) -> np.ndarray:
        loss = torch.tensor(0.0)
        self.optimizer.zero_grad()

        if len(self.memory) > self.batch_size:

            # Sample minibatch
            states, actions, rewards, next_states, dones, valid = self.memory.sample(
                batch_size=self.batch_size
            )

            # Convert to torch tensors
            states = torch.from_numpy(states)
            next_states = torch.from_numpy(next_states)
            actions = torch.from_numpy(actions)
            rewards = torch.from_numpy(rewards)
            dones = torch.from_numpy(dones)
            valid = torch.from_numpy(valid)

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
                self.GAMMA**self.n_step
                * Q_targets_next.to(self.device)
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

            self.soft_update()

        return loss.detach().numpy()

    def soft_update(self) -> None:
        for target_param, local_param in zip(
            self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
        ):
            target_param.data.copy_(
                self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data
            )

    def start_epoch_action(self, **kwargs) -> None:
        # Add empty frames to the replay buffer
        self.memory.add_empty()
        # If it's time to sync, load the local network weights to the target network.
        if kwargs["epoch"] % self.sync_freq == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def end_epoch_action(self, **kwargs) -> None:
        pass

class GCNObservationSpec(ObservationSpec[tuple[np.ndarray, np.ndarray], Gridworld]):
    def __init__(
        self,
        entity_list: list[str],
        full_view: bool,
        vision_radius: int | None = None,
        env_dims: Sequence[int] | None = None,
        fill_entity_kind: str = "Wall",
        encode_fn=None,
    ):
        super().__init__(
            entity_list, full_view, vision_radius, env_dims, fill_entity_kind
        )

        self.encode_fn = encode_fn  # user can override mapping of entity -> features

        # number of nodes depends on visibility
        if self.full_view:
            width, height = env_dims
            self.num_nodes = width * height
        else:
            size = (2 * self.vision_radius + 1)
            self.num_nodes = size * size

        # Let each node have len(entity_list) features by default
        self.num_features = len(entity_list)
        self.input_size = (self.num_nodes, self.num_features)

    def generate_map(self, entity_list: list[str]) -> dict[str, np.ndarray]:
        entity_map: dict[str, np.ndarray] = {}
        num_classes = len(entity_list)

        for idx, ent in enumerate(entity_list):
            if ent == "EmptyEntity":
                entity_map[ent] = np.zeros(num_classes, dtype=float)
            else:
                vec = np.zeros(num_classes, dtype=float)
                vec[idx] = 1.0
                entity_map[ent] = vec

        return entity_map

    def observe(
        self,
        world: Gridworld,
        location: tuple | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.full_view and location is None:
            raise TypeError(
                "location must be provided when full_view=False"
            )

        grid = visual_field(
            world=world,
            entity_map=self.entity_map,
            vision=self.vision_radius if not self.full_view else None,
            location=location if not self.full_view else None,
            fill_entity_kind=self.fill_entity_kind,
        )
        
        # grid shape: (channels, W, H)
        # Example: (num_features, width, height)

        # Move channels last → shape (W, H, C)
        grid = np.moveaxis(grid, 0, -1)
        W, H, C = grid.shape
        x = grid.reshape(W * H, C)  # shape (num_nodes, num_features)

        edges = []
        def node_id(r, c):
            return r * H + c

        for r in range(W):
            for c in range(H):
                u = node_id(r, c)
                # Right
                if c + 1 < H:
                    v = node_id(r, c + 1)
                    edges.append((u, v))
                    edges.append((v, u))
                # Down
                if r + 1 < W:
                    v = node_id(r + 1, c)
                    edges.append((u, v))
                    edges.append((v, u))

        edge_index = np.array(edges, dtype=np.int64).T  # shape (2, E)

        return x, edge_index
    
class GCNBuffer(Buffer):
    def __init__(self, capacity, obs_shape, states_shape = None, n_frames=1):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.states = np.zeros((capacity, *obs_shape) if states_shape == None else states_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.n_frames = n_frames
