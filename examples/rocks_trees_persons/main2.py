# from tkinter.tix import Tree
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from examples.rocks_trees_persons.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
    plot_time_decay,
)
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from examples.rocks_trees_persons.iRainbow_clean import iRainbowModel
from examples.rocks_trees_persons.env import RPG
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video

import torch.optim as optim
from examples.rocks_trees_persons.elements import EmptyObject, Wall

import time
import numpy as np
import random
import torch

from collections import deque, namedtuple
from scipy.spatial import distance

from datetime import datetime

from sklearn.neighbors import NearestNeighbors


# save_dir = "/Users/yumozi/Projects/gem_data/RPG3_test/"
# save_dir = "/Users/socialai/Dropbox/M1_ultra/"
save_dir = "/Users/ethan/attitudes-output"
# save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")

import time


# ----- episodic weighted average model -----

import math

import warnings

# Ignore future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


def create_episodic_data(num, n_samples=40, leave_out=False):
    object_memory_states_tensor = torch.tensor(
        [obj_mem[0] for obj_mem in object_memory]
    )

    # from_episodic = np.random.randint(10, 31)
    from_episodic = 40
    # from_random = n_samples - from_episodic
    from_random = 0
    looking_for_example = True
    while looking_for_example:
        example = object_memory[np.random.choice(len(object_memory))]
        if abs(example[1]) > 0:
            looking_for_example = False
        else:
            if random.random() > 0.9:
                looking_for_example = False

    curr_target = example

    mems = k_most_similar_recent_states(
        torch.tensor(curr_target[0]),
        state_knn,
        object_memory,
        object_memory_states_tensor,
        decay_rate=5.0,
        k=from_episodic,
    )

    selected_memories = random.sample(object_memory, from_random)
    all_mem = mems + selected_memories
    # Now sorting all_mem based on the time attribute (index 2 in each memory tuple)
    # all_mem.sort(key=lambda x: x[2])
    random.shuffle(all_mem)

    to_subtract = curr_target[0]

    updated_replay_buffer = [
        (
            [
                max(0, min(1, 1 - (abs(a - b) / 255))) ** 2
                for a, b in zip(state, to_subtract)
            ],
            reward,
        )
        for state, reward in all_mem  # Adjusted to unpack the time variable as well
    ]

    memory_tensor = torch.FloatTensor(
        [
            item
            for sublist in updated_replay_buffer
            for item in sublist[0] + [sublist[1]]
        ]
    )

    return memory_tensor.view(n_samples, -1), torch.FloatTensor(
        [curr_target[1]] * n_samples
    ).view(n_samples, -1)


def create_episodic_data_batch(e_types, batch_size, leave_out=False):
    memory_tensors = []
    curr_targets = []
    for num in e_types:
        memory_tensor, curr_target = create_episodic_data(num)

        # Debugging: Check the shape of memory_tensor and type of curr_target[1]
        # print(f"Debug: memory_tensor.shape = {memory_tensor.shape}, curr_target[1] = {curr_target[1]} (data type: {curr_target[1].__class__})")

        memory_tensors.append(memory_tensor)
        curr_targets.append(curr_target[1])

    # Convert to PyTorch tensor and NumPy array and return
    return torch.stack(memory_tensors), np.array(
        curr_targets, dtype=np.float32
    ).reshape(batch_size, 1)


class RewardPredictor(nn.Module):
    def __init__(self):
        super(RewardPredictor, self).__init__()
        self.fc1 = nn.Linear(720, 64 * 2)
        self.fc2 = nn.Linear(64 * 2, 32 * 2)
        self.fc3 = nn.Linear(32 * 2, 1)  # Output is a single scalar value (the reward)

        # Initialize the optimizer within the model
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        # Initialize the loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def learn(self, batch, object_reward):
        # Convert batch and object_reward to PyTorch tensors
        batch_tensor = torch.FloatTensor(batch)  # Shape should be [batch_size, 160]
        object_reward_tensor = torch.FloatTensor(object_reward).view(
            -1, 1
        )  # Shape should be [batch_size, 1]

        # Forward pass: Compute predicted y by passing x to the model
        output = self(batch_tensor)

        # Compute loss
        loss = self.criterion(output, object_reward_tensor)

        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# ----- end episodic weighted average model -----


class RewardPredictor1(nn.Module):
    def __init__(self,
                 d_model,
                 num_tokens,
                 num_heads,
                 dim_feedforward,
                 num_encoder_layers,
                 dropout_p):
        super().__init__()

        self.model_type = "Transformer"
        self.d_model = d_model
        self.positional_encoder = PositionalEncoding(
            dim_model=d_model,
            dropout_p=dropout_p,
            max_len=5000
        )

        self.embedding = nn.Embedding(num_tokens, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        #  fully connected layer
        self.out = nn.Linear(d_model, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Initialize the loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        indices = normalize_and_discretize(x)
        x = self.embedding(indices) * math.sqrt(self.d_model)  # attention
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # not sure what this is for (layer norm?)
        x = self.out(x)

        return x

    def learn(self, batch, object_reward):
        # Convert batch and object_reward to PyTorch tensors
        batch_tensor = torch.FloatTensor(batch)  # Shape should be [batch_size, 160]
        object_reward_tensor = torch.FloatTensor(object_reward).view(
            -1, 1
        )  # Shape should be [batch_size, 1]

        # Forward pass: Compute predicted y by passing x to the model
        output = self(batch_tensor)

        # Compute loss
        loss = self.criterion(output, object_reward_tensor)

        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().item()


def normalize_and_discretize(inputs, num_bins=100):
    min_value = torch.min(inputs)
    max_value = torch.max(inputs)
    normalized_inputs = (inputs - min_value) / (max_value - min_value)
    indices = (normalized_inputs * num_bins).to(torch.long)
    return indices


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

def k_most_similar_recent_states(
    state, knn: NearestNeighbors, memories, object_memory_states_tensor, decay_rate, k=5
):
    if USE_KNN_MODEL:
        # Get the indices of the k most similar states (without selecting them yet)
        state = state.cpu().detach().numpy().reshape(1, -1)
        k_indices = knn.kneighbors(state, n_neighbors=k, return_distance=False)[0]
    else:
        # Perform a brute-force search for the k most similar states
        # distances = [distance.euclidean(state, memory[0]) for memory in memories]
        # k_indices = np.argsort(distances)[:k]

        # let's try another way using torch operations...

        # Calculate the squared Euclidean distance
        squared_diff = torch.sum((object_memory_states_tensor - state) ** 2, dim=1)
        # Take the square root to get the Euclidean distance
        distance = torch.sqrt(squared_diff)
        # Argsort and take top-k
        k_indices = torch.argsort(distance, dim=0)[:k]

    # Gather the k most similar memories based on the indices, preserving the order
    most_similar_memories = [memories[i] for i in k_indices]

    return most_similar_memories


def compute_weighted_average(
    state, memories, similarity_decay_rate=1, time_decay_rate=1
):
    if not memories:
        return 0

    memory_states, rewards = zip(*memories)
    memory_states = np.array(memory_states)
    state = np.array(state)

    # Compute Euclidean distances
    distances = np.linalg.norm(memory_states - state, axis=1)
    max_distance = np.max(distances) if distances.size else 1

    # Compute similarity weights with exponential decay
    similarity_weights = (
        np.exp(-distances / max_distance * similarity_decay_rate)
        if max_distance != 0
        else np.ones_like(distances)
    )

    # Compute time weights with exponential decay
    N = len(memories)
    time_weights = np.exp(-np.arange(N) / (N - 1) * time_decay_rate)

    # Combine the weights
    weights = similarity_weights * time_weights

    # Compute the weighted sum
    weighted_sum = np.dot(weights, rewards)
    total_weight = np.sum(weights)

    return weighted_sum / total_weight if total_weight != 0 else 0


SEED = time.time()  # Seed for replicating training runs
# np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# If True, use the KNN model when computing k-most similar recent states. Otherwise, use a brute-force search.
USE_KNN_MODEL = True
# Run profiling on the RL agent to see how long it takes per step
RUN_PROFILING = False

print(f"Using device: {device}")
print(f"Using KNN model: {USE_KNN_MODEL}")
print(f"Running profiling: {RUN_PROFILING}")


import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class ResourceModel(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim=64,
        memory_size=5000,
        learning_rate=0.001,
    ):
        super(ResourceModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 3)  # Three outputs for three classes

        self.replay_buffer = deque(maxlen=memory_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        probabilities = torch.softmax(self.fc4(x), dim=-1)
        return probabilities

    def sample(self, num_memories):
        return random.sample(
            self.replay_buffer, min(num_memories, len(self.replay_buffer))
        )

    def learn(self, memories, batch_size=32, class_weights=False):
        if class_weights:
            # Calculate class weights
            all_outcomes = [outcome for _, outcome in self.replay_buffer]
            num_samples = len(all_outcomes)
            class_counts = [sum([out[i] for out in all_outcomes]) for i in range(3)]

            # Adding a small epsilon to prevent division by zero
            epsilon = 1e-10
            class_weights = torch.tensor(
                [(num_samples / (count + epsilon)) for count in class_counts]
            ).to(torch.float32)

            for _ in range(len(memories) // batch_size):
                batch = random.sample(memories, batch_size)
                states, targets = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)

                self.optimizer.zero_grad()
                probabilities = self.forward(states)

                # Weighted Cross-Entropy Loss
                loss = F.cross_entropy(
                    probabilities, torch.argmax(targets, dim=1), weight=class_weights
                )

                loss.backward()
                self.optimizer.step()

        else:
            for _ in range(len(memories) // batch_size):
                batch = random.sample(memories, batch_size)
                states, targets = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)

                self.optimizer.zero_grad()
                probabilities = self.forward(states)

                # Cross-Entropy Loss without weights
                loss = F.cross_entropy(probabilities, torch.argmax(targets, dim=1))

                loss.backward()
                self.optimizer.step()

        return loss.item()

    def add_memory(self, state, outcome):
        self.replay_buffer.append((state, outcome))


class ValueModel(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim=64,
        memory_size=5000,
        learning_rate=0.001,
        num_tau=32,
    ):
        super(ValueModel, self).__init__()
        self.num_tau = num_tau
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_tau)

        self.replay_buffer = deque(maxlen=memory_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        taus = torch.linspace(0, 1, steps=self.num_tau, device=x.device).view(1, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        quantiles = self.fc4(x)

        # Extract the 25th, 50th, and 75th percentiles
        percentiles = quantiles[
            :,
            [
                int(self.num_tau * 0.1) - 1,
                int(self.num_tau * 0.5) - 1,
                int(self.num_tau * 0.9) - 1,
            ],
        ]
        return percentiles, taus

    def sample(self, num_memories):
        return random.sample(
            self.replay_buffer, min(num_memories, len(self.replay_buffer))
        )

    def learn(self, memories, batch_size=32):
        for _ in range(len(memories) // batch_size):
            batch = random.sample(memories, batch_size)
            states, rewards = zip(*batch)
            states = torch.tensor(states, dtype=torch.float32)
            rewards = (
                torch.tensor(rewards, dtype=torch.float32)
                .view(-1, 1)
                .repeat(1, self.num_tau)
            )

            self.optimizer.zero_grad()
            # Forward pass to get all quantiles, not just the 25th, 50th, and 75th percentiles
            x = torch.relu(self.fc1(states))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            quantiles = self.fc4(x)  # Shape [batch_size, num_tau]

            errors = rewards - quantiles
            huber_loss = torch.where(
                errors.abs() < 1, 0.5 * errors**2, errors.abs() - 0.5
            )
            taus = (
                torch.linspace(0, 1, steps=self.num_tau, device=states.device)
                .view(1, -1)
                .repeat(batch_size, 1)
            )
            quantile_loss = (taus - (errors < 0).float()).abs() * huber_loss
            loss = quantile_loss.mean()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def add_memory(self, state, reward):
        self.replay_buffer.append((state, reward))


value_model = ValueModel(state_dim=17, memory_size=250)


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        iRainbowModel(
            in_channels=5,
            num_filters=5,
            cnn_out_size=567,  # 910
            state_size=torch.tensor(
                [5, 9, 9]
            ),  # this seems to only be reading the first value
            action_size=4,
            layer_size=250,  # 100
            n_step=3,  # Multistep IQN (rainbow paper uses 3)
            BATCH_SIZE=64,
            BUFFER_SIZE=1024,
            LR=0.00025,  # 0.00025
            TAU=1e-3,  # Soft update parameter
            GAMMA=0.95,  # Discout factor 0.99
            N=12,  # Number of quantiles
            device=device,
            seed=SEED,
        )
    )

    return models


world_size = 25
value_losses = []
trainable_models = [0]
sync_freq = 200  # https://openreview.net/pdf?id=3UK39iaaVpE
modelUpdate_freq = 4  # https://openreview.net/pdf?id=3UK39iaaVpE
epsilon = 0.99

turn = 1


def eval_attiude_model(value_model=value_model):
    atts = []
    s = torch.zeros(7)
    r = value_model(s)
    atts.append(round(r.item(), 2))
    for a in range(7):
        s = torch.zeros(7)
        s[a] = 255.0
        r = value_model(s)
        atts.append(round(r.item(), 2))
    return atts


object_exp2 = deque(maxlen=2500)
object_memory = deque(maxlen=250)
state_knn = NearestNeighbors(n_neighbors=15)
state_knn_CMS = NearestNeighbors(n_neighbors=15)


models = create_models()
env = RPG(
    height=world_size,
    width=world_size,
    layers=1,
    gem1p=0.03,
    gem2p=0.03,
    wolf1p=0.03,  # rename gem3p
    # group_probs=[0.5, 0.5],
    # num_people=40,
    tile_size=(1, 1),
    defaultObject=EmptyObject(20),
)
# env.game_test()


def run_game(
    models,
    env,
    turn,
    epsilon,
    epochs=10000,
    max_turns=100,
    epsilon_decay=0.9999,
    attitude_condition="implicit_attitude",
    switch_epoch=1000,
    episodic_decay_rate=1.0,
    similarity_decay_rate=1.0,
    memory_counter=0,
):
    """
    This is the main loop of the game
    """
    losses = 0
    local_value_losses = 0
    game_points = [0, 0]
    gems = [0, 0, 0, 0]
    decay_rate = 0.2  # Adjust as needed
    change = True
    gem_changes = 0

    for epoch in range(epochs):
        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """
        if epoch % switch_epoch == 0:
            gem_changes = gem_changes + 1
            env.change_gem_values()
        epsilon = epsilon * epsilon_decay
        done, withinturn = 0, 0

        # create a new gameboard for each epoch and repopulate
        # the resset does allow for different params, but when the world size changes, odd
        env.reset_env(
            height=world_size,
            width=world_size,
            layers=1,
            gem1p=0.03,
            gem2p=0.03,
            gem3p=0.03,
            change=change,
        )

        working_memory = 1

        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            env.world[loc].init_replay(working_memory)
            env.world[loc].init_rnn_state = None

        start_resource = np.random.choice([0, 1])
        if start_resource == 0:
            env.world[loc].wood = 1
            env.world[loc].stome = 0
        if start_resource == 1:
            env.world[loc].wood = 0
            env.world[loc].stome = 1

        agent_wood = env.world[loc].wood
        agent_stone = env.world[loc].stone

        turn = 0

        start_time = time.time()

        while done == 0:
            """
            Find the agents and move them
            """
            turn = turn + 1
            withinturn = withinturn + 1

            # --------------------------------------------------------------
            # note the sync models lines may need to be deleted
            # the IQN has a soft update, so we should test dropping
            # the lines below

            if epoch % sync_freq == 0:
                # update the double DQN model ever sync_frew
                for mods in trainable_models:
                    models[mods].qnetwork_target.load_state_dict(
                        models[mods].qnetwork_local.state_dict()
                    )
            # --------------------------------------------------------------

            agentList = find_instance(env.world, "neural_network")

            random.shuffle(agentList)

            for loc in agentList:
                """
                Reset the rewards for the trial to be zero for all agents
                """
                env.world[loc].reward = 0
            agent_num = 0

            for loc in agentList:
                # reset the variables to be safe

                for i in range(world_size):
                    for j in range(world_size):
                        env.world[i, j, 0].appearance[-5] = (
                            env.world[loc].wood * 255.0
                        )  # note, currently overwriting the individual info
                        env.world[i, j, 0].appearance[-4] = env.world[loc].stone * 255.0
                        env.world[i, j, 0].appearance[-3] = 0.0
                        env.world[i, j, 0].appearance[-2] = 0.0
                        env.world[i, j, 0].appearance[-1] = 0.0

                # these are the different attitude models that are used.
                # these can be put into a function to just call the one that is
                # currently being used: this is set in the input to the run
                # game function with a string

                # for speed, the model computes the attiutdes at the beginning
                # of each game rather than each trial. For some games where
                # within round learning is important, this could be changed

                if "tree_rocks" in attitude_condition:
                    if epoch > 2:
                        for i in range(world_size):
                            for j in range(world_size):
                                object_state = torch.tensor(
                                    env.world[i, j, 0].appearance[:-3]
                                ).float()
                                predict = resource_model(object_state)
                                predict = predict.detach().numpy()

                                env.world[i, j, 0].appearance[-3] = predict[0] * 255
                                env.world[i, j, 0].appearance[-2] = predict[1] * 255
                                env.world[i, j, 0].appearance[-1] = predict[2] * 255
                                # if epoch % 100 == 0:
                                #    print(
                                #        "tree rocks",
                                #        epoch,
                                #        i,
                                #        j,
                                #        env.world[i, j, 0].appearance[-3:],
                                #        env.world[i, j, 0].kind,
                                #    )

                # --------------------------------------------------------------
                # this model creates a neural network to learn the reward values
                # --------------------------------------------------------------

                if "implicit_attitude" in attitude_condition:
                    if epoch > 2:
                        for i in range(world_size):
                            for j in range(world_size):
                                object_state = torch.tensor(
                                    env.world[i, j, 0].appearance[:-3]
                                ).float()
                                # object_state = torch.concat(
                                #    object_state, agent_wood, agent_stone
                                # )
                                rs, _ = value_model(object_state.unsqueeze(0))
                                r = rs[0][1]
                                env.world[i, j, 0].appearance[-2] = r.item() * 255
                    testing = False
                    if testing and epoch % 100 == 0:
                        atts = eval_attiude_model()
                        print(epoch, atts)

                # --------------------------------------------------------------
                # this is the no attitude condition, simple IQN learning
                # --------------------------------------------------------------

                if (
                    attitude_condition == "no_attitude"
                ):  # this sets a control condition where no attitudes are used
                    for i in range(world_size):
                        for j in range(world_size):
                            env.world[i, j, 0].appearance[-2] = 0.0

                # --------------------------------------------------------------
                # this is our episodic memory model with search and weighting
                # --------------------------------------------------------------

                if (
                    "EWA" in attitude_condition and epoch > 20
                ):  # this sets a control condition where no attitudes are used
                    object_memory_states_tensor = torch.tensor(
                        [obj_mem[0] for obj_mem in object_memory]
                    )
                    full_view = False
                    if full_view:
                        for i in range(world_size):
                            for j in range(world_size):
                                o_state = env.world[i, j, 0].appearance[:-3]
                                mems = k_most_similar_recent_states(
                                    torch.tensor(o_state),
                                    state_knn,
                                    object_memory,
                                    object_memory_states_tensor,
                                    decay_rate=1.0,
                                    k=100,
                                )
                                env.world[i, j, 0].appearance[-1] = (
                                    compute_weighted_average(
                                        o_state,
                                        mems,
                                        similarity_decay_rate=similarity_decay_rate,
                                        time_decay_rate=episodic_decay_rate,
                                    )
                                    * 255
                                )
                    else:
                        for i in range(9):
                            for j in range(9):
                                if (
                                    (loc[0] + i - 4) >= 0
                                    and (loc[1] + j - 4) >= 0
                                    and (loc[0] + i + 4) < world_size
                                    and (loc[1] + j + 4) < world_size
                                ):
                                    o_state = env.world[
                                        loc[0] + i - 4, loc[1] + j - 4, 0
                                    ].appearance[:-3]
                                    mems = k_most_similar_recent_states(
                                        torch.tensor(o_state),
                                        state_knn,
                                        object_memory,
                                        object_memory_states_tensor,
                                        decay_rate=1.0,
                                        k=10,
                                    )
                                    env.world[
                                        loc[0] + i - 4, loc[1] + j - 4, 0
                                    ].appearance[-1] = (
                                        compute_weighted_average(
                                            o_state,
                                            mems,
                                            similarity_decay_rate=similarity_decay_rate,
                                            time_decay_rate=episodic_decay_rate,
                                        )
                                        * 255
                                    )

                # --------------------------------------------------------------
                # this is complementary learning system model
                # --------------------------------------------------------------

                if (
                    "CMS" in attitude_condition and epoch > 20
                ):  # this sets a control condition where no attitudes are used
                    object_memory_states_tensor = torch.tensor(
                        [obj_mem[0] for obj_mem in object_memory]
                    )
                    for i in range(world_size):
                        for j in range(world_size):
                            o_state = env.world[i, j, 0].appearance[:-3]
                            mems = k_most_similar_recent_states(
                                torch.tensor(o_state),
                                state_knn_CMS,  # HERE IS THE ERROR!
                                object_exp2,
                                object_memory_states_tensor,
                                decay_rate=1.0,
                                k=100,
                            )
                            env.world[i, j, 0].appearance[-1] = (
                                compute_weighted_average(
                                    o_state,
                                    mems,
                                    similarity_decay_rate=similarity_decay_rate,
                                    time_decay_rate=episodic_decay_rate,
                                )
                                * 255
                            )

                agent_num = agent_num + 1
                if env.world[loc].kind != "deadAgent":
                    holdObject = env.world[loc]
                    device = models[holdObject.policy].device

                    # state is 1,1,11,9,9 [need to get wood and stone into state space]
                    state = env.pov(loc)
                    num_wood = torch.tensor(holdObject.wood).float()
                    num_stone = torch.tensor(holdObject.stone).float()

                    batch, timesteps, channels, height, width = state.shape

                    action = models[env.world[loc].policy].take_action(state, epsilon)

                    (
                        env.world,
                        reward,
                        next_state,
                        done,
                        new_loc,
                        object_info,
                        resource_outcome,
                    ) = holdObject.transition(env, models, action[0], loc)

                    # --------------------------------------------------------------
                    # create object memory
                    # this sets up the direct reward experience and state information
                    # to be saved in a replay and also learned from

                    state_object = object_info[0:-3]
                    state_object_input = torch.tensor(state_object).float()

                    rs, _ = value_model(state_object_input.unsqueeze(0))
                    if epoch < 100:
                        mem = (state_object, reward)
                        object_exp2.append(mem)
                        # memory_counter += 1
                    else:
                        if reward > torch.max(rs) or reward < torch.min(rs):
                            mem = (state_object, reward)
                            object_exp2.append(mem)
                            # memory_counter += 1

                    object_exp = (state_object, reward)
                    value_model.add_memory(state_object, reward)
                    memory_counter += 1

                    # learn resource of target
                    if reward != 0:
                        resource_model.add_memory(state_object, resource_outcome)
                    else:
                        if random.random() > 0.5:  # seems to work if downsample nothing
                            resource_model.add_memory(state_object, resource_outcome)

                    if len(resource_model.replay_buffer) > 33 and turn % 2 == 0:
                        resource_loss = resource_model.learn(
                            resource_model.sample(32), batch_size=32
                        )
                    # if epoch > 100 and epoch % 40 and turn % 2:
                    #    if resource_outcome != [1, 0, 0]:
                    #        predict = resource_model(torch.tensor(state_object).float())
                    #        print(predict, resource_outcome)

                    # note, to save time we can toggle the line below to only learn the
                    # implicit attitude when the implicit attitude is being used.

                    if len(value_model.replay_buffer) > 51 and turn % 2 == 0:
                        memories = value_model.sample(50)
                        value_loss = value_model.learn(memories, 25)
                    object_memory.append(object_exp)
                    if USE_KNN_MODEL:
                        # Fit a k-NN model to states extracted from the replay buffer
                        state_knn.fit([exp[0] for exp in object_memory])
                        state_knn_CMS.fit([exp[0] for exp in object_exp2])

                    # --------------------------------------------------------------
                    reward_values = env.gem_values
                    reward_values = sorted(reward_values, reverse=True)

                    if reward == 10:
                        gems[0] = gems[0] + 1
                    if reward == -2:
                        gems[1] = gems[1] + 1
                    if reward == 0:
                        gems[2] = gems[2] + 1
                    if reward == -1:
                        gems[3] = gems[3] + 1

                    # note, the line for PER is commented out. We may want to use IQN
                    # with PER as a better comparison

                    done_flag = 0
                    if (
                        withinturn > max_turns
                        or len(find_instance(env.world, "neural_network")) == 0
                    ):
                        done_flag = 1
                    exp = (
                        # models[env.world[new_loc].policy].max_priority,
                        1,
                        (
                            state,
                            action,
                            reward,
                            next_state,
                            done_flag,
                        ),
                    )

                    env.world[new_loc].episode_memory.append(exp)

                    if env.world[new_loc].kind == "agent":
                        # game_points[0] = game_points[0] + reward / reward_values[0]
                        game_points[0] = game_points[0] + reward

            # determine whether the game is finished (either max length or all agents are dead)
            if (
                withinturn > max_turns
                or len(find_instance(env.world, "neural_network")) == 0
            ):
                done = 1

            if len(trainable_models) > 0:
                """
                Update the next state and rewards for the agents after all have moved
                And then transfer the local memory to the model memory
                """
                # this updates the last memory to be the final state of the game board
                env.world = update_memories(
                    env,
                    find_instance(env.world, "neural_network"),
                    done,
                    end_update=True,
                )

                # transfer the events for each agent into the appropriate model after all have moved
                models = transfer_world_memories(
                    models, env.world, find_instance(env.world, "neural_network")
                )

            if epoch > 10 and withinturn % modelUpdate_freq == 0:
                """
                Train the neural networks within a eposide at rate of modelUpdate_freq
                """
                for mods in trainable_models:
                    experiences = models[mods].memory.sample()
                    loss = models[mods].learn(experiences)
                    losses = losses + loss

        if epoch > 10:
            for mods in trainable_models:
                """
                Train the neural networks at the end of eac epoch
                reduced to 64 so that the new memories ~200 are slowly added with the priority ones
                """
                experiences = models[mods].memory.sample()
                loss = models[mods].learn(experiences)
                losses = losses + loss

        end_time = time.time()
        if RUN_PROFILING:
            print(f"Epoch {epoch} took {end_time - start_time} seconds")

        updateEps = False
        # TODO: the update_epsilon often does strange things. Needs to be reconceptualized
        if updateEps == True:
            # epsilon = update_epsilon(epsilon, turn, epoch)
            epsilon = max(epsilon - 0.00003, 0.2)

        if epoch % 20 == 0 and len(trainable_models) > 0 and epoch != 0:
            # print the state and update the counters. This should be made to be tensorboard instead
            print(
                epoch,
                withinturn,
                round(game_points[0]),
                gems,
                losses,
                epsilon,
                str(gem_changes),
                attitude_condition,
                # env.gem1_value,
                # env.gem2_value,
                # env.gem3_value,
            )
            # rs = show_weighted_averaged(object_memory)
            # print(epoch, rs)
            game_points = [0, 0]
            gems = [0, 0, 0, 0]
            losses = 0
    return models, value_model, resource_model, object_memory, env, turn, epsilon


# Initialize the model

batch_size = 6  # Feel free to change
losses = 0


def update_episodic_reward_model2(model, epochs=10000, batch_size=6):
    for epoch in range(epochs):
        types = np.random.choice([1, 2, 3, 4, 5, 6], batch_size)
        memory_tensor_batch, curr_target_batch = create_episodic_data_batch(
            types, batch_size
        )

        # Concatenate along axis 1 (the second axis)
        # memory_tensor_batch = torch.cat(memory_tensor_batch, dim=1)

        # Reshape to [batch_size, 160]
        memory_tensor_batch = memory_tensor_batch.view(batch_size, -1)

        # Convert curr_targets to a tensor and reshape to [batch_size, 1]
        curr_target_batch = torch.tensor(curr_target_batch, dtype=torch.float32).view(
            batch_size, -1
        )

        loss = model.learn(
            torch.FloatTensor(memory_tensor_batch), torch.FloatTensor(curr_target_batch)
        )

    return loss


def view_episodic_reward_model(model, n=40):
    losses = 0
    estimates = []
    targets = []
    for type in range(1, 7):
        memory_tensor, curr_target = create_episodic_data(type, n, False)
        estimate = model(memory_tensor.view(1, -1))
        estimates.append(round(estimate.item(), 2))
        targets.append(int(curr_target[1]))
    print(
        "Estimates:",
        " ".join(map(str, estimates)),
        "Targets:",
        " ".join(map(str, targets)),
    )


def run_models(
    run_params,
    models=None,
    value_model=None,
    resource_model=None,
    object_memory=None,
    env=None,
    turn=None,
    epsilon=None,
):
    for modRun in range(len(run_params)):
        # I'm not sure if these models are being made here
        models = create_models()
        value_model = ValueModel(state_dim=17, memory_size=250)
        resource_model = ResourceModel(state_dim=17, memory_size=2000)
        object_memory = deque(maxlen=run_params[modRun][6])
        state_knn = NearestNeighbors(n_neighbors=5)
        (
            models,
            value_model,
            resource_model,
            object_memory,
            env,
            turn,
            epsilon,
        ) = run_game(
            models,
            env,
            turn,
            run_params[modRun][0],
            epochs=run_params[modRun][1],
            max_turns=run_params[modRun][2],
            epsilon_decay=run_params[modRun][3],
            attitude_condition=run_params[modRun][4],
            switch_epoch=run_params[modRun][5],
            episodic_decay_rate=run_params[modRun][7],
            similarity_decay_rate=run_params[modRun][8],
        )
    return models, value_model, resource_model, object_memory, env, turn, epsilon


def train_episodic_model(
    model, epochs=1000, epochs_per_batch=25, batch_size=6, related_memories=40
):
    for e in range(epochs):
        loss = update_episodic_reward_model2(
            model, epochs=epochs_per_batch, batch_size=6
        )
        print(e, loss)
        view_episodic_reward_model(model, n=related_memories)
    return model  # is this needed?


#  ----- experiment starts below   ----------

memory_counter = 0
object_memory = deque(maxlen=25000)

# reward predictor model

# episodic_reward_model = RewardPredictor()

episodic_reward_model = RewardPredictor1(d_model=128,
                                         num_tokens=160,
                                         num_heads=4,
                                         dim_feedforward=50,
                                         num_encoder_layers=6,
                                         dropout_p=0.1)

run_params = (
    # [0.3, 1500, 20, 0.999, "implicit_attitude", 12000, 2500, 20.0, 20.0],
    # [0.3, 1500, 20, 0.999, "None", 12000, 2500, 20.0, 20.0],
    # [0.3, 1500, 20, 0.999, "implicit_attitude", 12000, 2500, 20.0, 20.0],
    # [0.5, 4010, 20, 0.999, "implicit_attitude", 12000, 2500, 20.0, 20.0],
    # [0.5, 4010, 20, 0.999, "CMS", 12000, 2500, 20.0, 20.0],
    [0.3, 100, 20, 0.999, "implicit_attitude", 12000, 2500, 20.0, 20.0],
    # [0.3, 1500, 20, 0.999, "implicit_attitude+EWA", 12000, 2500, 20.0, 20.0],
    # [0.3, 1500, 20, 0.999, "tree_rocks", 12000, 2500, 20.0, 20.0],
)


# should write some reset functions to reset the world

for new_people in range(10):
    print("new people", new_people)
    env = RPG(
        height=world_size,
        width=world_size,
        layers=1,
        gem1p=0.03,  # these gem values are not being used
        gem2p=0.03,  # these gem values are not being used
        wolf1p=0.03,  # these gem values are not being used
        tile_size=(1, 1),
        defaultObject=EmptyObject(20),
    )
    object_memory = deque(maxlen=25000)
    resource_model = ResourceModel(state_dim=17, memory_size=2000)
    value_model = ValueModel(state_dim=17, memory_size=250)
    models, value_model, resource_model, object_memory, env, turn, epsilon = run_models(
        run_params, env=env, object_memory=object_memory
    )
    print(" -----------------------------------------")
    print(" using the old model with the new memories")
    print(" -----------------------------------------")
    for examples in range(25):
        view_episodic_reward_model(episodic_reward_model)
    print(" -----------------------------------------")
    print(" learning the new model with the new memories")
    print(" -----------------------------------------")
    episodic_reward_model = train_episodic_model(episodic_reward_model, epochs=75)
    # note, it may only need 75 epochs


# ----- experiment ends above   ----------
