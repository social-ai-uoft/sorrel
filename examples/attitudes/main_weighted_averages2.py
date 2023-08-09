# from tkinter.tix import Tree
from examples.attitudes.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
)
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from examples.attitudes.iRainbow_clean import iRainbowModel
from examples.attitudes.env import RPG
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video

import torch.optim as optim
from examples.attitudes.elements import EmptyObject, Wall

import time
import numpy as np
import random
import torch

from sklearn.neighbors import NearestNeighbors

# save_dir = "/Users/yumozi/Projects/gem_data/RPG3_test/"
save_dir = "/Users/socialai/Dropbox/M1_ultra/"
# save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")

import time


def compute_weighted_average_old(state, memories):
    total_weight = 0
    weighted_sum = 0
    N = len(memories)

    # Calculate max_distance for normalization
    distances = [euclidean(state, memory_state) for memory_state, _ in memories]
    max_distance = max(distances) if distances else 1

    for i, (memory_state, reward) in enumerate(memories):
        # Compute the similarity weight based on Euclidean distance
        distance = euclidean(state, memory_state)
        similarity_weight = 1 - distance / max_distance if max_distance != 0 else 1

        # Compute the time weight (most recent has weight 1, oldest has weight 0)
        time_weight = i / (N - 1)

        # Combine the weights (you can adjust how you combine them if needed)
        weight = similarity_weight * time_weight

        weighted_sum += weight * reward
        total_weight += weight

    return weighted_sum / total_weight if total_weight != 0 else 0


import numpy as np


def k_most_similar_recent_states2(
    state, knn: NearestNeighbors, memories, decay_rate, k=5
):
    if USE_KNN_MODEL:
        # Get the indices of the k most similar states (without selecting them yet)
        state = state.cpu().detach().numpy().reshape(1, -1)
        k_indices = knn.kneighbors(state, n_neighbors=k, return_distance=False)[0]
    else:
        # Perform a brute-force search for the k most similar states
        distances = [distance.euclidean(state, memory[0]) for memory in memories]
        k_indices = np.argsort(distances)[:k]

    # Create a list of weights based on the decay rate, but only for the k most similar states
    # weights = [decay_rate**i for i in k_indices]

    EPSILON = 1e-10  # A small constant
    weights = [decay_rate**i + EPSILON for i in k_indices]

    # Normalize the weights
    weights = [w / sum(weights) for w in weights]

    # Sample from the k most similar memories with the given weights
    sampled_memories = random.choices(
        [memories[i] for i in k_indices], weights=weights, k=k
    )

    return sampled_memories


def k_most_similar_recent_states(
    state, knn: NearestNeighbors, memories, decay_rate, k=5
):
    if USE_KNN_MODEL:
        # Get the indices of the k most similar states (without selecting them yet)
        state = state.cpu().detach().numpy().reshape(1, -1)
        k_indices = knn.kneighbors(state, n_neighbors=k, return_distance=False)[0]
    else:
        # Perform a brute-force search for the k most similar states
        distances = [distance.euclidean(state, memory[0]) for memory in memories]
        k_indices = np.argsort(distances)[:k]

    # Gather the k most similar memories based on the indices, preserving the order
    most_similar_memories = [memories[i] for i in k_indices]

    return most_similar_memories


def compute_weighted_average(state, memories):
    if not memories:
        return 0

    memory_states, rewards = zip(*memories)
    memory_states = np.array(memory_states)
    state = np.array(state)

    # Compute Euclidean distances
    distances = np.linalg.norm(memory_states - state, axis=1)
    max_distance = np.max(distances) if distances.size else 1

    # Compute similarity weights
    similarity_weights = (
        1 - distances / max_distance if max_distance != 0 else np.ones_like(distances)
    )

    # Compute time weights
    N = len(memories)
    time_weights = np.arange(N) / (N - 1)

    # Combine the weights
    weights = similarity_weights * time_weights

    # Compute the weighted sum
    weighted_sum = np.dot(weights, rewards)
    total_weight = np.sum(weights)

    return weighted_sum / total_weight if total_weight != 0 else 0


def show_weighted_averaged(memories):
    rs = []
    for i in range(7):
        state = np.zeros(7)
        state[i] = 255

        mems = k_most_similar_recent_states(
            torch.tensor(state),
            state_knn,
            object_memory,
            decay_rate=1.0,
            k=250,
        )
        r = compute_weighted_average(state, mems)
        rs.append(r)
    print(rs)


SEED = time.time()  # Seed for replicating training runs
# np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

from collections import deque, namedtuple

# noisy_dueling

from scipy.spatial import distance
import numpy as np
import random

from datetime import datetime


# If True, use the KNN model when computing k-most similar recent states. Otherwise, use a brute-force search.
USE_KNN_MODEL = True
# Run profiling on the RL agent to see how long it takes per step
RUN_PROFILING = False

print(f"Using device: {device}")
print(f"Using KNN model: {USE_KNN_MODEL}")
print(f"Running profiling: {RUN_PROFILING}")


def k_most_similar_recent_states(
    state, knn: NearestNeighbors, memories, decay_rate, k=5
):
    if USE_KNN_MODEL:
        # Get the indices of the k most similar states (without selecting them yet)
        state = state.cpu().detach().numpy().reshape(1, -1)
        k_indices = knn.kneighbors(state, n_neighbors=k, return_distance=False)[0]
    else:
        # Perform a brute-force search for the k most similar states
        distances = [distance.euclidean(state, memory[0]) for memory in memories]
        k_indices = np.argsort(distances)[:k]

    # Create a list of weights based on the decay rate, but only for the k most similar states
    # weights = [decay_rate**i for i in k_indices]

    EPSILON = 1e-10  # A small constant
    weights = [decay_rate**i + EPSILON for i in k_indices]

    # Normalize the weights
    weights = [w / sum(weights) for w in weights]

    # Sample from the k most similar memories with the given weights
    sampled_memories = random.choices(
        [memories[i] for i in k_indices], weights=weights, k=k
    )

    return sampled_memories


from sklearn.neighbors import NearestNeighbors


def average_reward(memories):
    # Extract the rewards from the tuples (assuming reward is the second element in each tuple)
    rewards = [memory[1] for memory in memories]

    # Calculate the average reward
    average_reward = sum(rewards) / len(rewards)

    return average_reward


class ValueModel(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, memory_size=5000, learning_rate=0.001):
        super(ValueModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.replay_buffer = deque(maxlen=memory_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample(self, num_memories):
        return random.sample(
            self.replay_buffer, min(num_memories, len(self.replay_buffer))
        )

    def learn(self, memories, batch_size=32):
        for _ in range(len(memories) // batch_size):
            batch = random.sample(memories, batch_size)
            states, rewards = zip(*batch)
            states = torch.tensor(states, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)

            self.optimizer.zero_grad()
            predictions = self.forward(states)
            loss = nn.MSELoss()(predictions, rewards)
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def add_memory(self, state, reward):
        self.replay_buffer.append((state, reward))


value_model = ValueModel(state_dim=7, memory_size=250)


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        iRainbowModel(
            in_channels=8,
            num_filters=8,
            cnn_out_size=567,  # 910
            state_size=torch.tensor(
                [8, 9, 9]
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


object_memory = deque(maxlen=250)
state_knn = NearestNeighbors(n_neighbors=5)

models = create_models()
env = RPG(
    height=world_size,
    width=world_size,
    layers=1,
    defaultObject=EmptyObject(),
    gem1p=0.03,
    gem2p=0.03,
    wolf1p=0.03,  # rename gem3p
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
    episodic_decay_rate=0.2,
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

    for epoch in range(epochs):
        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """
        if epoch % switch_epoch == 0:
            change = not change
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

        if "episodic_attitude" in attitude_condition:
            if len(object_memory) > 50:
                for i in range(world_size):
                    for j in range(world_size):
                        for k in range(working_memory):
                            object_state = torch.tensor(
                                env.world[i, j, 0].appearance[:7]
                            ).float()
                            mems = k_most_similar_recent_states(
                                object_state,
                                state_knn,
                                object_memory,
                                decay_rate=episodic_decay_rate,
                                k=5,
                            )
                            r = average_reward(mems)
                            env.world[i, j, 0].appearance[7] = r * 255

        if attitude_condition == "implicit_attitude":
            if epoch > 2:
                for i in range(world_size):
                    for j in range(world_size):
                        object_state = torch.tensor(
                            env.world[i, j, 0].appearance[:7]
                        ).float()
                        r = value_model(object_state)
                        env.world[i, j, 0].appearance[7] = r.item() * 255
            testing = False
            if testing and epoch % 100 == 0:
                atts = eval_attiude_model()
                print(epoch, atts)

        if (
            attitude_condition == "no_attitude"
        ):  # this sets a control condition where no attitudes are used
            for i in range(world_size):
                for j in range(world_size):
                    env.world[i, j, 0].appearance[7] = 0.0

        if (
            attitude_condition == "weighted_average_attitude" and epoch > 10
        ):  # this sets a control condition where no attitudes are used
            for i in range(world_size):
                for j in range(world_size):
                    o_state = env.world[i, j, 0].appearance[:7]
                    mems = k_most_similar_recent_states(
                        torch.tensor(o_state),
                        state_knn,
                        object_memory,
                        decay_rate=1.0,
                        k=250,
                    )
                    env.world[i, j, 0].appearance[7] = (
                        compute_weighted_average(o_state, mems) * 255
                    )

        turn = 0

        start_time = time.time()

        while done == 0:
            """
            Find the agents and wolves and move them
            """
            turn = turn + 1
            withinturn = withinturn + 1

            if epoch % sync_freq == 0:
                # update the double DQN model ever sync_frew
                for mods in trainable_models:
                    models[mods].qnetwork_target.load_state_dict(
                        models[mods].qnetwork_local.state_dict()
                    )

            agentList = find_instance(env.world, "neural_network")

            random.shuffle(agentList)

            for loc in agentList:
                """
                Reset the rewards for the trial to be zero for all agents
                """
                env.world[loc].reward = 0
            agent_num = 0

            for loc in agentList:
                agent_num = agent_num + 1
                if env.world[loc].kind != "deadAgent":
                    holdObject = env.world[loc]
                    device = models[holdObject.policy].device

                    # rather than just the state, compute the whole map as a test
                    if attitude_condition == "construct_attitude_slow":
                        if len(object_memory) > 50:
                            for i in range(world_size):
                                for j in range(world_size):
                                    for k in range(working_memory):
                                        object_state = torch.tensor(
                                            env.world[i, j, 0].appearance[:7]
                                        ).float()
                                        mems = k_most_similar_recent_states(
                                            object_state,
                                            state_knn,
                                            object_memory,
                                            decay_rate=0.5,
                                            k=5,
                                        )
                                        r = average_reward(mems)
                                        env.world[i, j, 0].appearance[7] = r * 255

                    state = env.pov(loc)
                    batch, timesteps, channels, height, width = state.shape

                    if attitude_condition == "construct_attitude":
                        if len(object_memory) > 50:
                            for t in range(timesteps):
                                for h in range(height):
                                    for w in range(width):
                                        state[0, t, 7, h, w] = 0
                                        # if env.world[h, w, 0].kind != "empty":
                                        object_state = state[0, t, :7, h, w]
                                        mems = k_most_similar_recent_states(
                                            object_state,
                                            state_knn,
                                            object_memory,
                                            decay_rate=0.2,
                                            k=5,
                                        )

                                        r = average_reward(mems)
                                        state[0, t, 7, h, w] = r * 255

                    action = models[env.world[loc].policy].take_action(state, epsilon)

                    (
                        env.world,
                        reward,
                        next_state,
                        done,
                        new_loc,
                        object_info,
                    ) = holdObject.transition(env, models, action[0], loc)

                    # create object memory
                    state_object = object_info[0:7]

                    object_exp = (state_object, reward)
                    value_model.add_memory(state_object, reward)
                    if len(value_model.replay_buffer) > 51 and turn % 2 == 0:
                        memories = value_model.sample(50)
                        value_loss = value_model.learn(memories, 25)
                    object_memory.append(object_exp)
                    if USE_KNN_MODEL:
                        # Fit a k-NN model to states extracted from the replay buffer
                        state_knn.fit([exp[0] for exp in object_memory])

                    if reward == 15:
                        gems[0] = gems[0] + 1
                    if reward == 5:
                        gems[1] = gems[1] + 1
                    if reward == -5:
                        gems[2] = gems[2] + 1
                    if reward == -1:
                        gems[3] = gems[3] + 1

                    # these can be included on one replay

                    if attitude_condition == "construct_attitude":
                        if len(object_memory) > 50:
                            for t in range(timesteps):
                                for h in range(height):
                                    for w in range(width):
                                        next_state[0, t, 7, h, w] = 0
                                        object_state = state[0, t, :7, h, w]
                                        mems = k_most_similar_recent_states(
                                            object_state,
                                            state_knn,
                                            object_memory,
                                            decay_rate=0.5,
                                            k=5,
                                        )
                                        r = average_reward(mems)
                                        next_state[0, t, 7, h, w] = r * 255

                    exp = (
                        # models[env.world[new_loc].policy].max_priority,
                        1,
                        (
                            state,
                            action,
                            reward,
                            next_state,
                            done,
                        ),
                    )

                    env.world[new_loc].episode_memory.append(exp)

                    if env.world[new_loc].kind == "agent":
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
                change,
                attitude_condition,
            )
            # rs = show_weighted_averaged(object_memory)
            # print(epoch, rs)
            game_points = [0, 0]
            gems = [0, 0, 0, 0]
            losses = 0
    return models, env, turn, epsilon


def load_attitudes(
    state, env=env, object_memory=object_memory, state_knn=state_knn, decay_rate=0.5
):
    batch, timesteps, channels, height, width = state.shape
    if len(object_memory) > 50:
        for t in range(timesteps):
            for h in range(height):
                for w in range(width):
                    state[0, t, 7, h, w] = 0
                    object_state = state[0, t, :7, h, w]
                    mems = k_most_similar_recent_states(
                        object_state,
                        state_knn,
                        object_memory,
                        decay_rate,
                        k=5,
                    )
                    r = average_reward(mems)
                    state[0, t, 7, h, w] = r * 0
    return state


def test_memory(env=env, object_memory=object_memory, new_env=False):
    if new_env:
        env.reset_env(
            height=world_size,
            width=world_size,
            layers=1,
            gem1p=0.03,
            gem2p=0.03,
            gem3p=0.03,
        )
    # agentList = find_instance(env.world, "neural_network")
    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        env.world[loc].init_replay(1)
        env.world[loc].init_rnn_state = None

    for loc in find_instance(env.world, "neural_network"):
        state = env.pov(loc)

        # batch, timesteps, channels, height, width = state.shape
        # state_attitudes = load_attitudes(state)
        for i in range(state.shape[3]):
            for j in range(state.shape[4]):
                for t in range(state.shape[1]):
                    i2 = loc[0] - 4 + i
                    j2 = loc[1] - 4 + j
                    flagged = False
                    if i2 < 0:
                        flagged = True
                    if i2 >= world_size:
                        flagged = True
                    if j2 < 0:
                        flagged = True
                    if j2 >= world_size:
                        flagged = True

                    if flagged:
                        object_kind = "outside"
                        object_value = 0
                    else:
                        object_kind = env.world[i2, j2, 0].kind
                        object_value = env.world[i2, j2, 0].value

                    print(loc, t, i, j, object_kind, state[0, t, 7, i, j], object_value)


models = create_models()

# options here are:
#       no_attitude
#       episodic_attitude
#       implicit_attitude
#       construct_attitude
#       construct_attitude_slow (does all mappings at the beginning of the agents turn)

#       construct_attitude_slow works and construct_attitude does not,
#       suggesting that construct_attitude has a bug in it


run_params1 = (
    [0.5, 8100, 20, 0.999, "episodic_attitude_decay", 2000, 250, 1.0],
    [0.5, 8100, 20, 0.999, "episodic_attitude", 2000, 5000, 1.0],
    [0.5, 8100, 20, 0.999, "no_attitude", 2000, 250, 1.0],
    [0.5, 8100, 20, 0.999, "implicit_attitude", 2000, 250, 1.0],
)

run_params2 = (
    [0.5, 8100, 20, 0.999, "episodic_attitude_250_10", 2000, 250, 1.0],
    [0.5, 8100, 20, 0.999, "episodic_attitude_250_8", 2000, 250, 0.8],
    [0.5, 8100, 20, 0.999, "episodic_attitude_250_5", 2000, 250, 0.5],
    [0.5, 8100, 20, 0.999, "episodic_attitude_250_2", 2000, 250, 0.2],
    [0.5, 8100, 20, 0.999, "episodic_attitude_1000_10", 2000, 1000, 1.0],
    [0.5, 8100, 20, 0.999, "episodic_attitude_1000_8", 2000, 1000, 0.8],
    [0.5, 8100, 20, 0.999, "episodic_attitude_1000_5", 2000, 1000, 0.5],
    [0.5, 8100, 20, 0.999, "episodic_attitude_1000_2", 2000, 1000, 0.2],
    [0.5, 8100, 20, 0.999, "episodic_attitude_2500_10", 2000, 2500, 1.0],
    [0.5, 8100, 20, 0.999, "episodic_attitude_2500_8", 2000, 2500, 0.8],
    [0.5, 8100, 20, 0.999, "episodic_attitude_2500_5", 2000, 2500, 0.5],
    [0.5, 8100, 20, 0.999, "episodic_attitude_2500_2", 2000, 2500, 0.2],
)

run_params2b = (
    [0.5, 4000, 20, 0.999, "episodic_attitude_2500_10", 2000, 2500, 1.0],
    [0.5, 4000, 20, 0.999, "episodic_attitude_2500_8", 2000, 2500, 0.8],
    [0.5, 4000, 20, 0.999, "episodic_attitude_2500_5", 2000, 2500, 0.5],
    [0.5, 4000, 20, 0.999, "episodic_attitude_2500_2", 2000, 2500, 0.2],
)

run_params2b = (
    [0.5, 4000, 20, 0.999, "episodic_attitude_2500_10", 2000, 250, 1.0],
    [0.5, 4000, 20, 0.999, "episodic_attitude_2500_8", 2000, 250, 0.8],
    [0.5, 4000, 20, 0.999, "episodic_attitude_2500_5", 2000, 250, 0.5],
    [0.5, 4000, 20, 0.999, "episodic_attitude_2500_2", 2000, 250, 0.2],
)

run_params1a = (
    [0.5, 8100, 20, 0.999, "implicit_attitude", 2000, 250, 1.0],
    [0.5, 8100, 20, 0.999, "episodic_attitude_decay", 2000, 250, 1.0],
    [0.5, 8100, 20, 0.999, "no_attitude", 2000, 250, 1.0],
)

run_params1b = (
    [0.5, 8100, 20, 0.999, "no_attitude", 2000, 250, 1.0],
    [0.5, 8100, 20, 0.999, "implicit_attitude", 2000, 250, 1.0],
    [0.5, 8100, 20, 0.999, "episodic_attitude_decay", 2000, 250, 1.0],
)

run_params1c = (
    [0.5, 8100, 20, 0.999, "episodic_attitude_decay", 2000, 250, 1.0],
    [0.5, 8100, 20, 0.999, "no_attitude", 2000, 250, 1.0],
    [0.5, 8100, 20, 0.999, "implicit_attitude", 2000, 250, 1.0],
)

order = np.random.choice([0, 1, 2])

if order == 0:
    run_params = run_params1a
if order == 1:
    run_params = run_params1b
if order == 2:
    run_params = run_params1c

run_params = run_params2b

run_params = (
    [0.5, 8100, 20, 0.999, "weighted_average_attitude", 2000, 2500, 1.0],
    [0.5, 8100, 20, 0.999, "no_attitude", 2000, 2500, 1.0],
    [0.5, 8100, 20, 0.999, "implicit_attitude", 2000, 2500, 1.0],
    [0.5, 8100, 20, 0.999, "episodic_attitude_decay", 2000, 250, 1.0],
)


# Convert the tuple of lists to a list of lists
# run_params_list = list(run_params)

# Shuffle the list of lists
# random.shuffle(run_params_list)

# If you need the result as a tuple again
# run_params_list = tuple(run_params_list)

# the version below needs to have the keys from above in it
for modRun in range(len(run_params)):
    models = create_models()
    object_memory = deque(maxlen=run_params[modRun][6])
    state_knn = NearestNeighbors(n_neighbors=5)
    models, env, turn, epsilon = run_game(
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
    )
    # atts = eval_attiude_model()
    # print(atts)


# notes:
#      retreived memorories can be put back into a replay buffer
#      need to have long term memories that get stored somehow
#      if we can get the decay to work right, decay can be something that
#      is modulated (and maybe learned) to retain memories for longer
