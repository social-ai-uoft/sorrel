import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import copy

from collections import deque, namedtuple


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# Define the DQN architecture
class DQN_dynamic(nn.Module):
    def __init__(self):
        super(DQN_dynamic, self).__init__()
        self.fc1 = nn.Linear(3, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 3)
        self.fc4 = nn.Linear(24, 3)

    def forward(self, x, restricted):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        if restricted == True:
            return self.fc3(x)
        else:
            return self.fc3(x), self.fc4(x)


# Define the Double DQN logic
class DoubleDQN_dynamic:
    def __init__(self):
        self.dqn = DQN_dynamic()
        self.target_dqn = DQN_dynamic()
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.optimizer = optim.Adam(self.dqn.parameters())
        self.criterion = nn.MSELoss()

        self.replay_buffer_unrestricted = []
        self.replay_buffer_restricted = []
        self.buffer_capacity = 10000

    def push_to_buffer(self, exp, restricted):  # redo as deque
        state, action, reward, next_state, done = exp
        if restricted == True:
            self.replay_buffer_restricted.append(
                (state, action, reward, next_state, done)
            )
            if len(self.replay_buffer_restricted) > self.buffer_capacity:
                del self.replay_buffer_restricted[0]
        if restricted == False:
            self.replay_buffer_unrestricted.append(
                (state, action, reward, next_state, done)
            )
            if len(self.replay_buffer_unrestricted) > self.buffer_capacity:
                del self.replay_buffer_unrestricted[0]

    def sample_from_buffer(self, batch_size, restricted):
        if restricted == True:
            return random.sample(self.replay_buffer_restricted, batch_size)
        if restricted == False:
            return random.sample(self.replay_buffer_unrestricted, batch_size)

    def take_action(self, state, restricted=True, epsilon=0.5):
        if restricted == True:
            if np.random.rand() < epsilon:
                return np.random.choice(3)
            else:
                with torch.no_grad():
                    q1 = self.dqn(state, restricted)
                    action1 = torch.argmax(q1)
                    return action1.item()
        if restricted == False:
            q1, q2 = self.dqn(torch.FloatTensor(state), restricted)
            action1 = torch.argmax(q1)
            action2 = torch.argmax(q2)
            action1 = action1.item()
            action2 = action2.item()
            if np.random.rand() < epsilon:
                action1 = np.random.choice(3)
            if np.random.rand() < epsilon:
                action2 = np.random.choice(3)
            return action1, action2

    def update_target(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def train(self, restricted=False, batch_size=32, gamma=0.0):  # Set gamma to 0
        if restricted == True:
            replay_buffer = self.replay_buffer_restricted
        if restricted == False:
            replay_buffer = self.replay_buffer_unrestricted

        if len(replay_buffer) < batch_size:
            return

        batch = self.sample_from_buffer(batch_size, restricted)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)  # Stack the tensors
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)  # Stack the tensors
        dones = torch.FloatTensor(dones)

        if restricted == True:
            q_values = (
                self.dqn(states, restricted).gather(1, actions.unsqueeze(1)).squeeze(1)
            )
            target = rewards  # Directly set target to rewards as gamma is 0
            loss = self.criterion(q_values, target)
        if restricted == False:
            q_values1, q_values2 = self.dqn(states, restricted)
            q_values1 = q_values1.gather(1, actions.unsqueeze(1)).squeeze(1)
            q_values2 = q_values2.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss1 = self.criterion(q_values1, rewards)
            loss2 = self.criterion(q_values2, rewards)
            loss = loss1 + loss2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# Define the DQN architecture
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(3, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the Double DQN logic
class DoubleDQN:
    def __init__(self):
        self.dqn = DQN()
        self.target_dqn = DQN()
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.optimizer = optim.Adam(self.dqn.parameters())
        self.criterion = nn.MSELoss()

        self.replay_buffer = []
        self.buffer_capacity = 10000

    def push_to_buffer(self, exp):
        state, action, reward, next_state, done = exp
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.buffer_capacity:
            del self.replay_buffer[0]

    def sample_from_buffer(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)

    def take_action(self, state, restricted=True, epsilon=0.5):
        if np.random.rand() < epsilon:
            return np.random.choice(3)
        else:
            with torch.no_grad():
                return torch.argmax(self.dqn(torch.FloatTensor(state))).item()

    def update_target(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def learnQ(self, batch_size, gamma=0.99):
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.sample_from_buffer(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        print(states.shape)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        states = torch.squeeze(states, dim=1)

        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = (
            self.target_dqn(next_states)
            .gather(1, torch.argmax(self.dqn(next_states), dim=1).unsqueeze(1))
            .squeeze(1)
        )
        target = rewards + gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, restricted=False, batch_size=32, gamma=0.0):  # Set gamma to 0
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.sample_from_buffer(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)  # Stack the tensors
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)  # Stack the tensors
        dones = torch.FloatTensor(dones)

        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        target = rewards  # Directly set target to rewards as gamma is 0

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# MultiAgentEnv Class
class MultiAgentEnv:
    def __init__(self, num_agents=40, num_teams=4):
        self.num_agents = num_agents
        self.num_teams = num_teams
        self.agents = self.initialize_agents()
        self.pairings = []
        self.team_points = self.compute_team_points()
        self.revolution_count = 0
        self.agent_rewards = [0] * self.num_agents
        self.outcome_lookup_table = {
            (0, 0): (2, 2),  # Both cooperate
            (0, 1): (0, 5),  # One cooperates, the other defects
            (1, 0): (5, 0),  # One defects, the other cooperates
            (1, 1): (1, 1),  # Both defect
            (-1, 0): (-1, -1),  # One revolts, the other cooperates
            (0, -1): (-1, -1),  # One cooperates, the other revolts
            (-1, 1): (-1, -1),  # One revolts, the other defects
            (1, -1): (-1, -1),  # One defects, the other revolts
            (-1, -1): (-1, -1),  # Both revolt
        }

    def initialize_agents(self):
        agents = []
        for team in range(self.num_teams):
            team_agents = []
            for _ in range(self.num_agents // self.num_teams):
                points = random.randint(1, 10)
                team_agents.append(
                    {
                        "team": team,
                        "points": points,
                        "available_actions": [1, 1, 1],
                        "turn_points": 0,
                        "replay_buffer": deque(maxlen=1000),
                    }
                )
            agents.extend(team_agents)
        return agents

    def compute_team_points(self):
        team_points = [0] * self.num_teams
        for agent in self.agents:
            team_points[agent["team"]] += agent["points"]
        return team_points

    def reset(self):
        self.agents = self.initialize_agents()
        self.pairings = []
        self.team_points = self.compute_team_points()
        self.revolution_count = 0
        self.agent_rewards = [0] * self.num_agents

    def new_turn(self):
        self.pairings = []
        self.team_points = self.compute_team_points()
        for i in range(len(self.agents)):
            self.agents[i]["available_actions"] = [1, 1, 1]
        self.pair_agents()

    def pair_agents(self):
        indices = list(range(len(self.agents)))
        random.shuffle(indices)
        self.pairings = []

        for i in range(0, len(indices), 2):
            idx1 = indices[i]
            idx2 = indices[i + 1]
            agent1 = self.agents[idx1]
            agent2 = self.agents[idx2]

            # If agents are from the same team, find another pair
            if agent1["team"] == agent2["team"]:
                for j in range(i + 2, len(indices)):
                    idx2 = indices[j]
                    agent2 = self.agents[idx2]
                    if agent1["team"] != agent2["team"]:
                        # Swap the elements to make a valid pair
                        indices[i + 1], indices[j] = indices[j], indices[i + 1]
                        break

            self.pairings.append((idx1, idx2))

    def get_current_pairings(self):
        return self.pairings

    def get_current_teams(self):
        return [(agent1["team"], agent2["team"]) for agent1, agent2 in self.pairings]

    def restrict_actions(self):
        # get the logic of restricting actions in here
        pass

    def check_revolution(self, revolution_prercentage=0.5):
        revolution = 0
        even_share = None
        if self.revolution_count > self.num_agents * revolution_prercentage:
            total_points = sum(agent["points"] for agent in self.agents)
            redistributed_points = math.floor(total_points / 3)
            even_share = redistributed_points // self.num_agents
            # for agent in self.agents: # NEED TO COMPUTE THE POINTS GAINED OR LOST FROM REVOLUTION (THIS IS A FUNCTION OF WHAT HAD)
            #    agent["points"] = even_share
            # self.agent_rewards = [even_share] * self.num_agents
            revolution = 1
        self.revolution_count = 0
        return revolution, even_share


class DuelingNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DuelingNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = 32
        self.action_dim = action_dim

        self.input = nn.Linear(input_dim, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # For restricted
        self.value_stream1 = nn.Linear(self.hidden_dim, 1)
        self.advantage_stream1 = nn.Linear(self.hidden_dim, action_dim)

        # For unrestricted
        self.value_stream2 = nn.Linear(self.hidden_dim, 1)
        self.advantage_stream2 = nn.Linear(self.hidden_dim, action_dim)

    def forward(self, x, restricted):
        x = torch.relu(self.input(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        if restricted:
            value = self.value_stream1(x)
            advantage = self.advantage_stream1(x)
            q_values1 = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            value = self.value_stream1(x)
            advantage = self.advantage_stream1(x)
            q_values1 = value + (advantage - advantage.mean(dim=1, keepdim=True))
            value2 = self.value_stream2(x)
            advantage2 = self.advantage_stream2(x)
            q_values2 = value2 + (advantage2 - advantage2.mean(dim=1, keepdim=True))

        if restricted:
            return q_values1
        else:
            return q_values1, q_values2


class DynamicModel:
    def __init__(self, input_dim, action_dim):
        self.local_model = DuelingNetwork(input_dim, action_dim)
        self.target_model = copy.deepcopy(self.local_model)

        self.optimizer = optim.Adam(self.local_model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.restricted_replay = deque(maxlen=5000)
        self.unrestricted_replay = deque(maxlen=5000)

    def update_target_model(self):
        self.target_model.load_state_dict(self.local_model.state_dict())

    def forward(self, state, restricted, target=False):
        model = self.target_model if target else self.local_model
        return model(state, restricted)

    def take_action(self, state, restricted, epsilon=0.0):
        if restricted:
            q_values = self.forward(state, restricted)
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                action = torch.argmax(q_values).item()
            return action
        else:
            q_values1, q_values2 = self.forward(state, restricted)
            if random.random() < epsilon:
                action1 = random.randint(0, 2)
                action2 = random.randint(0, 2)
            else:
                action1 = torch.argmax(q_values1).item()
                action2 = torch.argmax(q_values2).item()
            return action1, action2

    def train(self, restricted, batch_size=32, gamma=0.0):
        # this is failing, just it also isn't set up correctly yet
        # even if it was working. Next states have not been set up
        # in the learning environment yet, so the gamma to next
        # states is meaningless at the moment
        if restricted:
            replay_buffer = self.restricted_replay
        else:
            replay_buffer = self.unrestricted_replay

        # Sample a mini-batch of size 32
        mini_batch = random.sample(replay_buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        states = torch.stack(states).float()
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        next_states = torch.stack(next_states).float()
        dones = torch.tensor(dones).float()

        # all these if restricted can be cleaned up after working

        if restricted:
            predicted_rewards = self.forward(states, restricted, target=True)
        else:
            predicted_rewards, predicted_rewards2 = self.forward(
                states, restricted, target=True
            )

        target_reward = predicted_rewards[torch.arange(batch_size), 0, actions]

        if restricted == False:
            target_reward2 = predicted_rewards2[torch.arange(batch_size), 0, actions]

        if restricted:
            loss = self.criterion(target_reward, rewards)

        else:
            loss1 = self.criterion(target_reward, rewards)
            loss2 = self.criterion(target_reward2, rewards)
            loss = loss1 + loss2

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# net = DynamicModel(5, 3)
net = DoubleDQN_dynamic()

# Initialize environment and PPO network
env = MultiAgentEnv()
epsilon = 0.8
losses1 = 0
losses2 = 0
forbidden_actions = 0

for epoch in range(1000000000):
    env.reset()
    adv_actions = [0, 0, 0]
    didadv_actions = [0, 0, 0]
    restricted_actions = [0, 0, 0]
    if epoch == 1000:
        epsilon = 0.5
    if epoch > 1000:
        epsilon = epsilon * 0.9999

    for round in range(100):
        done = 0
        env.new_turn()
        env.pair_agents()
        pairings = env.get_current_pairings()
        adv_players = []
        disadv_players = []

        for pair in pairings:
            agent1_idx, agent2_idx = pair
            agent1_team = env.agents[agent1_idx]["team"]
            agent2_team = env.agents[agent2_idx]["team"]

            if env.team_points[agent2_team] == env.team_points[agent1_team]:
                power = np.random.choice([0, 1])
                if power == 0:
                    restricting_agent = agent1_idx
                    restricted_agent = agent2_idx
                if power == 1:
                    restricting_agent = agent2_idx
                    restricted_agent = agent1_idx
            if env.team_points[agent1_team] > env.team_points[agent2_team]:
                restricting_agent = agent1_idx
                restricted_agent = agent2_idx
            if env.team_points[agent2_team] > env.team_points[agent1_team]:
                restricting_agent = agent2_idx
                restricted_agent = agent1_idx

            adv_players.append(restricting_agent)
            disadv_players.append(restricted_agent)

            adv_agent_state = (
                torch.tensor(env.agents[restricting_agent]["available_actions"])
                .float()
                .reshape(
                    3,
                )
            )  # placeholder state

            # extra_elements = torch.tensor([[1, 0]], dtype=torch.float32)

            # Concatenating along dimension 1 (columns)
            # adv_agent_state = torch.cat((adv_agent_state, extra_elements), dim=1)

            # action1, restricted_action = net.take_action(
            #    adv_agent_state, restricted=False, epsilon=epsilon
            # )

            strategy = np.random.choice([0, 1], p=[0.7, 0.3])
            if strategy == 0:
                restricted_action = 1
                action1 = 0
            if strategy == 1:
                restricted_action = 2
                action1 = 1

            env.agents[restricted_agent]["available_actions"][restricted_action] = 0

            disadv_agent_state = (
                torch.tensor(env.agents[restricted_agent]["available_actions"])
                .float()
                .reshape(
                    3,
                )
            )  # placeholder state

            # Tensor you want to append
            # extra_elements = torch.tensor([[0, 1]], dtype=torch.float32)

            # Concatenating along dimension 1 (columns)
            # disadv_agent_state = torch.cat((disadv_agent_state, extra_elements), dim=1)

            action2 = net.take_action(
                disadv_agent_state, restricted=True, epsilon=epsilon
            )

            reward1, reward2 = env.outcome_lookup_table.get((action1, action2), (0, 0))

            if action1 == 2:
                reward1 = reward1 - 10
            if action2 == 2:
                reward2 = reward2 - 10

            if env.agents[restricted_agent]["available_actions"][action2] == 0:
                reward2 = -100
                forbidden_actions += 1

            env.agents[restricting_agent]["points"] += reward1
            env.agents[restricted_agent]["points"] += reward2

            env.agents[restricting_agent]["replay_buffer"].append(
                [adv_agent_state, action1, reward1, adv_agent_state, done]
            )
            env.agents[restricted_agent]["replay_buffer"].append(
                [disadv_agent_state, action2, reward2, disadv_agent_state, done]
            )

            if action1 == 2:
                env.revolution_count += 1
            if (
                action2 == 2
                and env.agents[restricted_agent]["available_actions"][restricted_action]
                != 1
            ):
                env.revolution_count += 1

            adv_actions[action1] += 1
            didadv_actions[action2] += 1
            restricted_actions[restricted_action] += 1

        revolution, redistribution = env.check_revolution(
            revolution_prercentage=1.0
        )  # temporary setting so no revolutions
        if revolution == 1:
            for agent in range(env.num_agents):
                env.agents[agent]["replay_buffer"][-1][2] = redistribution

        for agent in adv_players:
            env.agents[agent]["available_actions"] = [1, 1, 1]
        #    net.unrestricted_replay.append(env.agents[agent]["replay_buffer"][-1])

        for agent in disadv_players:
            net.push_to_buffer(env.agents[agent]["replay_buffer"][-1], restricted=True)
            # net.restricted_replay.append(env.agents[agent]["replay_buffer"][-1])
            env.agents[agent]["available_actions"] = [1, 1, 1]

        env.new_turn()

    if epoch > 20:
        loss1 = net.train(restricted=True)
        losses1 = loss1 + losses1
        # loss2 = net.train(restricted=False)
        # losses2 = loss2 + losses2

    if epoch % 100 == 0:
        print(
            epoch,
            epsilon,
            losses1,
            losses2,
            adv_actions,
            didadv_actions,
            restricted_actions,
            forbidden_actions,
        )
        adv_actions = [0, 0, 0]
        didadv_actions = [0, 0, 0]
        restricted_actions = [0, 0, 0]
        forbidden_actions = 0
        losses1 = 0
        losses2 = 0
    if epoch % 200 == 0:
        net.update_target()
