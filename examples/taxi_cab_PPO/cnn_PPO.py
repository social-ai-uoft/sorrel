import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


# Memory class
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class CNN_CLD(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(CNN_CLD, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=in_channels, out_channels=num_filters, kernel_size=1
        )
        self.avg_pool = nn.MaxPool2d(3, 1, padding=0)

    def forward(self, x):
        x = x / 255  # note, a better normalization should be applied
        y1 = F.relu(self.conv_layer1(x))
        y2 = self.avg_pool(y1)  # ave pool is intentional (like a count)
        y2 = torch.flatten(y2, 1)
        y1 = torch.flatten(y1, 1)
        y = torch.cat((y1, y2), 1)
        return y


# class Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.cnn = CNN_CLD(
            in_channels=4, num_filters=5
        )  # make this general once working
        self.rnn = nn.LSTM(
            input_size=state_dim,
            hidden_size=64,  # make this general once working
            num_layers=1,
            batch_first=True,
        )
        self.l1 = nn.Linear(64, 64)  # right here?

        # actor
        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, x, init_rnn_state=None):
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in, init_rnn_state)
        state = F.relu(self.l1(r_out[:, -1, :]))

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), (h_n, h_c)

    def evaluate(self, state, action, init_rnn_state=None):

        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        batch_size, timesteps, C, H, W = state.size()
        c_in = state.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in, init_rnn_state)
        state = F.relu(self.l1(r_out[:, -1, :]))

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


# Class Model
class PPO:
    def __init__(
        self,
        device,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
    ):

        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.max_priority = 1.0

        # self.buffer = RolloutBuffer()

        self.model1 = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model1.actor.parameters(), "lr": lr_actor},
                {"params": self.model1.critic.parameters(), "lr": lr_critic},
            ]
        )

        # self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        # self.policy_old.load_state_dict(self.model1.state_dict())

        self.loss_fn = nn.MSELoss()

    def take_action(self, state, hidden_state=None):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            # action, action_logprob = self.policy_old.act(state)
            action, action_logprob, hidden = self.model1.act(state, hidden_state)

        # buffer.states.append(state)
        # buffer.actions.append(action)
        # buffer.logprobs.append(action_logprob)

        return action, action_logprob, hidden

    def training(self, buffer, entropy_coefficient=0.01):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(buffer.rewards), reversed(buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(self.device)
        )
        # print('state:', old_states.shape)
        old_actions = (
            torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(self.device)
        )
        # print('state:', old_actions.shape)
        old_logprobs = (
            torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(self.device)
        )
        # print('state:', old_logprobs.shape)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.model1.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            critic_loss = self.loss_fn(state_values, rewards)
            # print('c_loss:', critic_loss)
            policy_loss = -torch.min(surr1, surr2)
            # print('p-loss', dist_entropy.shape)
            loss = policy_loss + 0.5 * critic_loss - entropy_coefficient * dist_entropy
            # loss = policy_loss + 0.5*critic_loss

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        # self.policy_old.load_state_dict(self.model1.state_dict())

        # clear buffer
        # self.buffer.clear()

        return loss.mean()

    def save(self, checkpoint_path):
        # torch.save(self.policy_old.state_dict(), checkpoint_path)
        torch.save(self.model1.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.model1.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
