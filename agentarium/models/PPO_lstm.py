"""
An implementation of the Proximal Policy Gradient (PPO) algorithm. 

Adapted from the Github Repository: 
Minimal PyTorch Implementation of Proximal Policy Optimization
https://github.com/nikhilbarhate99/PPO-PyTorch
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from agentarium.models.DDQN import ClaasyReplayBuffer as Buffer
import numpy as np 
import torch.nn.functional as F

# Memory 
class RolloutBuffer:
    def __init__(self):
        """Initialize the memory buffer."""
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.h_in = []
        self.h_out = []

    def clear(self):
        """Clear the stored memories."""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.h_in[:]
        del self.h_out[:]


# Actor-critic network 
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # actor 
        self.actor = nn.Sequential(
                        nn.Linear(hidden_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 128),
                        nn.Tanh(),
                        # nn.Dropout(p=0.2),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(hidden_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 128),
                        nn.Tanh(),
                        # nn.Dropout(p=0.2),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

        self.double()
        
        

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, hidden):
        """Takes an action with the input state."""
        lstm_in = F.relu(self.fc1(state))
        lstm_in = lstm_in.view(-1, 1, self.hidden_dim)
        lstm_out, hidden = self.lstm(lstm_in, hidden)
        action_probs = self.actor(lstm_out)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach(), hidden
    
    def evaluate(self, state, action, hidden):
        """Evaluate the specified action and state."""
        lstm_in = F.relu(self.fc1(state))
        lstm_in = lstm_in.view(-1, 1, self.hidden_dim)
        lstm_out, hidden = self.lstm(lstm_in, hidden)
        action_probs = self.actor(lstm_out)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(lstm_out)
        
        return action_logprobs, state_values, dist_entropy
    
    def init_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_dim).double(),
                torch.zeros(1, batch_size, self.hidden_dim).double())


# PPO  
class PPO:
    def __init__(self, device, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.device = device 
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.model_main = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.model_main.actor.parameters(), 'lr': lr_actor},
                        {'params': self.model_main.critic.parameters(), 'lr': lr_critic}
                    ])
        self.loss_fn = nn.MSELoss()
        self.memory = Buffer(
            capacity=1024,
            obs_shape=(state_dim,)
        )
        self.epsilon = 0

    def take_action(self, state, hidden):
        """Choose an action based on the observed state."""
        with torch.no_grad():
            state = state.to(self.device)
            action, action_logprob, hidden = self.model_main.act(state, hidden)

        return action, action_logprob, hidden

    def training(self, buffer, entropy_coefficient=0.01):
        """Train the model with the memories stored in the buffer."""
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float64).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(self.device)
        old_h1_in = torch.cat(list(zip(*(buffer.h_in)))[0], dim=1).detach().to(self.device)
        old_h2_in = torch.cat(list(zip(*(buffer.h_in)))[1], dim=1).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.model_main.evaluate(old_states, old_actions, (old_h1_in, old_h2_in))
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
           
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO 
            critic_loss = self.loss_fn(state_values, rewards)
            policy_loss = -torch.min(surr1, surr2) 
            loss = policy_loss + 0.5*critic_loss - entropy_coefficient * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        return loss.mean() 
    
    def save(self, checkpoint_path):
        """Save the current model."""
        torch.save(self.model_main.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        """Load the specified model."""
        self.model_main.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
