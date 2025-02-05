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

# Memory 
class RolloutBuffer:
    def __init__(self):
        """Initialize the memory buffer."""
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.predictions = []
        self.gt = []

    def clear(self):
        """Clear the stored memories."""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.predictions[:]
        del self.gt[:]


# Actor-critic model_mainwork 
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        
        # # actor 
        # self.actor = nn.Sequential(
        #                 nn.Linear(state_dim, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 128),
        #                 nn.Tanh(),
        #                 # nn.Dropout(p=0.2),
        #                 nn.Linear(128, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, action_dim),
        #                 nn.Softmax(dim=-1)
        #             )
        # # critic
        # self.critic = nn.Sequential(
        #                 nn.Linear(state_dim, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 128),
        #                 nn.Tanh(),
        #                 # nn.Dropout(p=0.2),
        #                 nn.Linear(128, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 1)
        #             )
        
        # feature extractor
        self.feature_extractor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 128),
                        nn.Tanh(),
                        # nn.Dropout(p=0.2),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                    )

        self.actor = nn.Sequential(
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)
                    )

        self.critic = nn.Sequential(
                        nn.Linear(64, 1)
                    )

        # predictor
        self.predictor = nn.Sequential(
                        nn.Linear(64+1, 64),
                        nn.Tanh(),
                        nn.Linear(64, state_dim),
                    )
        
        self.double()
        

    def forward(self):
        raise NotImplementedError
    
    def predict(self, hidden_state, action):
        """Predict the next state based on the input state and action."""
        prediction = self.predictor(torch.cat([hidden_state, action], dim=1))
        
        return prediction
    
    def recursive_predict(self, observed_state, action, steps):
        """Predict the future states based on the input state and action,
        recursively for the specified number of steps."""
        state = observed_state
        for step in range(steps):
            hidden_state = self.feature_extractor(state)
            action = self.actor(hidden_state)
            prediction = self.predict(hidden_state, action)
            state = prediction

        return state

    def act(self, hidden_state):
        """Takes an action with the input state."""
        action_probs = self.actor(hidden_state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        """Evaluate the specified action and state."""
        hidden_state = self.feature_extractor(state)
        action_probs = self.actor(hidden_state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(hidden_state)
        
        return action_logprobs, state_values, dist_entropy


# PPO  
class PPO:
    def __init__(self, 
                 device, 
                 state_dim, 
                 action_dim, 
                 lr_actor, 
                 lr_critic, 
                 gamma, 
                 K_epochs, 
                 eps_clip, 
                 is_a2c=False,
                 pred_depth=None,
                 ):
        
        self.device = device 
        self.is_a2c = is_a2c
        self.pred_depth = pred_depth
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        if self.is_a2c:
            self.K_epochs = 1
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

    def take_action(self, state, whether_to_predict=False, steps=1):
        """Choose an action based on the observed state."""
        with torch.no_grad():
            state = state.to(self.device)
            hidden_state = self.model_main.feature_extractor(state)
            action, action_logprob = self.model_main.act(hidden_state)
            # print(action, hidden_state.size())
        with torch.enable_grad():
            if whether_to_predict:
                prediction = self.model_main.recursive_predict(state, action, steps=steps)
                return action, action_logprob, prediction
        return action, action_logprob

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
        # predictions = torch.squeeze(torch.stack(buffer.predictions, dim=0)).detach().to(self.device)
        # gts = torch.squeeze(torch.stack(buffer.gt, dim=0)).detach().to(self.device)
        # TODO: curate gts

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.model_main.evaluate(old_states, old_actions)
            
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

            if self.is_a2c:
                action_probs = torch.exp(self.model_main.evaluate(old_states, old_actions)[0])
                policy_loss = -action_probs * advantages 
            else:
                policy_loss = -torch.min(surr1, surr2) 

            # calculate prediction error
            # prediction_error = self.loss_fn(predictions, gts)
            prediction_error = 0

            loss = policy_loss + 0.5*critic_loss - entropy_coefficient * dist_entropy + prediction_error

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
