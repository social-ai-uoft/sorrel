from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from models.memory import Memory
from gem.models.perception import agent_visualfield


import random
import numpy as np
from collections import deque
import pdb

from gem.models.priority_replay import Memory, SumTree


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


class Combine_CLD(nn.Module):
    """
    TODO: need to be able to have an input for non CNN layers to add additional inputs to the model
            likely requires an MLP before the LSTM where the CNN and the additional
            inputs are concatenated
    """

    def __init__(
        self,
        in_channels,
        num_filters,
        in_size,
        hid_size1,
        hid_size2,
        out_size,
        n_layers=2,
        batch_first=True,
    ):
        super(Combine_CLD, self).__init__()
        self.cnn = CNN_CLD(in_channels, num_filters)
        self.rnn = nn.LSTM(
            input_size=in_size,
            hidden_size=hid_size1,
            num_layers=n_layers,
            batch_first=True,
        )
        self.l1 = nn.Linear(hid_size1, hid_size1) # right here?
        self.l2 = nn.Linear(hid_size1, hid_size2)
        self.adv = nn.Linear(hid_size2, out_size)
        self.value = nn.Linear(hid_size2, 1)

        self.dropout = nn.Dropout(0.15)

    def forward(self, x, init_rnn_state):
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in, init_rnn_state)
        y = F.relu(self.l1(r_out[:, -1, :])) 
        y = F.relu(self.l2(y))

        value = self.value(y)
        adv = self.adv(y)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q, (h_n, h_c)


class Model_CNN_LSTM_DQN:

    kind = "cnn_lstm_dqn"  # class variable shared by all instances

    def __init__(self, cfg):
        
        self.modeltype = "cnn_lstm_dqn"

        # Unpack the parameters from cfg
        self.epsilon = cfg.parameters.epsilon
        in_channels = cfg.parameters.in_channels
        num_filters = cfg.parameters.num_filters
        lr = cfg.parameters.lr
        replay_size = cfg.parameters.replay_size
        in_size = cfg.parameters.in_size
        hid_size1 = cfg.parameters.hid_size1
        hid_size2 = cfg.parameters.hid_size2
        out_size = cfg.parameters.out_size
        priority_replay = cfg.parameters.priority_replay
        self.sync_freq = cfg.parameters.sync_freq
        self.inepoch_training_freq = cfg.parameters.inepoch_sync_freq
        self.gamma = cfg.parameters.gamma
        self.batch_size = cfg.parameters.batch_size

        self.model1 = Combine_CLD(
            in_channels, num_filters, in_size, hid_size1, hid_size2, out_size
        )
        self.model2 = Combine_CLD(
            in_channels, num_filters, in_size, hid_size1, hid_size2, out_size
        )
        self.optimizer = torch.optim.Adam(
            self.model1.parameters(), lr=lr, weight_decay=0.01
        )
        
        self.loss_fn = nn.MSELoss()
        self.sm = nn.Softmax(dim=1)
        self.priority_replay = priority_replay
        if priority_replay == True:
            self.max_priority = 1.0
            self.PER_replay = Memory(
                replay_size,
                e=0.01,
                a=0.6,  # set this to 0 for uniform sampling (check these numbers)
                beta=0.4,  # 0.4, set this to 0 for uniform sampling (check these numbers)
                beta_increment_per_sampling=0.0001,  # set this to 0 for uniform sampling (check these numbers)
            )
        if priority_replay == False:
            self.max_priority = 1.0
            self.PER_replay = Memory(
                replay_size,
                e=0.01,
                a=0,  # set this to 0 for uniform sampling (check these numbers)
                beta=0,  # set this to 0 for uniform sampling (check these numbers)
                beta_increment_per_sampling=0,  # set this to 0 for uniform sampling (check these numbers)
            )

        self.device = cfg.device
        self.model1.to(cfg.device)
        self.model2.to(cfg.device)

    def take_action(self, state, init_rnn_state):
        """
        Takes action from the input
        """
        params = self.generate_input(state, init_rnn_state)

        inp, epsilon, init_rnn_state = params
        Q, (c, h) = self.model1(inp, init_rnn_state)

        p = self.sm(Q).cpu().detach().numpy()[0]

        if epsilon > 0.3:
            if random.random() < epsilon:
                action = np.random.randint(0, len(p))
            else:
                action = np.argmax(Q.detach().cpu().numpy())
        else:
            action = np.random.choice(np.arange(len(p)), p=p)
        
        return action, (c, h)

    def training(self):
        """
        DQN batch learning
        """
        loss = torch.tensor(0.0)
        batch_size = self.batch_size
        gamma = self.gamma

        current_replay_size = batch_size + 1

        if current_replay_size > batch_size:

            # note, rewrite to be a min of batch_size or current_replay_size
            # need to figure out how to get current_replay_size
            minibatch, idxs, is_weight = self.PER_replay.sample(batch_size)

            # the do(device) below should not be necessary
            # but on mps, action, reward, and done are being bounced back to the cpu
            # currently removed for a test on CUDA

            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch]).to(self.device)
            action_batch = torch.tensor([a for (s1, a, r, s2, d) in minibatch]).to(self.device)
            reward_batch = torch.tensor([r for (s1, a, r, s2, d) in minibatch]).to(self.device)
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch]).to(self.device)
            done_batch = torch.tensor([d for (s1, a, r, s2, d) in minibatch]).to(self.device)
            
            """
            # TODO: to get the rnn_batch to be fed in the initial state of the LSTM for GPU
                    can not be None, but needs to be set to a tensor. This will not be that
                    much work, but needs to be done. This method with None works reading in the
                    rnn_batch below on just a CPU, but we should add the functionality only once
                    it works on GPU as well
            """

            #rnn_batch = torch.tensor([d for (s1, a, r, s2, d) in minibatch], device=self.device)

            Q1, (c1, h1) = self.model1(state1_batch, None)
            with torch.no_grad():
                Q2, (c1, h1) = self.model2(state2_batch, None)
            
            Y = reward_batch + gamma * (
                (1 - done_batch) * torch.max(Q2.detach(), dim=1)[0]
            )

            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

            errors = torch.abs(Y - X).data.cpu().numpy()

            # there should be better ways of doing the following
            self.max_priority = np.max(errors)

            # update priority
            for i in range(len(errors)):
                idx = idxs[i]
                self.PER_replay.update(idx, errors[i])

            self.optimizer.zero_grad()
            if self.priority_replay == False:
                loss = self.loss_fn(X, Y.detach())
            if self.priority_replay == True:
                replay_stable = 0
                if replay_stable == 1:
                    loss = self.loss_fn(X, Y.detach())
                if replay_stable == 0:
                    loss = (
                        torch.FloatTensor(is_weight).to(self.device)
                        * ((X - Y.detach()) ** 2)
                    ).mean()
            # the step below is where the M1 chip fails
            loss.backward()
            self.optimizer.step()
        return loss

    def updateQ(self):
        """
        Update double DQN model
        """
        self.model2.load_state_dict(self.model1.state_dict())

    def transfer_memories(self, agent, extra_reward=True, seqLength=4):
        """
        Transfer the indiviudual memories to the model
        TODO: We need to have a single version that works for both DQN and
              Actor-criric models (or other types as well)
        """
        exp = agent.episode_memory[-1]
        high_reward = exp[1][2]

        self.PER_replay.add(exp[0], exp[1])
        if extra_reward == True and abs(high_reward) > 9:
            for _ in range(seqLength):
                self.PER_replay.add(exp[0], exp[1])

    def generate_input(self, state, init_rnn_state):
        params = (state.to(self.device), self.epsilon, init_rnn_state)
        return params
    
    def start_epoch_action(self, **kwargs):
        if kwargs['epoch'] % self.sync_freq == 0:
            self.model2.load_state_dict(self.model1.state_dict())

    def end_epoch_action(self, **kwargs):
        # loss gets discarded
        if kwargs['turn'] % self.inepoch_training_freq == 0:
            loss = self.training()