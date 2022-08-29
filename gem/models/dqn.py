from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.memory import Memory
from models.perception import agentVisualField


class DQN(nn.Module):
    def __init__(self, numFilters, insize, hidsize1, hidsize2, outsize):
        super(DQN, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=3, out_channels=numFilters, kernel_size=1
        )
        self.l2 = nn.Linear(insize, hidsize1)
        self.l3 = nn.Linear(hidsize1, hidsize1)
        self.l4 = nn.Linear(hidsize1, hidsize2)
        self.l5 = nn.Linear(hidsize2, outsize)
        self.avg_pool = nn.MaxPool2d(3, 1, padding=0)
        self.dropout = nn.Dropout(0.1)
        self.conv_bn = nn.BatchNorm2d(5)

    def forward(self, x):
        """
        forward of DQN
        """
        x = x / 255  # note, a better normalization should be applied
        y1 = F.relu(self.conv_layer1(x))
        y2 = self.avg_pool(y1)  # ave pool is intentional (like a count)
        y2 = torch.flatten(y2, 1)
        y1 = torch.flatten(y1, 1)
        y = torch.cat((y1, y2), 1)
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        y = F.relu(self.l4(y))
        value = self.l5(y)
        return value


class modelDQN:

    kind = "double_dqn"  # class variable shared by all instances

    def __init__(self, numFilters, lr, replaySize, insize, hidsize1, hidsize2, outsize):
        self.modeltype = "double_dqn"
        self.model1 = DQN(numFilters, insize, hidsize1, hidsize2, outsize)
        self.model2 = DQN(numFilters, insize, hidsize1, hidsize2, outsize)
        self.optimizer = torch.optim.Adam(
            self.model1.parameters(), lr=lr, weight_decay=0.01
        )
        self.loss_fn = nn.MSELoss()
        self.replay = deque([], maxlen=replaySize)
        self.sm = nn.Softmax(dim=1)

    def createInput(self, world, i, j, holdObject, numImages=-1):
        img = agentVisualField(world, (i, j), holdObject.vision)
        input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        return input

    def takeAction(self, params):
        inp, epsilon = params
        Q = self.model1(inp)
        p = self.sm(Q).detach().numpy()[0]

        if epsilon > 0.3:
            if random.random() < epsilon:
                action = np.random.randint(0, len(p))
            else:
                action = np.argmax(Q.detach().numpy())
        else:
            action = np.random.choice(np.arange(len(p)), p=p)
        return action

    def training(self, batch_size, gamma):

        loss = torch.tensor(0.0)

        if len(self.replay) > batch_size:

            minibatch = random.sample(self.replay, batch_size)
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

            # next test, remove the reshape and see if this still works
            # Q1 = self.model1(state1_batch.reshape(batch_size,3,9,9))
            Q1 = self.model1(state1_batch)
            with torch.no_grad():
                # Q2 = self.model2(state2_batch.reshape(batch_size,3,9,9))
                Q2 = self.model2(state2_batch)

            Y = reward_batch + gamma * (
                (1 - done_batch) * torch.max(Q2.detach(), dim=1)[0]
            )
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

            self.optimizer.zero_grad()
            loss = self.loss_fn(X, Y.detach())
            loss.backward()
            self.optimizer.step()
        return loss

    def updateQ(self):
        self.model2.load_state_dict(self.model1.state_dict())

    def transferMemories(self, world, i, j, extraReward=True):
        # transfer the events from agent memory to model replay
        exp = world[i, j, 0].replay[-1]
        self.replay.append(exp)
        if extraReward == True and abs(exp[2]) > 9:
            for _ in range(5):
                self.replay.append(exp)
