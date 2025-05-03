import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Classification(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr=0.000):
        super(Classification, self).__init__()
        self.criterion = nn.CrossEntropyLoss()  # Loss function for classification
        self.f1 = nn.Linear(input_size, 2 * hidden_size)
        self.f2 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.f3 = nn.Linear(2 * hidden_size, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # Optimizer as an attribute
        self.lr = lr  # Store learning rate as an attribute
    
    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)  # No activation for logits
        return x 
    
    def train_model(self, task_memory):
        x_batch, y_batch = task_memory['pred'], task_memory['gt']
        # print('shape:', x_batch[0].size(), len(x_batch))
        x_batch = torch.stack(x_batch, dim=0)
        y_batch = torch.stack(y_batch, dim=0)
        self.optimizer.zero_grad()  # Reset gradients
        # logits = self.forward(x_batch)  # Forward pass
        logits = x_batch
        loss = self.criterion(logits, y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update weights
        return loss  # Return loss value for monitoring
    
#TODO: Can nn.Module child class use training()?
    

        