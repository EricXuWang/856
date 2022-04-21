import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor_discrete(nn.Module):

    def __init__(self, state_size, action_size, hidden=[128,128]):
        super(Actor_discrete, self).__init__()
        hidden = [state_size] + hidden
        self.feature = nn.ModuleList(nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(hidden[:-1], hidden[1:]))
        self.output = nn.Linear(hidden[-1], action_size)

    def forward(self, state):
        x = state
        for layer in self.feature:
            x = F.relu(layer(x))
        log_probs = F.log_softmax(self.output(x), dim=1) #dim = 1
        return log_probs
    

    
class Critic(nn.Module):

    def __init__(self, state_size, hidden=[256, 256]):
        super(Critic, self).__init__()
        hidden = [state_size] + hidden
        self.feature = nn.ModuleList(nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(hidden[:-1], hidden[1:]))
        self.output = nn.Linear(hidden[-1], 1)
        
    def forward(self, state):
        x = state
        for layer in self.feature:
            x = F.relu(layer(x))
        values = self.output(x)
        return values
    

