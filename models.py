import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        state = torch.tensor(state)
        out = F.relu(self.fc1(state))
        out = torch.softmax(self.fc2(out), dim=0)
        return out


class StateValue(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state):
        state = torch.tensor(state)
        out = F.relu(self.fc1(state))
        out = self.fc2(out)
        return out
