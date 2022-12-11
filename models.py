import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, action_dim)

    def forward(self, state):
        state = torch.tensor(state)
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = torch.softmax(self.fc3(out), dim=0)
        return out


class StateValue(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, state):
        state = torch.tensor(state)
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
