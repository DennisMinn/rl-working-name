import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.softmax(self.fc2(out), dim=-1)
        return out

    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        m = Categorical(self(state))
        action = m.sample()
        log_prob = m.log_prob(action)

        return action.item(), log_prob

    @staticmethod
    # Doesn't work with auto-grad
    def td_error(G, state_vals):
        return F.mse_loss(G, state_vals)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = self.fc2(out)
        return out

    @staticmethod
    def reinforce_baseline_loss(G, svalues, log_probs):
        deltas = torch.tensor([g - svalue for g, svalue in zip(G, svalues)])
        loss = sum([d * -log_p for d, log_p in zip(deltas, log_probs)])
        return loss
