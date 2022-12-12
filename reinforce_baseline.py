import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from tqdm.auto import tqdm

from grid_world import GridWorldEnv
from models import StateValue, Policy
from utils import G_t, select_action


def reinforce_baseline_cartpole():
    returns = []
    t = []

    env = gym.make("CartPole-v1")
    svalue = StateValue(4)
    policy = Policy(4, 2)

    svalue_optimizer = optim.AdamW(svalue.parameters(), lr=2e-3)
    policy_optimizer = optim.AdamW(policy.parameters(), lr=2e-4)

    for episode in tqdm(range(2000)):
        history = []
        state = env.reset()
        isTerminal = False

        # episode
        while not isTerminal:
            action, log_prob = select_action(policy, state=state)

            state_prime, reward, isTerminal, info = env.step(action)

            history.append((state, (action, log_prob), reward))
            state = state_prime

        for i in range(len(history)):
            svalue_optimizer.zero_grad()
            policy_optimizer.zero_grad()

            state, action, _ = history[i]

            G = torch.tensor(G_t(history[i:])).unsqueeze(0)
            svalue_loss = F.mse_loss(svalue(state), G)
            policy_loss = svalue_loss.item() * -action[1]

            policy_loss.backward()
            svalue_loss.backward()

            svalue_optimizer.step()
            policy_optimizer.step()

        returns.append(G_t(history))
        t.append(len(history))

    return returns, t


def reinforce_baseline_gridworld():
    returns = []
    t = []

    env = GridWorldEnv()
    svalue = StateValue(1)
    policy = Policy(1, 4)
    returns = []

    svalue_optimizer = optim.AdamW(svalue.parameters(), lr=2e-5)
    policy_optimizer = optim.AdamW(policy.parameters(), lr=2e-6)

    for episode in tqdm(range(2000)):

        history = []
        state = env.reset()
        isTerminal = False

        # episode
        while not isTerminal:
            with torch.no_grad():
                state = torch.tensor([state/25], dtype=torch.float)
                action = np.random.choice(np.arange(4), p=policy(state).numpy())

            state_prime, reward, isTerminal, info = env.step(action)
            history.append((state, action, reward))
            state = state_prime

        for i in range(len(history)):
            svalue_optimizer.zero_grad()
            policy_optimizer.zero_grad()

            state, action, _ = history[i]

            G = torch.tensor(G_t(history[i:], 0.9), dtype=torch.float).unsqueeze(0)
            svalue_loss = F.mse_loss(svalue(state), G)
            policy_loss = svalue_loss.item() * -policy(state)[action]

            policy_loss.backward()
            svalue_loss.backward()

            svalue_optimizer.step()
            policy_optimizer.step()

        returns.append(G_t(history, 0.9))
        t.append(len(history))

    return returns, t
