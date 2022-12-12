import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

from grid_world import GridWorldEnv
from models import Actor, Critic
from utils import G_t


def reinforce_baseline_gridworld():
    returns = []
    t = []

    env = GridWorldEnv()
    critic = Critic(1)
    actor = Actor(1, 4)
    returns = []

    critic_optimizer = optim.AdamW(critic.parameters(), lr=2e-5)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=2e-6)

    for episode in tqdm(range(2000)):

        history = []
        state = env.reset()
        isTerminal = False

        # episode
        while not isTerminal:
            state = torch.tensor([state/25], dtype=torch.float)
            action = np.random.choice(np.arange(4), p=actor(state).detach().numpy())

            state_prime, reward, isTerminal, info = env.step(action)
            history.append((state, action, reward))
            state = state_prime

        for i in range(len(history)):
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()

            state, action, _ = history[i]

            G = torch.tensor(G_t(history[i:], 0.9), dtype=torch.float).unsqueeze(0)
            critic_loss = F.mse_loss(critic(state), G)
            actor_loss = critic_loss.item() * -actor(state)[action]

            actor_loss.backward()
            critic_loss.backward()

            critic_optimizer.step()
            actor_optimizer.step()

        returns.append(G_t(history, 0.9))
        t.append(len(history))

    return returns, t
