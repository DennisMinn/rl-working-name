import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm

import gym
from grid_world import GridWorldEnv
from models import Actor, Critic
from utils import G_t, cum_G_t


def reinforce_baseline_cartpole(num_episodes):
    t, returns = [], []

    env = gym.make('CartPole-v1')

    actor = Actor(env.observation_space.shape[0], env.action_space.n)
    critic = Critic(env.observation_space.shape[0])

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-2)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-2)

    for episode in tqdm(range(num_episodes)):

        state = env.reset()
        trajectory = []
        score = 0
        isTerminal = False

        # Run episode
        while not isTerminal:
            action, log_prob = actor.sample_action(state)
            state_prime, reward, isTerminal, _ = env.step(action)
            score += reward
            trajectory.append([state, action, log_prob, reward])

            state = state_prime

        # Clear Gradient
        critic_optimizer.zero_grad()
        actor_optimizer.zero_grad()

        # Actor
        states, actions, log_probs, rewards = map(list, zip(*trajectory))
        G = cum_G_t(rewards)
        state_vals = critic(torch.tensor(states)).reshape(-1)
        actor_loss = F.mse_loss(state_vals, G)
        actor_loss.backward()

        # Critic
        critic_loss = Critic.reinforce_baseline_loss(G, state_vals, log_probs)
        critic_loss.backward()

        # Update Params
        critic_optimizer.step()
        actor_optimizer.step()

        # Update Metrics
        t.append(len(trajectory))
        returns.append(G_t(trajectory))

    env.close()
    return t, returns


def reinforce_baseline_gridworld():
    returns = []
    t = []

    env = GridWorldEnv()
    critic = Critic(1)
    actor = Actor(1, 4)
    returns = []

    critic_optimizer = optim.AdamW(critic.parameters(), lr=2e-3)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=2e-3)

    for episode in tqdm(range(2000)):

        trajectory = []
        state = env.reset()
        isTerminal = False

        # episode
        while len(trajectory) < 50 and not isTerminal:
            state = torch.tensor([(state-12)/25], dtype=torch.float)
            action, log_prob = actor.sample_action(state)

            state_prime, reward, isTerminal, info = env.step(action)
            trajectory.append((state, action, reward, log_prob))
            state = state_prime

        critic_optimizer.zero_grad()
        actor_optimizer.zero_grad()

        # Clear Gradient
        actor_optimizer.zero_grad()

        # Actor
        states, actions, rewards, log_probs = map(list, zip(*trajectory[-10:]))
        G = cum_G_t(rewards, 0.9)
        state_vals = critic(torch.tensor(states).reshape(-1, 1)).reshape(-1)
        actor_loss = F.mse_loss(state_vals, G)
        actor_loss.backward()

        critic_optimizer.zero_grad()
        # Critic
        critic_loss = Critic.reinforce_baseline_loss(G, state_vals, log_probs)
        critic_loss.backward()

        # Update Params
        critic_optimizer.step()
        actor_optimizer.step()
        t.append(len(trajectory))
        print(len(trajectory), end="\r")

    return returns, t
