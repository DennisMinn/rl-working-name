from tqdm.auto import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from models import Actor, Critic


def actor_critic_cartpole():
    gamma = 0.9
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    count_ca = []
    count_tca = []

    count_episode = range(1, 2000+1)

    for k in range(1):
        env = gym.make('CartPole-v1')
        actor = Actor(state_size, action_size)
        critic = Critic(state_size)

        count_actions = []
        total_count_actions = []
        total_a = 0

        for episode in tqdm(range(1, 2000+1)):
            actor_optim = optim.SGD(actor.parameters(), lr=0.001)
            critic_optim = optim.SGD(critic.parameters(), lr=0.001)
            state = env.reset()
            isTerminal = False
            score = 0
            count_a = 0

            while not isTerminal:
                count_a += 1

                action, log_prob = actor.sample_action(state)
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                v_curr = critic(state_tensor)
                state_prime, reward, isTerminal, info = env.step(action)
                state_prime_tensor = torch.from_numpy(state_prime).float().unsqueeze(0)
                v_next = critic(state_prime_tensor)

                score += reward

                if isTerminal:
                    v_next = torch.tensor([0]).float().unsqueeze(0)

                critic_loss = F.mse_loss(reward + gamma * v_next, v_curr)
                advantage = reward + gamma*v_next.item()-v_curr.item()
                actor_loss = -log_prob * advantage

                actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                actor_optim.step()

                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                if isTerminal:
                    break

                state = state_prime

        count_actions.append(count_a)
        total_a += count_a
        total_count_actions.append(total_a)

    env.close()
    count_ca.append(count_actions)
    count_tca.append(total_count_actions)

    avg_ca = np.array(count_ca)
    avg_ca = np.average(count_ca, axis=0)
    plt.figure()
    plt.title('Count of Episodes vs Count of Actions')
    plt.xlabel('Count of Episodes')
    plt.ylabel('Count of Actions')
    plt.plot(count_episode, avg_ca)
    plt.savefig('count_actions_ac.jpg')
    plt.show()

    avg_tca = np.array(count_tca)
    avg_tca = np.average(count_tca, axis=0)
    plt.figure()
    plt.title('Total Actions vs Count of Episodes ')
    plt.ylabel('Count of Episodes')
    plt.xlabel('Total Count of Actions')
    plt.plot(avg_tca, count_episode)
    plt.savefig('total_actions_ac.jpg')
    plt.show()
