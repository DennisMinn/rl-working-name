from itertools import repeat, chain
import functools
import torch
import numpy as np
from matplotlib import pyplot as plt


def cum_G_t(rewards, gamma=1):
    G = []
    cum_sum = 0
    for r in reversed(rewards):
        cum_sum = r + cum_sum * gamma
        G.insert(0, cum_sum)

    G = torch.tensor(G, dtype=torch.float)
    G = (G - G.mean())/G.std()
    return G


def G_t(trajectory, gamma=1):
    rewards = [reward for _, _, _, reward in trajectory]
    _return = functools.reduce(
        lambda _return, reward: _return + reward[1] * pow(gamma, reward[0]),
        enumerate(rewards),
        0
    )

    return _return


def plot_t_per_episode(metrics):
    t_mean = np.array(metrics["t"]).mean(axis=0)
    t_std = np.array(metrics["t"]).std(axis=0)
    episodes = np.arange(len(t_mean))

    plt.yticks(ticks=(np.arange(10)*100))
    plt.errorbar(episodes, t_mean, t_std, linestyle=None, marker=None, ecolor="#bfe6ff")


def plot_action_per_episode(metrics):
    t_mean = np.array(metrics["t"]).mean(axis=0, dtype=int)

    t_mean = list(
        chain.from_iterable(
            repeat(episode, t)
            for episode, t in enumerate(t_mean)
        )
    )
    plt.plot(t_mean)


def plot_return_per_episode(metrics):
    t_mean = np.array(metrics["returns"]).mean(axis=0)
    t_std = np.array(metrics["returns"]).std(axis=0)
    episodes = np.arange(len(t_mean))

    plt.errorbar(episodes, t_mean, t_std, linestyle=None, marker=None, ecolor="#bfe6ff")
