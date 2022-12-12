import gym
import numpy as np


class GridWorldEnv(gym.Env):
    def __init__(self, render_mode=None):
        n, num_states, num_actions = 5, 25, 4

        # Actions
        actions = [[-1, 0], [0, 1], [1, 0], [0, -1]]

        # Transition Function
        transition_fn = np.zeros((num_states, num_actions, num_states))
        dynamics = [0, 1, -1, np.Inf]
        dynamic_probs = [0.8, 0.05, 0.05, 0.1]
        for s in range(num_states):
            if s == 12 or s == 17:
                continue

            r = s // n
            c = s % n
            for a in range(num_actions):
                for d in range(len(dynamics)):
                    # Action + Dynamics
                    if a + dynamics[d] != np.Inf:
                        action = actions[(a+dynamics[d]) % 4]
                    else:
                        action = [0, 0]

                    # Boundaries
                    if (
                        r+action[0] < 0 or r+action[0] >= n or
                        c+action[1] < 0 or c+action[1] >= n
                    ):
                        action = [0, 0]

                    # Obstacles
                    if (
                        (r+action[0] == 2 and c+action[1] == 2) or
                        (r+action[0] == 3 and c+action[1] == 2)
                    ):
                        action = [0, 0]

                    s_prime = (r + action[0]) * n + (c + action[1])

                    transition_fn[s, a, s_prime] += dynamic_probs[d]

        # Reward Function
        reward_fn = np.zeros(num_states)
        reward_fn[24] = 10
        reward_fn[22] = -10

        # Initial State Distribution
        d0 = np.ones(num_states) / (num_states-3)
        d0[12] = d0[17] = d0[24] = 0

        self.d0 = d0
        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.terminal_state = 24

    def reset(self, seed=None, options=None):
        self.state = np.random.choice(np.arange(25), p=self.d0)
        return self.state

    def step(self, action):
        self.state = np.random.choice(
            np.arange(25),
            p=self.transition_fn[self.state, action]
        )

        reward = self.reward_fn[self.state]
        done = self.state == self.terminal_state
        info = None

        return self.state, reward, done, info
