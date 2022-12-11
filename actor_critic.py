import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
import pdb 

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear3 = nn.Linear(128, self.action_size)

    def forward(self, state):
        output = F.relu((self.linear1(state)))
        output = self.linear3(output)
        distribution = F.softmax(output, dim=-1)
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, state):
        output = F.relu((self.linear1(state)))
        # output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

def select_action(network, state):
    ''' Selects an action given current state
    Args:
    - network (Torch NN): network to process state
    - state (Array): Array of action space in an environment

    Return:
    - (int): action that is selected
    - (float): log probability of selecting that action given state and network
    '''

    #convert state to float tensor, add 1 dimension, allocate tensor on device
    # state = torch.from_numpy(state).float().unsqueeze(0)

    #use network to predict action probabilities
    # print(network(state))
    # pdb.set_trace()
    action_probs = network(state.unsqueeze(0))
    state = state.detach()

    #sample an action using the probability distribution
    m = Categorical(action_probs)
    action = m.sample()

    #return action
    return action, m.log_prob(action)



gamma = .1
num_episodes = 100
num_steps = 10000
env = gym.make("MountainCar-v0")
# env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

actor = Actor(state_size, action_size)
critic = Critic(state_size, action_size)

count_episode = range(1,num_episodes+1)
count_actions = []
total_count_actions = []
total_a = 0
for episode in range(num_episodes):
    actor_optim = optim.Adam(actor.parameters(),lr=0.05)
    critic_optim = optim.Adam(critic.parameters(), lr=0.05)
    state = env.reset()
    isTerminal = False
    score = 0
    
    count_a = 0
    
    for i in range(num_steps):       
        count_a += 1
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action, log_prob = select_action(network=actor, state=state)
            # action = torch.argmax(actor(state))
            # log_prob = torch.max(actor(state))
        state_prime, reward, isTerminal, info = env.step(action.item())
        state_prime = torch.FloatTensor(state_prime)
        if state_prime[0] >= 0.5:
            print(f'Num episodes {episode}, num actions {i} {isTerminal}')
            v_next = torch.tensor([0]).float().unsqueeze(0)
        # if isTerminal:
        #     print(f'Num episodes {episode}, num actions {i} {isTerminal}')
        v_curr = critic(state)
        v_next = critic(state_prime)
            

        td_target = reward + gamma * v_next
        td_error = reward + ((gamma*v_next)-v_curr)
        
        # print(v_curr)
        # print(log_prob)
        # Policy
        actor_loss = (td_error)
        actor_loss *= -log_prob
        actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_optim.step()

        # Value
        critic_loss = F.mse_loss(td_target,v_curr)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()
        state = state_prime

        # print(f'Actor loss is {actor_loss} and critic loss is {critic_loss}')
        if state_prime[0] >= 0.5:
            break
        # print(state)
    print(count_a)
    count_actions.append(count_a)
    total_a += count_a
    total_count_actions.append(total_a)
      
torch.save(actor, 'actor.pkl')
torch.save(critic, 'critic.pkl')
env.close()        

plt.figure()
plt.title('Count of Episodes vs Count of Actions')
plt.xlabel('Count of Episodes')
plt.ylabel('Count of Actions')
plt.plot(count_episode, count_actions)
plt.savefig('count_actions_ac.jpg')
plt.show()

plt.figure()
plt.title('Total Actions vs Count of Episodes ')
plt.ylabel('Count of Episodes')
plt.xlabel('Total Count of Actions')
plt.plot(total_count_actions, count_episode)
plt.savefig('total_actions_ac.jpg')
plt.show()