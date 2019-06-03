import gym
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# constants
GAMMA = 0.99
MAX_MEMORY = 2000
EPISODE = 1000
BATCH = 32
RANDOM_THERS = 0.5  #70% Exploit

'''
Set up OpenAI Gym
'''
D = [] # Data or History of the game
env = gym.make('CartPole-v0')
print (env.action_space)
print (env.observation_space)

class Q_Network(nn.Module):
    def __init__(self):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

Q = Q_Network()
optimizer = optim.Adam(Q.parameters(), lr=1e-3)

def get_action(state):
    state = torch.FloatTensor(state)
    out = Q(state)
    value, idx = torch.max(out, 0)
    print (out)
    # print (value, idx)
    if idx == 0:
        print ("YES")
    else:
        print ("NO")
    print ("==========")
    return value, idx.item()

def sample_training_set():
    '''
    Select sample from training set 
    According to batch size
    '''
    # sample = []
    # for i in range(BATCH):
    #     idx = random.randint(0, len(D)-1)
    #     sample.append(D[idx])
    # return sample
    idx = random.randint(0, len(D)-1)
    return D[idx:idx+32]

'''
Training
'''
for ep in range(EPISODE):
    observation = env.reset()
    total_reward = 0
    for t in range(1000):
        # env.render()
        action = 0
        cur_state = observation
        # cur_state = get_state(observation)

        if random.random() > RANDOM_THERS: # Explore
            action = env.action_space.sample()
        else: # Exploit
            q_value0, action = get_action(cur_state)

        observation, reward, done, info = env.step(action)
        total_reward += reward
        # next_state = get_state(observation)
        next_state = observation
        D.append((cur_state, action, reward, next_state, done))
        if len(D) > 2000:
            del D[0]
            
        if done:
            break

    print ("Episode: {}, Reward: {}".format(ep+1, total_reward))
    # if ep % 5 == 0:
        # random.shuffle(D)

    # Train the network
    samples = sample_training_set()
    loss = 0
    for s0, a0, r, s1, done in samples:
        target = r
        if not done:
            q_value1, a1 = get_action(s1)
            target = r + GAMMA * q_value1
        q_value0, _ = get_action(s0)
        # print (q_value0)
        loss += (target - q_value0) ** 2
    loss /= len(samples)

    # Backpropagate
    Q.zero_grad()
    loss.backward()
    optimizer.step()

env.close()