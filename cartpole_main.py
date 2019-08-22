import gym
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# constants
GAMMA = 0.99
MAX_MEMORY = 10000
EPISODE = 100000
BATCH = 128
RANDOM_THERS = 0.7  #70% Exploit

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
        self.fc1 = nn.Linear(4, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

policy_net = Q_Network()
target_net = Q_Network()
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

def get_action(state):
    state = torch.FloatTensor(state)
    out = policy_net(state)
    value, idx = torch.max(out, 0)
    # print (out)
    # print (value, idx)
    # if idx == 0:
    #     print ("YES")
    # else:
    #     print ("NO")
    # print ("==========")
    return value, idx.item()

def get_target_action(state):
    state = torch.FloatTensor(state)
    out = target_net(state)
    value, idx = torch.max(out, 0)
    # print (out)
    # # print (value, idx)
    # if idx == 0:
    #     print ("YES")
    # else:
    #     print ("NO")
    # print ("==========")
    return value, idx.item()

def sample_training_set():
    '''
    Select sample from training set 
    According to batch size
    '''
    return random.sample(D, BATCH)

'''
Training
'''
def train():
    if len(D) < BATCH:
        return

    samples = sample_training_set()
    loss = 0
    for s0, a0, r, s1, done1 in samples:
        target = r
        if not done1:
            q_value1, a1 = get_target_action(s1)
            target = r + GAMMA * q_value1
        q_value0, _ = get_action(s0)
        tmp_loss =  q_value0 - target
        loss += tmp_loss ** 2
        loss /= len(samples)

    # Backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for ep in range(EPISODE):
    observation = env.reset()
    total_reward = 0
    for t in range(1000):
        action = 0
        cur_state = observation
        if random.random() > RANDOM_THERS: # Explore
            action = random.randint(0, 1)
        else: # Exploit
            _, action = get_action(cur_state)

        observation, reward, done, info = env.step(action)
        if done:
            reward = -1

        total_reward += reward
        next_state = observation
        D.append((cur_state, action, reward, next_state, done))
        if len(D) > MAX_MEMORY:
            del D[0]

        train()

        if done:
            break

    print ("Episode: {}, Reward: {}".format(ep+1, total_reward))
    
    if ep % 10 == 0:
        # for name, param in policy_net.named_parameters():
        #     if param.requires_grad:
        #         print (name, param.data)
        target_net.load_state_dict(policy_net.state_dict())

env.close()