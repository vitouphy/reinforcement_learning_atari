import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make('MsPacman-ram-v0')

def select_action():
    '''
    Total observation are 9
    But only 1-4 are necessary.
    1:up, 2:right, 3:left, 4:down
    '''
    return random.randint(1,4)

# print (env.action_space)
#
# observation, reward, done, info = env.step(0)
# print (observation)
# print (reward)
# print (done)
# print (info)
# env.close()

NUM_EPISODES = 5
for _ in range(NUM_EPISODES):
    observation = env.reset()
    done = False
    while not done:
        action = select_action()
        observation, reward, done, info = env.step(action)
        print (reward)
        # env.render()
    print ("======")
env.close()
