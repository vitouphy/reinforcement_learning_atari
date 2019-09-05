import gym
import sys
import time
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

ENV = "CartPole-v1"
experiment = "{}_1".format(ENV)
EPISODE = 100000000
learning_rate = 0.001
gamma = 0.95

print_model_every = 100
save_model_every = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class REINFORCE(nn.Module):
    def __init__(self):
        super(REINFORCE, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs):
        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=0)

        return out

    def sample_action(self, obs):
        probs = self.forward(obs)
        m = Categorical(probs)
        action = m.sample().item()
        prob = probs[action]
        return action, prob

    def put_data(self, data):
        self.data.append(data)

    def train(self):
        R = 0
        for reward, log_probs in self.data[::-1]:
            R = reward + R * gamma
            loss = -log_probs * R

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.data = []


def main():

    env = gym.make(ENV)
    model = REINFORCE().to(device)
    avg_reward = 0

    for ep in range(EPISODE):

        obs = env.reset()
        done = False
        net_reward = 0

        while not done:
            action, probab = model.sample_action(torch.Tensor(obs).to(device))
            new_obs, reward, done, info = env.step(action)

            model.put_data((reward, torch.log(probab)))
            net_reward += reward
            # env.render()

        model.train()

        # average decay
        avg_reward = avg_reward*0.99 + net_reward*0.01

        # print the reward
        if (ep + 1) % print_model_every == 0:
            print ("episode: {} | reward: {} | avg_reward: {}".format(ep+1, net_reward, avg_reward))

        # save the model every
        if (ep + 1) % save_model_every == 0:
            save_path = "./checkpoints/{}/{}.pt".format(experiment, (ep+1))
            torch.save(model.state_dict(), save_path)

    env.close()


if __name__ == "__main__":
    main()
