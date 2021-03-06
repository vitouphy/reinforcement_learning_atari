import gym
import sys
import time
import random
import collections
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from atari_wrappers import make_atari, wrap_deepmind


buffer_limit = 50000
NUM_EPISODES = 1000
epsilon = 0.5
learning_rate = 1e-3
batch_size = 64
print_every_ep = 5
GAMMA = 0.99
use_cuda = False

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n, use_cuda=False):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        s_list = torch.tensor(s_lst, dtype=torch.float)
        a_list = torch.tensor(a_lst)
        r_list = torch.tensor(r_lst)
        s_prime_list = torch.tensor(s_prime_lst, dtype=torch.float)
        done_mask_list= torch.tensor(done_mask_lst)

        if use_cuda:
            s_list = s_list.cuda()
            a_list = a_list.cuda()
            r_list = r_list.cuda()
            s_prime_list = s_prime_list.cuda()
            done_mask_list = done_mask_list.cuda()

        return s_list, a_list, r_list, s_prime_list, done_mask_list


    def size(self):
        return len(self.buffer)

class Q_Network(nn.Module):
    def __init__(self):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(128, 256) # states into action pair
        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):
        ''' x is the state of the game '''
        out = F.relu(self.fc1(x))
        return self.fc2(out)

    def sampling_action(self, x, epsilon):
        actions = self.forward(x)
        if random.random() < epsilon:
            action = random.randint(0, 8) # random
        else:
            action = torch.argmax(actions) # choose maximum
        return action

def train(q_policy, q_target, optimizer, memory):

    loss = 0
    s, a, r, s_prime, done = memory.sample(batch_size, use_cuda)

    actions = q_policy(s)
    policy_values = torch.gather(actions, 1, a.view(-1,1))
    target_actions = q_target(s_prime)
    target_values = r + (GAMMA * torch.max(target_actions, 1).values * done)
    loss += ((target_values - policy_values) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def eval():
    env = gym.make('Breakout-v0')
    q_policy = Q_Network()
    q_policy.load_state_dict(torch.load('./checkpoints/pacman_ram_custom/965.pt'))
    observation = env.reset()
    done = False
    while not done:
        action = q_policy.sampling_action(torch.Tensor(observation), 0.2)
        observation, reward, done, info = env.step(action)
        env.render()
    env.close()

def main():
    env = gym.make('Breakout-v0')
    memory = ReplayBuffer()
    q_policy = Q_Network()
    q_target = Q_Network()

    if use_cuda:
        q_policy = q_policy.cuda()
        q_target = q_target.cuda()

    q_target.load_state_dict(q_policy.state_dict())
    q_target.eval()
    optimizer = torch.optim.Adam(q_policy.parameters(), lr=learning_rate)

    score = 0
    start_time = time.time()
    for episode in range(NUM_EPISODES):
        done = False
        observation = env.reset()

        while not done:
            tmp_obs = torch.Tensor(observation)
            if use_cuda:
                tmp_obs = tmp_obs.cuda()
            action = q_policy.sampling_action(tmp_obs, epsilon)
            observation_new, reward, done, info = env.step(action)
            done = 1.0 if done else 0.0
            memory.put( (observation, action, reward, observation_new, done) )
            observation = observation_new

            # train the network
            score += reward
            if done: break

        if memory.size() > 2000:
            train(q_policy, q_target, optimizer, memory)

        # Save the network and so on
        if episode % print_every_ep == 0:
            q_target.load_state_dict(q_policy.state_dict())
            avg_score = score / print_every_ep
            score = 0
            duration = time.time() - start_time
            start_time = time.time()
            print ("epsiode: {}, avg_score: {}, duration: {}".format(episode, avg_score, duration))

            save_path = './checkpoints/pacman_ram_custom/{}.pt'.format(episode)
            torch.save(q_policy.state_dict(), save_path)
            sys.stdout.flush()

    env.close()

if __name__ == "__main__":
    main()
    # eval()
