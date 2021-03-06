import gym
import sys
import time
import random
import collections
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from atari_wrappers import make_atari, wrap_deepmind

ENV = "CartPole-v1"
experiment = "CartPole-v1_minimal"
buffer_limit = 500000
NUM_EPISODES = 100000000
learning_rate = 0.005
batch_size = 32
GAMMA = 0.95
use_cuda = False
action_space = 4
print_every_ep = 20
save_every_ep = 1000
save_every_step = 1000
max_epsilon = 1
min_epsilon = 0.1 # decay from 1 to 0.05 in 300,000 steps

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
        self.fc1 = nn.Linear(4, 32) # states into action pair
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sampling_action(self, x, epsilon):
        if random.random() < epsilon:
            action = random.randint(0, action_space-1) # random
        else:
            actions = self.forward(x)
            print(actions)
            action = torch.argmax(actions).item() # choose maximum
        return action

def train(q_net, optimizer, memory):

    s, a, r, s_prime, done = memory.sample(batch_size, use_cuda)
    actions = q_net(s)
    policy_values = torch.gather(actions, 1, a.view(-1,1))
    target_actions = q_net(s_prime)
    target_values = r + (GAMMA * torch.max(target_actions, 1).values.view(-1,1) * done)
    loss = ((policy_values - target_values) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def eval(weight_file):
    global action_space

    env = gym.make(ENV)
    action_space = env.action_space.n

    q_net = Q_Network()
    q_net.load_state_dict(torch.load(weight_file, map_location='cpu'))
    q_net.eval()

    observation = env.reset()
    done = False
    total_reward = 0
    while not done:
        tmp_obs = torch.Tensor(observation)
        action = q_net.sampling_action(tmp_obs, 0)
        observation_new, reward, done, info = env.step(action)
        observation = observation_new
        total_reward += reward
        # time.sleep(0.1)
        env.render()
    print ("Total Reward: ", total_reward)
    env.close()



def main(weight=None):
    global action_space

    # Initialization
    logs = "./checkpoints/{}/logs".format(experiment)
    summary_writer = tf.summary.FileWriter(logs)

    # Setup environment
    env = gym.make(ENV)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    print ("Max Step: ", env._max_episode_steps)
    print ("Observation Space: ", observation_space)
    print ("Action Space: ", env.action_space.n)

    # Preparation for the network
    memory = ReplayBuffer()
    q_net = Q_Network()

    # Resume the Training
    if weight is not None:
        q_net.load_state_dict(torch.load(weight, map_location='cpu'))

    if use_cuda:
        q_net = q_net.cuda()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)

    score = 0
    loss = 0
    step = 0
    start_time = time.time()

    # Play the episodes
    for episode in range(NUM_EPISODES):
        done = False
        observation = env.reset()

        while not done:
            # Get an action
            tmp_obs = torch.Tensor(observation)
            if use_cuda:
                tmp_obs = tmp_obs.cuda()

            epsilon = max(min_epsilon, max_epsilon - 0.01*(step/10000))
            action = q_net.sampling_action(tmp_obs, epsilon)

            observation_new, reward, done, info = env.step(action)
            done_flag = 0.0 if done else 1.0
            memory.put( (observation, action, reward, observation_new, done_flag) )
            observation = observation_new

            # train the network
            score += reward
            if done: break

            # Train from experience replay
            if memory.size() > 3000:
            # for _ in range (32):
                step += 1
                loss += train(q_net, optimizer, memory)

                if (step % save_every_step == 0):
                    print ("step: {} | avg_loss: {}".format(step, loss / save_every_step))

                    # Save for tensorboard
                    tf_loss = tf.Summary()
                    tag_name = 'loss'
                    tf_loss.value.add(tag='loss', simple_value=loss / save_every_step)
                    summary_writer.add_summary(tf_loss, step)
                    loss = 0


        # Save the network
        if episode % print_every_ep == 0:

            # Compute score and time
            avg_score = score / print_every_ep
            duration = time.time() - start_time
            start_time = time.time()
            print ("step: {} | epsiode: {} | avg_score: {} | duration: {}".format(step, episode, avg_score, duration))

            # Save for tensorboard
            tf_score = tf.Summary()
            tag_name = 'score'
            tf_score.value.add(tag='score', simple_value=avg_score)
            summary_writer.add_summary(tf_score, step)

            score = 0
            sys.stdout.flush()

        if episode % save_every_ep == 0:
            save_path = './checkpoints/{}/{}.pt'.format(experiment, episode)
            torch.save(q_net.state_dict(), save_path)

    env.close()

if __name__ == "__main__":
    # main()
    # main(weight="./checkpoints/breakout/18300.pt")

    # Evaluation
    weight = "./checkpoints/{}/23000.pt".format(experiment)
    print (weight)
    eval(weight)
