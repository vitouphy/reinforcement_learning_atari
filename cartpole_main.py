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
experiment = "CartPole-v1"
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

        # Change order for Conv2D from N x W x H x C -> N x C x W x H
        # s_list = s_list.permute(0, 3, 1, 2)
        # s_prime_list = s_prime_list.permute(0, 3, 1, 2)

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
        # self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(4, 8) # states into action pair
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, action_space)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = x.view(-1, 2592)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sampling_action(self, x, epsilon):
        if random.random() < epsilon:
            action = random.randint(0, action_space-1) # random
        else:
            actions = self.forward(x)
            action = torch.argmax(actions).item() # choose maximum
        return action

def train(q_policy, q_target, optimizer, memory):

    s, a, r, s_prime, done = memory.sample(batch_size, use_cuda)
    actions = q_policy(s)
    policy_values = torch.gather(actions, 1, a.view(-1,1))
    target_actions = q_target(s_prime)
    target_values = r + (GAMMA * torch.max(target_actions, 1).values.view(-1,1) * done)
    # loss = F.smooth_l1_loss(policy_values, target_values)
    loss = ((policy_values - target_values) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def eval(weight_file):
    q_policy = Q_Network()
    q_policy.load_state_dict(torch.load(weight_file, map_location='cpu'))
    q_policy.eval()

    env = make_atari(ENV)
    env = wrap_deepmind(env, frame_stack=True)

    observation = env.reset()
    done = False
    while not done:
        tmp_obs = torch.Tensor(observation).unsqueeze(0).permute(0, 3, 1, 2)
        action = q_policy.sampling_action(tmp_obs, 0.1)
        print(action)
        observation_new, reward, done, info = env.step(action)
        time.sleep(1)
        env.render()

    env.close()



def main(weight=None):
    global action_space

    # Initialization
    logs = "./checkpoints/{}/logs".format(experiment)
    summary_writer = tf.summary.FileWriter(logs)

    # Setup environment
    # env = make_atari(ENV)
    env = gym.make(ENV)
    print ("Max Step: ", env._max_episode_steps)
    # env = wrap_deepmind(env, episode_life=False, frame_stack=True)
    observation_space = env.observation_space.shape[0]
    print ("Observation Space: ", observation_space)
    action_space = env.action_space.n
    print ("Action Space: ", env.action_space.n)

    # Preparation for the network
    memory = ReplayBuffer()
    q_policy = Q_Network()
    q_target = Q_Network()

    # Resume the Training
    if weight is not None:
        q_policy.load_state_dict(torch.load(weight, map_location='cpu'))

    if use_cuda:
        q_policy = q_policy.cuda()
        q_target = q_target.cuda()

    q_target.load_state_dict(q_policy.state_dict())
    q_target.eval()
    optimizer = torch.optim.Adam(q_policy.parameters(), lr=learning_rate)


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
            # tmp_obs = torch.Tensor(observation).unsqueeze(0).permute(0, 3, 1, 2)
            tmp_obs = torch.Tensor(observation)
            if use_cuda:
                tmp_obs = tmp_obs.cuda()

            epsilon = max(min_epsilon, max_epsilon - 0.01*(step/10000))
            action = q_policy.sampling_action(tmp_obs, epsilon)

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
                loss += train(q_policy, q_target, optimizer, memory)

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
            # Save the network
            q_target.load_state_dict(q_policy.state_dict())

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
            torch.save(q_policy.state_dict(), save_path)


        # print (obs[:,:,0] == obs[:,:,1])
        # print (obs[:,:,0])
        # cv2.imshow("Frame-0", obs[:,:,0])
        # cv2.imshow("Frame-1", obs[:,:,1])
        # cv2.imshow("Frame-2", obs[:,:,2])
        # cv2.imshow("Frame-3", obs[:,:,3])
        # cv2.waitKey(0)

    env.close()

if __name__ == "__main__":
    main()
    # main(weight="./checkpoints/breakout/18300.pt")

    # Evaluation
    # weight = "./checkpoints/breakout/9800.pt"
    # eval(weight)
