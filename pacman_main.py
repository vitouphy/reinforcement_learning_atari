import gym
import sys
import time
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


buffer_limit = 50000
NUM_EPISODES = 1000
epsilon = 0.5
learning_rate = 1e-3
batch_size = 128
print_every_ep = 5

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

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

    s_list, a_list, r_list, s_prime_list, done_list = memory.sample(batch_size)
    loss = 0
    for i in range(batch_size):
        s = s_list[i]
        a = a_list[i]
        r = r_list[i]
        s_prime = s_prime_list[i]
        done = done_list[i]

        policy_value = q_policy(s)[a]
        target_value = r
        if not done:
            target_actions = q_target(s_prime)
            target_value += torch.max(target_actions)
        loss += (target_value - policy_value) ** 2

    loss /= batch_size

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    env = gym.make('MsPacman-ram-v0')
    memory = ReplayBuffer()
    q_policy = Q_Network()
    q_target = Q_Network()
    q_target.load_state_dict(q_policy.state_dict())
    optimizer = torch.optim.Adam(q_policy.parameters(), lr=learning_rate)

    score = 0
    start_time = time.time()
    for episode in range(NUM_EPISODES):
        done = False
        observation = env.reset()

        while not done:
            action = q_policy.sampling_action(torch.Tensor(observation), epsilon)
            observation_new, reward, done, info = env.step(action)
            done = 1 if done else 0
            memory.put( (observation, action, reward, observation_new, done) )
            observation = observation_new

            # train the network
            score += reward
            if done: break

            if memory.size() > 30000:
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
