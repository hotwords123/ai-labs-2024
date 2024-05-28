import gym
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
import random

# see https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
# understand environment, state, action and other definitions first before your dive in.

ENV_NAME = 'CartPole-v0'

# Hyper Parameters
# Following params work well if your implement Policy Gradient correctly.
# You can also change these params.
EPISODE = 3000  # total training episodes
STEP = 5000  # step limitation in an episode
EVAL_EVERY = 10  # evaluation interval
TEST_NUM = 5  # number of tests every evaluation
GAMMA = 0.95  # discount factor
LEARNING_RATE = 3e-3  # learning rate for mlp


# A simple mlp implemented by PyTorch #
# it receives (N, D_in) shaped torch arrays, where N: the batch size, D_in: input state dimension
# and outputs the possibility distribution for each action and each sample, shaped (N, D_out)
# e.g.
# state = torch.randn(10, 4)
# outputs = mlp(state)  #  output shape is (10, 2) in CartPole-v0 Game
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class REINFORCE:
    def __init__(self, env):
        # init parameters
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        self.net = MLP(input_dim=self.state_dim, output_dim=self.action_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def predict(self, observation, deterministic=False):
        observation = torch.FloatTensor(observation).unsqueeze(0)
        action_score = self.net(observation)
        probs = F.softmax(action_score, dim=1)
        m = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=1)
        else:
            action = m.sample()
        return action, probs

    def store_transition(self, s, a, p, r):
        self.states.append(s)
        self.actions.append(a)
        self.action_probs.append(p)
        self.rewards.append(r)

    def learn(self):
        # Please make sure all variables used to calculate loss are of type torch.Tensor, or autograd may not work properly.
        # You need to calculate the loss of each step of the episode and store them in '''loss'''.
        # The variables you should use are: self.rewards, self.action_probs, self.actions.
        # self.rewards=[R_1, R_2, ...,R_T], self.actions=[A_0, A_1, ...,A_{T-1}]
        # self.action_probs corresponds to the probability of different actions of each timestep, see predict() for details

        loss = []
        # -------------------------------
        # Your code goes here
        # TODO Calculate the loss of each step of the episode and store them in '''loss'''
         
        # -------------------------------

        # code for autograd and back propagation
        self.optim.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optim.step()

        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        return loss.item()


def main():
    # initialize OpenAI Gym env and PG agent
    master_seed = 0
    np.random.seed(master_seed)
    random.seed(master_seed)
    torch.manual_seed(master_seed)
    
    env = gym.make(ENV_NAME)
    # env = gym.make(ENV_NAME, render_mode="human")
    agent = REINFORCE(env)

    for episode in range(EPISODE):
        # initialize task
        state, _ = env.reset(seed=master_seed+episode*17)
        # Train
        for step in range(STEP):
            action, probs = agent.predict(state)
            next_state, reward, terminated, turcated, _ = env.step(action.item())
            done = terminated or turcated
            agent.store_transition(state, action, probs, reward)
            state = next_state
            if done:
                loss = agent.learn()
                break

        # Test
        if episode % EVAL_EVERY == 0:
            total_reward = 0
            for i in range(TEST_NUM):
                state, _ = env.reset()
                for j in range(STEP):
                    action, _ = agent.predict(state, deterministic=True)
                    state, reward,  terminated, turcated, _ = env.step(action.item())
                    done = terminated or turcated
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward / TEST_NUM

            # Your avg_reward should reach 200 after a number of episodes.
            print('episode: ', episode, 'Evaluation Average Reward:', avg_reward)


if __name__ == '__main__':
    main()
