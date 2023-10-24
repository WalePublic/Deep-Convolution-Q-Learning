# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 2023

@author: wale
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# OpenAI and Doom
import gym
import experience_replay, image_preprocessing

from torch.autograd import Variable
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from brain import CNN
from softmaxBody import SoftmaxBody




class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
        
    def __call__(self, inputs): 
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()


doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n


# Create AI
cqnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cqnn, body = softmax_body)

# Setting up experience replay
n_steps = experience_replay.NStepProgress(env=doom_env, ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=10000)\
    
    
# Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype= np.float32)))
        output = cqnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max(())
        
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
    
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    
    return torch.from_numpy(np.array(inputs, dtype= np.float32)), torch.stack(targets)

# 100 steps moving average
class MovingAverage:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
        
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
            
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    
    def average(self):
        return np.mean(self.list_of_rewards)
    
ma = MovingAverage(100)


# Training
loss = nn.MSELoss()
optimizer = optim.Adam(cqnn.parameters(), lr = 0.001)
nb_epochs = 100

for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cqnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))




