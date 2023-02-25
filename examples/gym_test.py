import os
import sys

import gym
import evogym.envs
from evogym import sample_robot
import numpy as np
from evogym import get_full_connectivity
import torch
import torch.nn as nn
import torch.nn.functional as F

#### part 0
pos = []
parent_path = os.path.dirname(sys.path[0])

if len(pos) > 0:
    for i in sorted(pos, reverse=True):
        sys.path.pop(i)
##### part 1
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '../evogym')
sys.path.insert(0, root_dir)

# Robot Generation
# body, connections = sample_robot((5, 5))
robot_body = np.array([[3, 3, 4, 4, 4],
                       [1, 4, 4, 4, 4],
                       [3, 4, 4, 4, 4],
                       [1, 4, 4, 4, 4],
                       [3, 3, 3, 3, 4]])

body, connections = robot_body, get_full_connectivity(robot_body)

# Environment Generation
env = gym.make('Walker-v0', body=body)
# num_state = env.observation_space.shape[0]
# num_action = env.action_space.n

# Environment activation
_ob = env.reset()
total_steps = 0
while True:
    action = env.action_space.sample() - 1
    ob, reward, done, info = env.step(action)
    # RL.store_transition(ob, action, reward, _ob)
    if not total_steps % 100:
        print(total_steps)
        env.render()
    # if total_steps > 500:
    # RL.learn()

    if done:
        env.reset()
    total_steps += 1
    _ob = ob

env.close()

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(num_state, 32)
#         self.fc2 = nn.Linear(32, num_action)
#
#     def forward(self, x):
#         x = F.relu(self.fc1)
#         agent_action = self.fc2(x)
#
#         return agent_action
