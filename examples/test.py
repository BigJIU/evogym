import os
import sys

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from collections import namedtuple
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import evogym.envs
from evogym import sample_robot
import numpy as np
from evogym import get_full_connectivity

sys.path.append('./utils')
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
from ppo import run_myppo
from algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure


# Hyper-parameters
seed = 1
render = True
num_episodes = 10
# env = gym.make('CartPole-v0').unwrapped
# num_state = env.observation_space.shape[0]
# num_action = env.action_space.n
# torch.manual_seed(seed)
# env.seed(seed)

#Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.input_layer = nn.Linear(16, 64)
#
#     def forward(self, x):
#         x = x.view(-1, 16)
#         # 通过输入层处理输入数据
#         x = self.input_layer(x)
#         return x


if __name__ == '__main__':

    # Robot Generation
    # body, connections = sample_robot((5, 5))
    robot_body = np.array([[3, 3, 4],
                           [1, 4, 4],
                           [1, 4, 4]])

    body, connections = robot_body, get_full_connectivity(robot_body)

    # Environment Generation
    # env = gym.make('Walker-v0', body=body)
    # num_action = env.observation_space.shape[0]
    # num_state = env.observation_space.shape[0]
    # # num_action = env.action_space.n
    #
    # # Environment activation
    # observation = env.reset()

    iters = 1100
    # 创建模型和优化器
    # model = Net()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)

    tc = TerminationCondition(iters)
    temp_structure = robot_body, get_full_connectivity(robot_body)
    robot_structure = Structure(*temp_structure, 0)
    save_path_controller = os.path.join(root_dir, "saved_data",  "controller")
    run_myppo(structure=(robot_structure.body, robot_structure.connections), termination_condition=tc,
              saving_convention=(save_path_controller, robot_structure.label))

    # while True:
    #     # run_myppo(structure=(robot_structure.body, robot_structure.connections), termination_condition=tc,
    #     #         saving_convention=(save_path_controller, robot_structure.label))
    #     if iters % 10 == 0:
    #         # Play Games

    # #agent = DQN()
    # for i_ep in range(num_episodes):
    #     state = env.reset()
    #     if render: env.render()
    #     for t in range(10000):
    #         action = agent.select_action(state)
    #         next_state, reward, done, info = env.step(action)
    #         if render: env.render()
    #         transition = Transition(state, action, reward, next_state)
    #         agent.store_transition(transition)
    #         state = next_state
    #         if done or t >= 9999:
    #             agent.writer.add_scalar('live/finish_step', t + 1, global_step=i_ep)
    #             agent.update()
    #             if i_ep % 10 == 0:
    #                 print("episodes {}, step is {} ".format(i_ep, t))
    #             break

    # 训练模型
    # for i in range(1000):
    #     action_probs = model(torch.tensor(observation).float().unsqueeze(0))
    #     action = torch.distributions.Categorical(action_probs).sample()
    #     observation, reward, done, info = env.step(action.item())
    #     env.render()
    #     if done:
    #         observation = env.reset()
    #     loss = -torch.log(action_probs[0, action]) * reward
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    # env.close()
    # while True:
    #     total_steps = 0
    #     action = env.action_space.sample() - 1
    #     ob, reward, done, info = env.step(action)
    #     RL.store_transition(ob, action, reward, _ob)
    #     env.render()
    #     if total_steps > 500:
    #     RL.learn()
    #
    #     if done:
    #         env.reset()
    #         break
    #     total_steps += 1
    #     _ob = ob

    env.close()
