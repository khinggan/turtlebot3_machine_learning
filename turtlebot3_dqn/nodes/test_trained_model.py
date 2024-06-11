#!/usr/bin/env python

# ** Author: khinggan ** 
# ** Email: khinggan2013@gmail.com **

"""Modification of ROBOTIS turtlebot3_machine_learning algorithm to PyTorch version 
according to PyTorch Official Tutorial of Reinforcement Learning: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import rospy
import numpy as np
import random
import time
from collections import deque, namedtuple
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import importlib

import torch
from torch import nn
import torch.nn.functional as F

import pickle

from script.read_config import yaml_config

# If you want to use CUDA. But, make sure all machines has CUDA compatibility. Otherwise, use cpu
# device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
print(f"Using {device} device")

# TEST
TEST_EPOCHES = 10
TEST_STAGE = 1

# RL
TRAINED_EPISODES = 2
TRAINED_STAGE = 1

# FRL
LOCAL_EPISODES = 10
ROUND = 1
STAGES = "21"

config = yaml_config()        # stages = config['RL']['stage']

# TEST
TEST_STAGE = config['TEST']['stage']               # Test stage
TEST_EPOCHES = config['TEST']['test_epoches']     # Test episodes
TYPE = config['TEST']['type']

# RL
TRAINED_EPISODES = config['RL']['episodes']
TRAINED_STAGE = config['RL']['stage']

# FRL
LOCAL_EPISODES = config['FRL']['client']['local_episode']
ROUND = config['FRL']['server']['round']
STAGES = config['FRL']['server']['stages']

from src.turtlebot3_dqn.environment_stage_test import Env

state_size = 26
action_size = 5
env = Env(action_size)

from ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.utils.agent import ReinforceAgent

if __name__ == '__main__':
    rospy.init_node('test_trained_model')

    state_size = 26
    action_size = 5

    env = Env(action_size)

    agent = ReinforceAgent(state_size, action_size)
    # load trained dict
    if TYPE == 'RL':
        model_dict_file_name = "{}_episode_{}_stage_{}.pkl".format(TYPE, TRAINED_EPISODES, TRAINED_STAGE)
    elif TYPE == 'FRL':
        model_dict_file_name = "{}_localep_{}_totalround_{}_stages_{}.pkl".format(TYPE, LOCAL_EPISODES, ROUND, STAGES)
    else:
        print("FAIL TO LOAD TRAINED MODEL!!!!")

    with open(os.environ['ROSFRLPATH'] + "model_dicts/saved_dict/" + model_dict_file_name, 'rb') as md:
        model_dict = pickle.load(md)

    # Initialize agent model with global model dict
    agent.model.load_state_dict(model_dict)
    agent.model.eval()

    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    collision = 0
    goal = 0

    for e in range(1, TEST_EPOCHES+1):
        done = False
        state = env.reset()
        score = 0
        for t in range(agent.episode_step):
            # action = agent.getAction(state)
            action = agent.model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).max(1).indices.view(1, 1).item()

            next_state, reward, done = env.step(action)

            score += reward
            state = next_state

            if t >= 180:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d score: %.2f time: %d:%02d:%02d',
                              e, score, h, m, s)
                
                if reward == 200:
                    goal += 1
                elif reward == -200:
                    collision += 1

                break

            global_step += 1
    
    print("Goal reached = {}, Collision = {}, Goal rate = {} Collision rate = {}".format(goal, collision, goal * 1.0 / TEST_EPOCHES, collision * 1.0 / TEST_EPOCHES))