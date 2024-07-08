#!/usr/bin/env python

# ** Author: khinggan ** 
# ** Email: khinggan2013@gmail.com **

"""Modification of ROBOTIS turtlebot3_machine_learning algorithm to PyTorch version 
according to PyTorch Official Tutorial of Reinforcement Learning: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
import importlib
import torch
import rospy
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle

from script.read_config import yaml_config
from ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.utils.agent import device

config = yaml_config()

MODEL = config['MODEL']

# TEST
TEST_STAGE = config['TEST']['env']               # Test environment
TEST_EPOCHES = config['TEST']['eps']             # Test episodes
TYPE = config['TEST']['type']

# RL
EPS = config['RL']['eps']
ENV = config['RL']['env']

# FRL
# LOCAL_EPISODES = config['FRL']['client']['local_episode']
# ROUND = config['FRL']['server']['round']
# STAGES = config['FRL']['server']['stages']

from ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.src.turtlebot3_dqn.environment_test import Env

agent_module = 'ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.utils.agent'
Agent = getattr(importlib.import_module(agent_module), f'{MODEL}Agent')

class TestTrainedModel:
    def __init__(self, state_size=26, action_size=5) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.agent = Agent(state_size, action_size)
        self.env = Env(action_size, ENV)

    def test_trained_model(self):
        # load trained dict
        if TYPE == 'RL':
            model_dict_file_name = "RL_{}_{}eps_env{}.pkl".format(MODEL, EPS, ENV)
        elif TYPE == 'FRL':
            # model_dict_file_name = "{}_localep_{}_totalround_{}_stages_{}.pkl".format(TYPE, LOCAL_EPISODES, ROUND, STAGES)
            print("FRL")
        else:
            print("FAIL TO LOAD TRAINED MODEL!!!!")
        
        with open(os.environ['ROSFRLPATH'] + "model_dicts/saved_dict/" + model_dict_file_name, 'rb') as md:
            model_dict = pickle.load(md)

        # Initialize agent model with global model dict
        self.agent.model.load_state_dict(model_dict)
        self.agent.model.eval()

        scores, episodes = [], []
        global_step = 0
        start_time = time.time()
        
        collision = 0
        goal = 0

        for e in range(1, TEST_EPOCHES+1):
            done = False
            state = self.env.reset()
            score = 0
            for t in range(self.agent.episode_step):
                action = self.agent.model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).max(1).indices.view(1, 1).item()

                next_state, reward, done = self.env.step(action)

                score += reward
                state = next_state

                if t >= 500:
                    rospy.loginfo("Time out!!")
                    done = True

                if done:
                    scores.append(score)
                    episodes.append(e)
                    s = int(time.time() - start_time)

                    rospy.loginfo('Ep: %d score: %.2f time: %f',
                                e, score, s)
                    
                    if reward == 2000:
                        goal += 1
                    elif reward == -2000:
                        collision += 1

                    break

                global_step += 1
        
        print("Goal reached = {}, Collision = {}, Goal rate = {} Collision rate = {}".format(goal, collision, goal * 1.0 / TEST_EPOCHES, collision * 1.0 / TEST_EPOCHES))

if __name__ == '__main__':
    """Train RL Model on Each Environment"""
    # For Stage 2, 3, 4, use 28 dim model input (obstacle_min_range, obstacle_angle)
    if ENV in (2, 3, 4) or TYPE == "FRL": 
        state_size = 28
    else:
        state_size = 26
    action_size = 5

    rospy.init_node('test_trained_model')
    rl_local = TestTrainedModel(state_size=state_size, action_size=action_size)
    rl_local.test_trained_model()
    rospy.loginfo("Test RL Model Trained on Stage {} Finished".format(ENV))
    rospy.spin()