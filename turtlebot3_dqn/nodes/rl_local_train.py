#!/usr/bin/env python

# ** Author: khinggan ** 
# ** Email: khinggan2013@gmail.com **

"""Modification of ROBOTIS turtlebot3_machine_learning algorithm to PyTorch version 
according to PyTorch Official Tutorial of Reinforcement Learning: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import math
import rospy
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import importlib
import csv
import pickle
from collections import deque

from script.read_config import yaml_config
from ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.utils.agent import ReinforceAgent

config = yaml_config()        # stages = config['RL']['stage']

STAGE = config['RL']['stage']
EPISODES = config['RL']['episodes']
stage_module_name = f'src.turtlebot3_dqn.environment_stage_{STAGE}'
Env = getattr(importlib.import_module(stage_module_name), 'Env')
TAU = 0.005

class RLLocal:
    def __init__(self, state_size=26, action_size=5) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.agent = ReinforceAgent(state_size, action_size)
        self.env = Env(action_size)
        
        self.global_step = 0
        self.best_score = float('-inf')
        self.best_model_dict = None

        # check for simulation stuck; which may leads to high score in useless model
        self.check_stuck = deque([i for i in range(20)], maxlen=20)
    
    def local_train(self):
        episodes, scores, memory_lens, epsilons, episode_seconds = [], [], [], [], []

        # start train EPISODES episodes
        start_time = time.time()
        for e in range(1, EPISODES+1):
            done = False
            state = self.env.reset()
            
            score = 0.0

            for t in range(self.agent.episode_step):
                action = self.agent.getAction(state)

                next_state, reward, done = self.env.step(action)

                self.agent.appendMemory(state, action, reward, next_state)

                self.agent.trainModel()
                
                score += reward
                state = next_state

                # check simulator stuck
                if self.sim_stuck(state=state):
                    score = -200
                    done = True
                
                if t >= 240:
                    rospy.loginfo("Time out!!")
                    done = True

                # update epsilon
                self.agent.epsilon = self.agent.epsilon_end + \
                                (self.agent.epsilon_start - self.agent.epsilon_end) * \
                                math.exp(-1. * self.global_step / self.agent.epsilon_decay)
                
                # soft update target network
                target_net_state_dict = self.agent.target_model.state_dict()
                policy_net_state_dict = self.agent.model.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.agent.target_model.load_state_dict(target_net_state_dict)

                if done:
                    scores.append(score)
                    episodes.append(e)
                    memory_lens.append(len(self.agent.memory))
                    epsilons.append(self.agent.epsilon)
                    s = int(time.time() - start_time)    # second
                    episode_seconds.append(s)

                    rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %f',
                                e, score, len(self.agent.memory), self.agent.epsilon, s)
                
                    # save best model
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model_dict = self.agent.model.state_dict()
                        # SAVE TRAINED DICT
                        save_dict_directory = os.environ['ROSFRLPATH'] + "model_dicts/saved_dict/"
                        if not os.path.exists(save_dict_directory):
                            os.makedirs(save_dict_directory)
                        with open(save_dict_directory + "RL_episode_{}_stage_{}.pkl".format(EPISODES, STAGE), 'wb') as md:
                            pickle.dump(self.agent.model.state_dict(), md)
                            print("BEST SCORE MODEL SAVE: Episode = {}, Best Score = {}".format(e, self.best_score))
                    break
                    # if e % 100 == 0:
                    #     save_dict_directory = os.environ['ROSFRLPATH'] + "model_dicts/saved_dict/"
                    #     if not os.path.exists(save_dict_directory):
                    #         os.makedirs(save_dict_directory)
                    #     with open(save_dict_directory + "RL_episode_{}_stage_{}.pkl".format(EPISODES, STAGE), 'wb') as md:
                    #         pickle.dump(self.agent.target_model.state_dict(), md)
                    # break  
                self.global_step += 1

        end_time = time.time()
        # SAVE EXPERIMENT DATA
        directory_path = os.environ['ROSFRLPATH'] + "data/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(directory_path + "RL_episode_{}_stage_{}.csv".format(EPISODES, STAGE), 'a') as d:
            writer = csv.writer(d)
            writer.writerows([item for item in zip(scores, episodes, memory_lens, epsilons, episode_seconds)])

        print("Total Train Time is : {} seconds".format(end_time - start_time))
    
    def sim_stuck(self, state):
        # store latest 20 state, if distance and heading not change in 20 steps, simulator is stucked
        if self.state_size == 28:
            self.check_stuck.append((state[-3], state[-4]))
        else:
            self.check_stuck.append((state[-1], state[-2]))
        
        first_element = self.check_stuck[0]
        if all(element == first_element for element in self.check_stuck):
            rospy.loginfo("!!!!!!!!Maybe Simulator is Stuck!!!!!!!!!")
            return True
        return False

if __name__ == '__main__':
    """Train RL Model on Each Environment"""
    # For Stage 2, 3, 4, use 28 dim model input (obstacle_min_range, obstacle_angle)
    if STAGE in (2, 3, 4): 
        state_size = 28
    else:
        state_size = 26
    action_size = 5

    rospy.init_node("rl_local_train")
    rl_local = RLLocal(state_size=state_size, action_size=action_size)
    rl_local.local_train()
    rospy.loginfo("RL Local Train on Stage {} Finished".format(STAGE))
    rospy.spin()