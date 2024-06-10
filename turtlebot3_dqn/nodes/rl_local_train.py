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

import torch

import csv
import pickle

from script.read_config import yaml_config

STAGE = 1
EPISODES = 2
TAU = 0.005

config = yaml_config()        # stages = config['RL']['stage']

STAGE = config['RL']['stage']
EPISODES = config['RL']['episodes']

stage_module_name = f'src.turtlebot3_dqn.environment_stage_{STAGE}'
# from src.turtlebot3_dqn.environment_stage_1 import Env
Env = getattr(importlib.import_module(stage_module_name), 'Env')

state_size = 26
action_size = 5
env = Env(action_size)

from ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.utils.agent import ReinforceAgent, device


if __name__ == '__main__':
    rospy.init_node("rl_local_train")
    agent = ReinforceAgent(state_size, action_size)
    
    scores, episodes, memory_lens, epsilons, episode_hours, episode_minutes, episode_seconds, collisions, goals = [], [], [], [], [], [], [], [], []
    global_step = 0
    best_score = 0
    best_model_dict = None

    # start train EPISODES episodes
    start_time = time.time()
    for e in range(1, EPISODES+1):
        done = False
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        score = 0
        collision = 0
        goal = 0
        for t in range(agent.episode_step):
            action = agent.getAction(state)

            next_state, reward, done = env.step(action)

            # check goal or collision
            if reward == 200:
                goal += 1
            
            if reward == -200:
                collision += 1

            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            agent.appendMemory(state, action, reward, next_state)

            agent.trainModel()
            score += reward
            state = next_state

            if t >= 180:
                rospy.loginfo("Time out!!")
                done = True

            # update epsilon
            agent.epsilon = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * \
                            math.exp(-1. * global_step / agent.epsilon_decay)
            
            # soft update target network
            target_net_state_dict = agent.target_model.state_dict()
            policy_net_state_dict = agent.model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            agent.target_model.load_state_dict(target_net_state_dict)

            if done:
                # agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                memory_lens.append(len(agent.memory))
                epsilons.append(agent.epsilon)
                episode_hours.append(h)
                episode_minutes.append(m)
                episode_seconds.append(s)
                collisions.append(collision)
                goals.append(goal)

                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              e, score, len(agent.memory), agent.epsilon, h, m, s)
                # save best model
                if score > best_score:
                    best_score = score
                    best_model_dict = agent.model.state_dict()
                    # SAVE TRAINED DICT
                    save_dict_directory = os.environ['ROSFRLPATH'] + "model_dicts/saved_dict/"
                    if not os.path.exists(save_dict_directory):
                        os.makedirs(save_dict_directory)
                    with open(save_dict_directory + "RL_episode_{}_stage_{}.pkl".format(EPISODES, STAGE), 'wb') as md:
                        pickle.dump(agent.model.state_dict(), md)
                        print("BEST SCORE MODEL SAVE: Episode = {}, Best Score = {}".format(e, best_score))
                break

            global_step += 1
            # if global_step % agent.target_update == 0:
            #     rospy.loginfo("UPDATE TARGET NETWORK")

        # if agent.epsilon > agent.epsilon_min:
        #     agent.epsilon *= agent.epsilon_decay
    end_time = time.time()
    # SAVE EXPERIMENT DATA
    directory_path = os.environ['ROSFRLPATH'] + "data/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(directory_path + "RL_episode_{}_stage_{}.csv".format(EPISODES, STAGE), 'a') as d:
        writer = csv.writer(d)
        writer.writerows([item for item in zip(scores, episodes, memory_lens, epsilons, episode_hours, episode_minutes, episode_seconds, collisions, goals)])
        print([item for item in zip(scores, episodes, memory_lens, epsilons, episode_hours, episode_minutes, episode_seconds, collisions, goals)])

    # SAVE TRAINED DICT
    if best_model_dict == None:
        save_dict_directory = os.environ['ROSFRLPATH'] + "model_dicts/saved_dict/"
        if not os.path.exists(save_dict_directory):
            os.makedirs(save_dict_directory)
        with open(save_dict_directory + "RL_episode_{}_stage_{}.pkl".format(EPISODES, STAGE), 'wb') as md:
            pickle.dump(agent.model.state_dict(), md)

    print("Total Train Time is : {} seconds".format(end_time - start_time))