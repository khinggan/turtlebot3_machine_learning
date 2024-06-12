#!/usr/bin/env python

# ** Author: khinggan ** 
# ** Email: khinggan2013@gmail.com **

"""
Federated reinforcement learning client local training  
1. Modified ROBOTIS turtlebot3_machine_learning algorithm (https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning) to PyTorch version according to PyTorch Official Tutorial of Reinforcement Learning: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
2. Federated Reinforcement Learning (FRL) Client. 
3. Using ros1_bridge (https://github.com/ros2/ros1_bridge) to separate clients and transmit data using customized service type (LocalTrain)

First, getting request (global model) from the FRL server, then, train it locally, finally upload the trained model to FRL server
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
from turtlebot3_dqn.srv import LocalTrain, LocalTrainResponse


from script.read_config import yaml_config

CURR_CID = 1
STAGE = 1
LOCAL_EPISODES = 2
ROUND = 2
TAU = 0.005

config = yaml_config()        # stages = config['FRL']['server']['stages']

CURR_CID = config['FRL']['client']['curr_cid']
STAGE = config['FRL']['client']['stage']
LOCAL_EPISODES = config['FRL']['client']['local_episode']
ROUND = config['FRL']['server']['round']

stage_module_name = f'src.turtlebot3_dqn.environment_stage_{STAGE}'
# from src.turtlebot3_dqn.environment_stage_1 import Env
Env = getattr(importlib.import_module(stage_module_name), 'Env')

state_size = 26
action_size = 5
env = Env(action_size)
from ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.utils.agent import ReinforceAgent

agent = ReinforceAgent(state_size, action_size)

def start_train(request):
    global_model_dict = request.req
    model_dict = pickle.loads(global_model_dict)

    print("#### ROUND {}: CLIENT {} local train on Stage {} #### ".format(request.round, CURR_CID, STAGE))

    # Initialize agent model with global model dict
    agent.model.load_state_dict(model_dict)
    agent.updateTargetModel()
    
    scores, episodes, episode_length, memory_lens, epsilons, episode_hours, episode_minutes, episode_seconds, collisions, goals = [], [], [], [], [], [], [], [], [], []
    global_step = 0
    best_score = 0
    best_model_dict = model_dict

    # start train EPISODES episodes
    start_time = time.time()
    for e in range(1, LOCAL_EPISODES+1):
        done = False
        state = env.reset()
        score = 0.0
        collision_times = 0
        goal_times = 0
        for t in range(agent.episode_step):
            action = agent.getAction(state)

            next_state, reward, done = env.step(action)

            # check goal or collision
            if reward == 200:
                goal_times += 1
            if reward == -200:
                collision_times += 1

            agent.appendMemory(state, action, reward, next_state)

            agent.trainModel()
            score += reward
            state = next_state

            if t >= 240:
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
                scores.append(score)
                episodes.append(e)
                episode_length.append(t)
                memory_lens.append(len(agent.memory))
                epsilons.append(agent.epsilon)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                episode_hours.append(h)
                episode_minutes.append(m)
                episode_seconds.append(s)
                collisions.append(collision_times)
                goals.append(goal_times)

                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              e, score, len(agent.memory), agent.epsilon, h, m, s)
                # save best model
                if score > best_score:
                    best_score = score
                    best_model_dict = agent.model.state_dict()
                    print("BEST SCORE MODEL SAVE: Episode = {}, Best Score = {}".format(e, best_score))
                break

            global_step += 1

        # if agent.epsilon > agent.epsilon_min:
        #     agent.epsilon *= agent.epsilon_decay
    
    state = env.reset()
    end_time = time.time()

    # SAVE EXPERIMENT DATA
    directory_path = os.environ['ROSFRLPATH'] + "data/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(directory_path + "FRL_localep_{}_totalround_{}_client_{}_stage_{}.csv".format(LOCAL_EPISODES,  ROUND, CURR_CID, STAGE), 'a') as d:
        writer = csv.writer(d)
        writer.writerows([item for item in zip(scores, episodes, memory_lens, epsilons, episode_hours, episode_minutes, episode_seconds, collisions, goals)])
        print([item for item in zip(scores, episodes, memory_lens, epsilons, episode_hours, episode_minutes, episode_seconds, collisions, goals)])

    print("Total Train Time on client {} is : {} seconds".format(CURR_CID, end_time - start_time))
    compressed_model_dict = pickle.dumps(best_model_dict)
    return compressed_model_dict

def handle_local_train(request):
    trained_model_dict = start_train(request)

    response = LocalTrainResponse()

    response.resp = trained_model_dict
    response.cid = CURR_CID
    response.round = request.round
    return response


def client_local_train():
    """client service that get global model, train locally, then return local trained model
    """
    rospy.init_node('client_{}_local_train'.format(CURR_CID))
    s = rospy.Service('client_{}_local_train_service'.format(CURR_CID), LocalTrain, handle_local_train)
    print("Client {} Train global model".format(CURR_CID))
    rospy.spin()

if __name__ == '__main__':
    client_local_train()