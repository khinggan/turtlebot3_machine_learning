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
from collections import deque
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
# from ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.utils.agent import ReinforceAgent

config = yaml_config()

current_client = config['FRL']['client']
# Find and print the matching client information
clients = config['FRL']['clients'].get(current_client, None)

CURR_CID = clients['cid']
ENV = clients['env']
ENVS = "".join([str(config['FRL']['clients'][v]['env']) for v in config['FRL']['clients']])
LOCAL_EPISODES = clients['lep']
ROUND = config['FRL']['server']['round']
MODEL = config['MODEL']

from ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.src.turtlebot3_dqn.environment_train import Env
agent_module = 'ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.utils.agent'
Agent = getattr(importlib.import_module(agent_module), f'{MODEL}Agent')

TAU = 0.005

class FRLClient:
    def __init__(self, state_size=26, action_size=5) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.agent = Agent(state_size, action_size)
        self.env = Env(action_size)
        
        self.global_step = 0
        self.best_score = float('-inf')
        self.best_model_dict = None
        self.score_queue = deque([], maxlen=10)
        self.model_queue = deque([], maxlen=10)

        # check for simulation stuck; which may leads to high score in useless model
        self.check_stuck = deque([i for i in range(20)], maxlen=20)

        self.new_best_score = False

        self.local_train_service = rospy.Service('client_{}_local_train_service'.format(CURR_CID), LocalTrain, self.handle_local_train)

    def handle_local_train(self, request):
        # Train and return trained model dict
        global_model_dict_pickle = request.req
        global_model_dict = pickle.loads(global_model_dict_pickle)
        print("#### ROUND {}: CLIENT {} local train on ENV {} #### ".format(request.round, CURR_CID, ENV))

        # Initialize agent model with global model dict, update target model
        self.agent.model.load_state_dict(global_model_dict)
        self.agent.updateTargetModel()
        
        episodes, scores, memory_lens, epsilons, episode_seconds = [], [], [], [], []

        # start train EPISODES episodes
        start_time = time.time()
        for e in range(1, LOCAL_EPISODES+1):
            done = False
            state = self.env.reset()
            score = 0.0

            # pdb.set_trace()

            for t in range(self.agent.episode_step):
                action = self.agent.getAction(state)

                next_state, reward, done = self.env.step(action)

                self.agent.appendMemory(state, action, reward, next_state)

                self.agent.trainModel()
                
                score += reward
                state = next_state

                # check simulator stuck
                if self.sim_stuck(state=state):
                    score = -2000
                    done = True
                
                if ENV == 4: 
                    thresh = 500
                else:
                    thresh = 240
                if t >= thresh:
                    rospy.loginfo("Time out!!")
                    done = True

                # # update epsilon
                self.agent.epsilon = self.agent.epsilon_end + \
                                    (self.agent.epsilon_start - self.agent.epsilon_end) * \
                                    math.exp(-1. * self.global_step / self.agent.epsilon_decay)
                
                # soft update target network
                # target_net_state_dict = self.agent.target_model.state_dict()
                # policy_net_state_dict = self.agent.model.state_dict()
                # for key in policy_net_state_dict:
                #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                # self.agent.target_model.load_state_dict(target_net_state_dict)

                if done:
                    scores.append(score)
                    episodes.append(e)
                    memory_lens.append(len(self.agent.memory))
                    epsilons.append(self.agent.epsilon)
                    s = int(time.time() - start_time)    # second
                    episode_seconds.append(s)

                    rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %f',
                                e, score, len(self.agent.memory), self.agent.epsilon, s)
                    
                    self.score_queue.append(score)
                    self.model_queue.append(self.agent.model.state_dict())
                    
                    break

                self.global_step += 1
                if self.global_step % self.agent.target_update == 0:
                    self.agent.updateTargetModel()
                    rospy.loginfo("UPDATE TARGET NETWORK")
            # if agent.epsilon > agent.epsilon_min:
            #     agent.epsilon *= agent.epsilon_decay
        
            # Get Best Model
            if e % 10 == 0:
                mean_score = sum(self.score_queue) / len(self.score_queue)
                # save best model
                if mean_score > self.best_score:
                    self.best_score = mean_score
                    # get corresponding model dict
                    max_score = max(self.score_queue)
                    max_score_ind = self.score_queue.index(max_score)
                    self.best_model_dict = self.model_queue[max_score_ind]
                    
                    # SAVE TRAINED DICT
                    save_dict_directory = os.environ['ROSFRLPATH'] + "model_dicts/saved_dict/"
                    if not os.path.exists(save_dict_directory):
                        os.makedirs(save_dict_directory)
                    with open(save_dict_directory + "FRL_{}_{}eps_env{}.pkl".format(MODEL, LOCAL_EPISODES, ENV), 'wb') as md:
                        pickle.dump(self.best_model_dict, md)
                        print("BEST SCORE MODEL SAVE: Episode = {}, Best Score = {}".format(e, self.best_score))

        # state = self.env.reset()
        end_time = time.time()

        # SAVE EXPERIMENT DATA
        directory_path = os.environ['ROSFRLPATH'] + "data/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(directory_path + "FRL_{}_{}leps_{}rnd_c{}_env{}_envs{}.csv".format(MODEL, LOCAL_EPISODES,  ROUND, CURR_CID, ENV, ENVS), 'a') as d:
            writer = csv.writer(d)
            writer.writerows([item for item in zip(scores, episodes, memory_lens, epsilons, episode_seconds)])

        print("Total Train Time on client {} is : {} seconds".format(CURR_CID, end_time - start_time))
        # if self.new_best_score: 
        #     trained_model_dict_pickle = pickle.dumps(self.best_model_dict)
        #     self.new_best_score = False
        # else: 
        #     trained_model_dict_pickle = pickle.dumps(self.agent.model.state_dict())

        response = LocalTrainResponse()
        response.resp = pickle.dumps(self.best_model_dict if self.best_model_dict else self.model_queue[0])
        response.cid = CURR_CID
        response.round = request.round
        return response
    
    def sim_stuck(self, state):
        # store latest 20 state, if distance and heading not change in 20 steps, simulator is stucked
        self.check_stuck.append((state[-1], state[-2]))
        
        first_element = self.check_stuck[0]
        if all(element == first_element for element in self.check_stuck):
            rospy.loginfo("!!!!!!!!Maybe Simulator is Stuck!!!!!!!!!")
            return True
        return False

if __name__ == '__main__':
    """client service that get global model, train locally, then return local trained model"""
    state_size = 28
    action_size = 5

    rospy.init_node('client_{}_local_train'.format(CURR_CID))
    frl_client = FRLClient(state_size=state_size, action_size=action_size)
    rospy.loginfo("Client {}:  Train global model!!!".format(CURR_CID))
    rospy.spin()