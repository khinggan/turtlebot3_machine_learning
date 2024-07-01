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
from ros1_ws.src.turtlebot3_machine_learning.turtlebot3_dqn.utils.agent import ReinforceAgent

config = yaml_config()        # stages = config['FRL']['server']['stages']
CURR_CID = config['FRL']['client']['curr_cid'] if config['FRL']['client']['curr_cid'] is not None else 1
STAGE = config['FRL']['client']['stage'] if config['FRL']['client']['stage'] is not None else 1
STAGES = config['FRL']['server']['stages']
LOCAL_EPISODES = config['FRL']['client']['local_episode'] if config['FRL']['client']['local_episode'] is not None else 2
ROUND = config['FRL']['server']['round'] if config['FRL']['server']['round'] is not None else 2
stage_module_name = f'src.turtlebot3_dqn.environment_stage_{STAGE}'
Env = getattr(importlib.import_module(stage_module_name), 'Env')
TAU = 0.005

class FRLClient:
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

        self.new_best_score = False

        self.local_train_service = rospy.Service('client_{}_local_train_service'.format(CURR_CID), LocalTrain, self.handle_local_train)

    def handle_local_train(self, request):
        # Train and return trained model dict
        global_model_dict_pickle = request.req
        global_model_dict = pickle.loads(global_model_dict_pickle)
        print("#### ROUND {}: CLIENT {} local train on Stage {} #### ".format(request.round, CURR_CID, STAGE))

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
                
                if t >= 240:
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
                
                    # save best model
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model_dict = self.agent.model.state_dict()
                        self.new_best_score = True
                        rospy.loginfo("New Best Score: {}, saved best model!!".format(self.best_score))
                    break

                self.global_step += 1
                if self.global_step % self.agent.target_update == 0:
                    self.agent.updateTargetModel()
                    rospy.loginfo("UPDATE TARGET NETWORK")
            # if agent.epsilon > agent.epsilon_min:
            #     agent.epsilon *= agent.epsilon_decay
        
        state = self.env.reset()
        end_time = time.time()

        # SAVE EXPERIMENT DATA
        directory_path = os.environ['ROSFRLPATH'] + "data/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(directory_path + "FRL_localep_{}_totalround_{}_client_{}_stage_{}_stages{}.csv".format(LOCAL_EPISODES,  ROUND, CURR_CID, STAGE, STAGES), 'a') as d:
            writer = csv.writer(d)
            writer.writerows([item for item in zip(scores, episodes, memory_lens, epsilons, episode_seconds)])

        print("Total Train Time on client {} is : {} seconds".format(CURR_CID, end_time - start_time))
        if self.new_best_score: 
            trained_model_dict_pickle = pickle.dumps(self.best_model_dict)
            self.new_best_score = False
        else: 
            trained_model_dict_pickle = pickle.dumps(self.agent.model.state_dict())

        response = LocalTrainResponse()
        response.resp = trained_model_dict_pickle
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
    # For Stage 2, 3, 4, use 28 dim model input (obstacle_min_range, obstacle_angle)
    if STAGE in (2, 3, 4, 5): 
        state_size = 28
    else:
        state_size = 26
    action_size = 5

    rospy.init_node('client_{}_local_train'.format(CURR_CID))
    frl_client = FRLClient(state_size=state_size, action_size=action_size)
    rospy.loginfo("Client {}:  Train global model!!!".format(CURR_CID))
    rospy.spin()