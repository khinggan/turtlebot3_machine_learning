#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
import xml.etree.ElementTree as ET

from script.read_config import yaml_config
config = yaml_config()
if config['TYPE'] == 'FRL':
    if config['MODE'] == 'TRAIN':
        current_client = config['FRL']['client']
        # Find and print the matching client information
        client = config['FRL']['clients'].get(current_client, None)
        ENV = client['env']
    elif config['MODE'] == 'TEST':
        ENV = config['TEST']['env']
    else:
        ENV = 0
elif config['TYPE'] == 'RL':
    if config['MODE'] == 'TRAIN':
        ENV = config['RL']['env']
    elif config['MODE'] == 'TEST':
        ENV = config['TEST']['env']
    else:
        ENV = 0
else:
    ENV = 0

NO_GOAL_SPAWN_MARGIN = 0.3

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        # self.modelPath = self.modelPath.replace('turtlebot3_machine_learning/turtlebot3_dqn/src/turtlebot3_dqn',
        #                                         'turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf')
        self.modelPath = os.environ['ROSFRLPATH'] + "ros1_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf"
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.goal_position = Pose()
        self.init_goal_x = -0.5
        self.init_goal_y = -1.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        # Env 2, 3
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        # Env 4
        # self.obstacle_5 = 0.0, 0.0
        # self.obstacle_6 = 0.0, 0.0
        # self.wall_obstacle = self.get_wall_obstacle()
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    def getPosition(self, position_check=False, delete=False):
        if delete:
            self.deleteModel()

        if ENV == 0:
            rospy.loginfo("!!!!!!!!!!!!!!!!!!!!!ENVIRONMENT IS ERROR!!!!!!!!!!!!!!!!!!!!!")

        if ENV in (1, 2, 3):
            while position_check:
                goal_x = random.randrange(-12, 13) / 10.0
                goal_y = random.randrange(-12, 13) / 10.0

                # goal_x = random.uniform(-1.2, -0.6) if random.random() < 0.5 else random.uniform(0.6, 1.3)
                # goal_y = random.uniform(-1.2, -0.6) if random.random() < 0.5 else random.uniform(0.6, 1.3)

                # goal_x = random.randrange(-6, 13) / 10.0
                # goal_y = random.uniform(-1.2, -0.6) if random.random() < 0.5 else random.uniform(0.6, 1.3)

                if abs(goal_x - self.obstacle_1[0]) <= 0.4 and abs(goal_y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_2[0]) <= 0.4 and abs(goal_y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_3[0]) <= 0.4 and abs(goal_y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_4[0]) <= 0.4 and abs(goal_y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                # To find far next distance
                if abs(goal_x - self.last_goal_x) < 2 and abs(goal_y - self.last_goal_y) < 2:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        elif ENV == 4:
            while position_check:
                # goal_x = random.randrange(-12, 13) / 10.0
                # goal_y = random.randrange(-12, 13) / 10.0
                # rospy.loginfo("Goal_X: {}, Gola_Y: {}".format(goal_x, goal_y))
                # if abs(goal_x - self.obstacle_5[0]) <= 0.4 and abs(goal_y - self.obstacle_5[1]) <= 0.4:
                #     position_check = True
                #     rospy.loginfo("+++Obstacle_1+++")
                # elif abs(goal_x - self.obstacle_6[0]) <= 0.4 and abs(goal_y - self.obstacle_6[1]) <= 0.4:
                #     position_check = True
                #     rospy.loginfo("---Obstacle_2---")
                # elif self.wall_obstacle != []:
                #     for obstacle in self.wall_obstacle:
                #         if goal_x < obstacle[0][0] and goal_x > obstacle[2][0]:
                #             if goal_y < obstacle[0][1] and goal_y > obstacle[2][1]:
                #                 position_check = True
                #     rospy.loginfo("===Obstacle_wall===")
                # else:
                #     position_check = False
                # # To find far next distance
                # if abs(goal_x - self.last_goal_x) < 2 and abs(goal_y - self.last_goal_y) < 2:
                #     position_check = True
                #     rospy.loginfo("***Obstacle_distance***")

                # self.goal_position.position.x = goal_x
                # self.goal_position.position.y = goal_y

                # goal_pose_list = [[1.0, 0.0], [2.0, -1.5], [0.2, -2.0], [2.0, 1.0], [0.7, 1.8],
                #                   [-1.9, 1.9], [-1.9,  0.2], [-1.9, -0.5], [-1.0, -2.0], [-0.5, -1.0],
                #                   [1.7, -1.0], [-0.5, 1.0], [-1.0, -2.0], [1.8, -0.2], [1.0, -1.9]]
                
                goal_pose_list = [[1.9130265557555863, 1.419769601729402], [1.9858305666673055, -1.5695984793679914], [-1.6830281284718707, -0.9828239539495467], [-2.0465180956021474, 0.8309501497125704], [1.8985508872255799, -1.1474801041168374], [-1.1643929197346798, 2.0697292138501227], [-1.9441451066198996, -0.18833570464729732], [-0.7051652367986321, 0.7206432058046168], [-2.0183836133273436, 1.1855550249178348], [-0.9249742580149467, -0.48321615766585], [0.16745489967128657, -0.9454205548414154], [1.4901635304952823, -2.0419928373834217], [-0.9229672659596331, 0.4149471465987742], [-1.842497000904093, 0.5117387388912062], [-0.6501667835014515, 1.8913601583562332], [-0.5132273726080512, 0.31605282454100125], [1.9654058221420967, 1.904806274611897], [-0.4452115131813823, -1.0691156216232653], [-0.49462140981575775, -0.4008886316477045], [-0.18718332657145664, -0.45000427216304906], [-0.9423652324324987, -2.0762603463601708], [0.9405371133648548, -2.0839052177482547], [-1.2652529603684615, -0.8195149668619834], [-0.014773793392194978, 1.3190925233123796], [0.8325896920350342, 0.33507507514842994], [0.9844127277481456, 0.7828819729564263], [-1.237571246193347, 1.1731208075358213], [0.8458209514253321, -0.9001506646366897], [0.6524461894318097, 1.2907632684823485], [0.11961962847846319, -1.727880615178349], [0.1835524779156299, 1.891497024096449], [-0.22980391250332222, 2.025532897971384], [0.6517933372088445, -0.15736397293901083], [0.1721844103233896, 0.9838469916488508], [-1.7906204314474419, 2.011262567895361], [-0.39194761066728745, 0.6184039595782802], [0.4484959769439305, -1.4457766946571509], [-1.0349196951518376, -1.6556139536869527], [-1.7601845373631644, -2.062099031075478], [-1.1752240513824856, -0.11323588714886723], [-1.33310521999402, -2.002549016652556], [1.7520972521184555, -0.4121618094037396], [-0.9754328623093458, -1.2210483662573384], [1.6098931661394413, 0.9459875678075096], [-0.616325265706231, -1.8864770827649084], [1.651553008291042, 2.0244878997794173], [-0.912408243027429, -0.8475742030218412], [-0.5361381979797366, 1.1823288647183414], [1.654442965285996, -0.10680670664462033], [1.2464206411439336, 0.023079674669073125]]
                self.index = random.randrange(0, len(goal_pose_list))
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False
                self.goal_position.position.x = goal_pose_list[self.index][0]
                self.goal_position.position.y = goal_pose_list[self.index][1]

                # goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
                # goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]
                # self.index = random.randrange(0, 13)
                # print(self.index, self.last_index)
                # if self.last_index == self.index:
                #     position_check = True
                # else:
                #     self.last_index = self.index
                #     position_check = False
                # self.goal_position.position.x = goal_x_list[self.index]
                # self.goal_position.position.y = goal_y_list[self.index]
        elif ENV == 5:
            while position_check:
                goal_pose_list = [[4, -1], [4, -2], [3, -2], [2, -2], [2, -3.3], 
                                  [3, -3.3], [4, -3], [4, -4], [3, -4], [2, -4],
                                  [1, -4], [1, -3], [2.5, -2.5], [3, -3.5], [1.8, -3.5]]
                self.index = random.randrange(0, len(goal_pose_list))
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False
                self.goal_position.position.x = goal_pose_list[self.index][0]
                self.goal_position.position.y = goal_pose_list[self.index][1]
        elif ENV == 6:
            while position_check:
                goal_pose_list = [[0, 4], [0.2, 3], [1, 3], [6, 1], [6, 0], 
                                  [5, 0.2], [4.5, 0.5], [5, -1.1], [5, 1.5], [1, -4],
                                  [-1, -2.5], [-1, -4], [1.5, 4.3], [3, 0.5], [7, 1]]
                self.index = random.randrange(0, len(goal_pose_list))
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False
                self.goal_position.position.x = goal_pose_list[self.index][0]
                self.goal_position.position.y = goal_pose_list[self.index][1]
        elif ENV == 7:
            while position_check:
                goal_pose_list = [[3,-3], [1.5, -3.5], [-2, -3], [-3, -2], [-3, -3], 
                                  [-3, 3], [-1.5, 3.5], [2, 3], [3, 3], [3, -2],
                                  [-3, -1], [-1.5, -1.5], [3, 2], [2, 2], [-0.5, 3.5]]
                self.index = random.randrange(0, len(goal_pose_list))
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False
                self.goal_position.position.x = goal_pose_list[self.index][0]
                self.goal_position.position.y = goal_pose_list[self.index][1]
        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y

    # def get_wall_obstacle(self):
    #     wall_obstacle = []
    #     if ENV == 4:
    #         model_sdf = os.environ['ROSFRLPATH'] + "ros1_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_plaza/model.sdf"
    #     # elif ENV in (1, 2, 3):
    #     #     model_sdf = os.environ['ROSFRLPATH'] + "ros1_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/model.sdf"
    #     else:
    #         model_sdf = ""
        
    #     if model_sdf != "":
    #         tree = ET.parse(model_sdf)
    #         root = tree.getroot()
    #         for wall in root.find('model').findall('link'):
    #             pose = wall.find('pose').text.split(" ")
    #             size = wall.find('collision').find('geometry').find('box').find('size').text.split()
    #             rotation = float(pose[-1])
    #             pose_x = float(pose[0])
    #             pose_y = float(pose[1])
    #             if rotation == 0:
    #                 size_x = float(size[0]) + NO_GOAL_SPAWN_MARGIN * 2
    #                 size_y = float(size[1]) + NO_GOAL_SPAWN_MARGIN * 2
    #             else:
    #                 size_x = float(size[1]) + NO_GOAL_SPAWN_MARGIN * 2
    #                 size_y = float(size[0]) + NO_GOAL_SPAWN_MARGIN * 2
    #             point_1 = [pose_x + size_x / 2, pose_y + size_y / 2]
    #             point_2 = [point_1[0], point_1[1] - size_y]
    #             point_3 = [point_1[0] - size_x, point_1[1] - size_y ]
    #             point_4 = [point_1[0] - size_x, point_1[1] ]
    #             wall_points = [point_1, point_2, point_3, point_4]
    #             wall_obstacle.append(wall_points)
    #     return wall_obstacle