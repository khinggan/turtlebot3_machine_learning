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

import pdb
import random
import time
import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel

class Env():
    def __init__(self, action_size) -> None:
        self.action_size = action_size
        
        # goal position
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.rob_ini_pos_x = -1.0
        self.rob_ini_pos_y = 0.0
        
        # robot pose
        self.position = Point()
        self.heading = 0.0
        
        # robot velocity
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        # laser scan
        self.laser_dim = 24
        self.scan_ranges = [float('Inf')] * self.laser_dim
        self.collide_range = 0.13   # If scan_range < collide_range -> collide

        # goal model spawner
        self.modelPath = "/home/khinggan/tb3_catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf"
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.check_goal_model)
        self.check_model = False
        self.goal_model_name = 'goal'

        # pubs,subs,services
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.get_odometry)
        self.sub_scan = rospy.Subscriber('scan', LaserScan, self.get_scan)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

    def get_odometry(self, odom):
        # set self.position, self.heading, self.linear_vel, self.angular_vel from `/odom` data
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.heading = yaw
        self.linear_vel = odom.twist.twist.linear.x
        self.angular_vel = odom.twist.twist.angular.z
    
    def get_scan(self, scan):
        # Initialize self.scan_ranges from `/scan` topic
        if len(scan.ranges) != self.laser_dim:
            print(f"more or less scans than expected! check model.sdf, got: {len(scan.ranges)}, expected: {self.laser_dim}")

        for i in range(self.laser_dim):
            if scan.ranges[i] == float('Inf'):
                self.scan_ranges[i] = 3.5    # max distance
            elif np.isnan(scan.ranges[i]):
                self.scan_ranges[i] = 0.0
            else:
                self.scan_ranges[i] = float(scan.ranges[i])

    def get_state(self):
        # get state 
        # X [scan, heading, distance, obstacle_min_range, obstacle_angle], done
        # O [scan, velocity, error], done; 
        #   [24dim: scan ranges; 2dim: linear_vel, angular_vel; 2dim: goal distance, goal heading], done
        collide = False
        if 0 < min(self.scan_ranges) < self.collide_range:
            collide = True

        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        goal = False
        if goal_distance < 0.2:
            goal = True

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x) - self.heading
        # normalize goal_angle in [-pi, pi]
        if goal_angle > pi:
            goal_angle -= 2 * pi
        elif goal_angle < -pi:
            goal_angle += 2 * pi
        
        state = self.scan_ranges + [self.linear_vel, self.angular_vel] + [goal_distance, goal_angle], collide, goal
        return state        

    def reward(self, state, collide, goal, angular_vel):
        # reward function
        r_goal_angle = -1 * abs(state[-1])                                 # goal_angle
        r_vangular = -1 * (angular_vel**2)                                 # action reward
        goal_dist_initial = math.sqrt((self.goal_x - self.rob_ini_pos_x)**2 + (self.goal_y - self.rob_ini_pos_y)**2)
        goal_dist = math.sqrt((self.goal_x - self.position.x)**2 + (self.goal_y - self.position.y)**2)
        r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1   # distance reward

        reward = r_goal_angle + r_distance + r_vangular - 1

        if collide:
            rospy.loginfo("------------- x Collision x -------------")
            self.pub_cmd_vel.publish(Twist())
            reward = -2000.0
        if goal:
            rospy.loginfo("------------- O Gooooooal O-------------")
            self.pub_cmd_vel.publish(Twist())
            reward = 2000.0

        return reward
    
    def step(self, action):
        # publish action (linear_v, angular_v)
        # return state, reward, done
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        state, collide, goal = self.get_state()
        reward = self.reward(state, collide, goal, ang_vel)
        done = collide or goal

        return np.array(state), reward, done

    def reset(self):
        # reset simulation
        # return state
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        self.goal_x, self.goal_y = self.respawn_goal()
        state, _, _ = self.get_state()

        return np.array(state)

    def respawn_goal(self):
        # resapwn goal in simulator
        # 1. delete goal
        # 2. check stage
        # 3. set goal position
        # 4. spawn goal model
        # 5. return goal position
        with open(self.modelPath, 'r') as f:
            goal_model = f.read()
        
        # delete model
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.goal_model_name)
                rospy.loginfo("Goal Model Deleted !!!")
                break
            else:
                pass
        
        goal_x = random.randrange(-7, 13) / 10.0
        goal_y = random.uniform(-1.2, -0.7) if random.random() < 0.5 else random.uniform(0.7, 1.3)
        time.sleep(0.5)

        # spawn model
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                goal_position = Point(goal_x, goal_y)
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.goal_model_name, goal_model, 'robotos_name_space', goal_position, "world")
                # rospy.loginfo("Goal position : %.1f, %.1f", 
                #               self.goal_position.position.x,
                #               self.goal_position.position.y)
                break
            else:
                pass
        
        return goal_x, goal_y
    
    def check_goal_model(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == self.goal_model_name:
                self.check_model = True