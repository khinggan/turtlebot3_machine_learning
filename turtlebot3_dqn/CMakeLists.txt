################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 2.8.3)
project(turtlebot3_dqn)

################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################
find_package(catkin REQUIRED COMPONENTS rospy roscpp std_msgs message_generation)

add_service_files(
  FILES
  LocalTrain.srv
)

################################################################################
# Setup for python modules and scripts
################################################################################
catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
)

################################################################################
# Declare ROS messages, services and actions
################################################################################

################################################################################
# Declare ROS dynamic reconfigure parameters
################################################################################

################################################################################
# Declare catkin specific configuration to be passed to dependent projects
################################################################################
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

################################################################################
# Build
################################################################################
include_directories(
  ${catkin_INCLUDE_DIRS}
)
################################################################################
# Install
################################################################################
catkin_install_python(PROGRAMS
  nodes/client.py
  nodes/moving_obstacle.py
  nodes/combination_obstacle_1.py
  nodes/combination_obstacle_2.py
  nodes/rl_local.py
  nodes/test.py
  src/${PROJECT_NAME}/environment_stage_5.py
  src/${PROJECT_NAME}/environment_train.py
  src/${PROJECT_NAME}/environment_test.py
  src/${PROJECT_NAME}/respawnGoal.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch save_model
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

################################################################################
# Test
################################################################################
