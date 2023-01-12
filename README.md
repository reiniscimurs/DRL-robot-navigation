# DRL-robot-navigation


Turtlebot branch of Deep Reinforcement Learning for mobile robot navigation in ROS Gazebo simulator. Using Twin Delayed Deep Deterministic Policy Gradient (TD3) neural network, a robot learns to navigate to a random goal point in a simulated environment while avoiding obstacles. Obstacles are detected by laser readings and a goal is given to the robot in polar coordinates. Trained in ROS Gazebo simulator with PyTorch.  Tested with ROS Noetic on Ubuntu 20.04 with python 3.8.10 and pytorch 1.10.

**Installation and code overview tutorial available** [here](https://medium.com/@reinis_86651/deep-reinforcement-learning-in-mobile-robot-navigation-tutorial-part1-installation-d62715722303)

Training example:
<p align="center">
    <img width=100% src="https://github.com/reiniscimurs/DRL-robot-navigation/blob/main/training.gif">
</p>



**ICRA 2022 and IEEE RA-L paper:**


Some more information about the implementation is available [here](https://ieeexplore.ieee.org/document/9645287?source=authoralert)

Please cite as:<br/>
```
@ARTICLE{9645287,
  author={Cimurs, Reinis and Suh, Il Hong and Lee, Jin Han},
  journal={IEEE Robotics and Automation Letters}, 
  title={Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning}, 
  year={2022},
  volume={7},
  number={2},
  pages={730-737},
  doi={10.1109/LRA.2021.3133591}}
```

## Installation
Main dependencies: 

* [ROS Noetic](http://wiki.ros.org/noetic/Installation)
* [PyTorch](https://pytorch.org/get-started/locally/)
* [Tensorboard](https://github.com/tensorflow/tensorboard)

Clone the repository:
```shell
$ cd ~
### Clone this repo
$ git clone https://github.com/reiniscimurs/DRL-robot-navigation
```
The network can be run with a standard 2D laser, but this implementation uses a simulated [3D Velodyne sensor](https://github.com/lmark1/velodyne_simulator)
This implementation branch supports turtlebot3 as robot models[Turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3)

Compile the workspace:
```shell
$ cd ~/DRL-robot-navigation/catkin_ws
### Compile
$ catkin_make
```

Open a terminal and set up sources:
```shell
$ export ROS_HOSTNAME=localhost
$ export ROS_MASTER_URI=http://localhost:11311
$ export ROS_PORT_SIM=11311
$ export GAZEBO_RESOURCE_PATH=~/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
$ source ~/.bashrc
$ cd ~/DRL-robot-navigation/catkin_ws
$ source devel/setup.bash
```

Run the training:
```shell
$ cd ~/DRL-robot-navigation/TD3
$ python3 train_velodyne_td3.py
```

To check the training process on tensorboard:
```shell
$ cd ~/DRL-robot-navigation/TD3
$ tensorboard --logdir runs
```

To kill the training process:
```shell
$ killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
```

Once training is completed, test the model:
```shell
$ cd ~/DRL-robot-navigation/TD3
$ python3 test_velodyne_td3.py
```

## Turtlebot setup
Install Turtlebot3 packages:
```shell
$ sudo apt-get install ros-kinetic-dynamixel-sdk
$ sudo apt-get install ros-kinetic-turtlebot3-msgs
$ sudo apt-get install ros-kinetic-turtlebot3
```

In file multi_robot_scenario.launch file change the value of following line to select the robot type:
https://github.com/reiniscimurs/DRL-robot-navigation/blob/12afc9558d864ff0312e4e52d430f2a9beefcade/TD3/assets/multi_robot_scenario.launch#L10

Supported turtlebo3 robot types - burger, waffle, waffle_pi

Turtlebot3 robots:
<p align="left">
    <img width=40% src="https://github.com/reiniscimurs/DRL-robot-navigation/blob/Noetic-Turtlebot/waffle.png">
    <img width=35.4% src="https://github.com/reiniscimurs/DRL-robot-navigation/blob/Noetic-Turtlebot/burger.png">
</p>

Gazebo environment:
<p align="center">
    <img width=80% src="https://github.com/reiniscimurs/DRL-robot-navigation/blob/main/env1.png">
</p>

Rviz:
<p align="center">
    <img width=80% src="https://github.com/reiniscimurs/DRL-robot-navigation/blob/main/velodyne.png">
</p>
