# DRL-robot-navigation

Deep Reinforcement Learning for mobile robot navigation in ROS Gazebo simulator. Using Twin Delayed Deep Deterministic Policy Gradient (TD3) neural network, a robot learns to navigate to a random goal point in a simulated environment while avoiding obstacles. Obstacles are detected by laser readings and a goal is given to the robot in polar coordinates. Trained in ROS Gazebo simulator with PyTorch. Currently un-tested ROS Noetic branch

Training example:
<p align="center">
    <img width=100% src="https://github.com/reiniscimurs/DRL-robot-navigation/blob/main/training.gif">
</p>


**Pre-print of the article:**

Some more information is given in the article at: https://arxiv.org/abs/2103.07119

Please cite as:<br/>
@misc{cimurs2021goaldriven,<br/>
      title={Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning}, <br/>
      author={Reinis Cimurs and Il Hong Suh and Jin Han Lee},<br/>
      year={2021},<br/>
      eprint={2103.07119},<br/>
      archivePrefix={arXiv},<br/>
      primaryClass={cs.RO}<br/>
}

Main dependencies: 

* [ROS Melodic](http://wiki.ros.org/melodic/Installation)
* [PyTorch](https://pytorch.org/get-started/locally/)

Clone the repository:
```shell
$ cd ~
### Clone this repo
$ git clone https://github.com/reiniscimurs/DRL-robot-navigation
```
The network can be run with a standard 2D laser, but this implementation uses a simulated [3D Velodyne sensor](https://github.com/lmark1/velodyne_simulator)

Compile the workspace:
```shell
$ cd ~/DRL-robot-navigation/catkin_ws
### Compile
$ catkin_make_isolated
```

Open a terminal and set up sources:
```shell
$ export ROS_HOSTNAME=localhost
$ export ROS_MASTER_URI=http://localhost:11311
$ export ROS_PORT_SIM=11311
$ export GAZEBO_RESOURCE_PATH=~/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
$ source ~/.bashrc
$ cd ~/DRL-robot-navigation/catkin_ws
$ source devel_isolated/setup.bash
### Run the training
$ cd ~/DRL-robot-navigation/TD3
$ python3 velodyne_td3.py
```

To kill the training process:
```shell
$ killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
```


Gazebo environment:
<p align="center">
    <img width=80% src="https://github.com/reiniscimurs/DRL-robot-navigation/blob/main/env1.png">
</p>

Rviz:
<p align="center">
    <img width=80% src="https://github.com/reiniscimurs/DRL-robot-navigation/blob/main/velodyne.png">
</p>
