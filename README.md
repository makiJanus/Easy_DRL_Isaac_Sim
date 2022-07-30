# Readme.md

<p align="center">
  <img src="images/banner_slim.jpg" alt="banner"/>
</p>

<p align="center">
  An easy-to-use library to develop deep reinforcement learning experiments in the Isaac Sim simulator. 
</p>


## Contents

- [Description](#Description)
- [Features](#Features)
- [How to install](#How-to-install)
- [How to use](#How-to-use)
- [Example](#Example)
- [Notes](#Notes)

## Description
<a name="Description"/>

<p align="justify">
  The library aims to conduct experiments and DRL training with mobile robots in a realistic environment using standard libraries such as OpenAi Gym, Pytorch, Isaac Sim extensions, and SB3 in a unified and ready-to-use framework. Furthermore, the library is easy to use, configure, and customize all different robots, sensors, environments, and methods that allow and facilitate research in AI-based mobile robots using Isaac Sim, speeding up time-consuming steps for training expert agents.
</p>

## Features
<a name="Features"/>

#### Environments (scenes)
* Three different flat grids (normal, black and curved).
* A simple, tiny room with a table at the center.
* Four whorehouse of different sizes and obstacles.
* One floor of a hospital building.
* One floor of an office building.
* A custom random obstacle map.

<p align="center">
  <img src="images/scenes.gif" alt="scenes"/>
</p>

#### Robots
* Jetbot      (differential).
* Carter V1   (differential).
* Transporter (differential).
* Kaya        (holonomic).

<p align="center">
  <img src="images/robots.gif" alt="robots"/>
</p>

#### Sensors
* Wheel lineal velocity sensor (encoder).
* Robot’s base lineal velocity sensor (3d velocity magnitude).
* Robot’s base angular velocity (of the yaw angle).
* Customizable RGB camera
* Customizable depth camera
* Customizable Lidar (range sensor).

#### Others
* General methods to control and mesure the robot joints and sensors.
* Example.

## How to install
<a name="How-to-install"/>

#### Requirements
You must hace a computer compatible with Isaac Sim 2021.2.1, please check the [official documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html).

#### Steps
 1. Download this Git.
 2. Copy DRL_Isaac_lib to ~/.local/share/ov/pkg/isaac_sim-2021.2.1

## How to use
<a name="How-to-use"/>

#### Train
Open a terminal and run the following commands:
 1. cd ~/.local/share/ov/pkg/isaac_sim-2021.2.1/DRL_Isaac_lib/
 2. ~/.local/share/ov/pkg/isaac_sim-2021.2.1/python.sh train_d.py

#### Tensorboard view
Open a terminal and run:
* ~/.local/share/ov/pkg/isaac_sim-2021.2.1/python.sh ~/.local/share/ov/pkg/isaac_sim-2021.2.1/tensorboard --logdir ./

#### Real-time nvidia-smi
If you want to watch the GPU ussage in real time run in a terminal:
* watch -n0.1 nvidia-smi


## Example
<a name="Example"/>

## Notes
<a name="Notes"/>

<p align="justify">
  An easy-to-use library to develop deep reinforcement learning experiments in the Isaac Sim simulator. 
</p>
