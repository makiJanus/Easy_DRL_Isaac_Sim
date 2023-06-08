# cd ~/.local/share/ov/pkg/isaac_sim-2021.2.1/DRL_Isaac_lib/
# ~/.local/share/ov/pkg/isaac_sim-2021.2.1/python.sh train.py

import omni
#omni.timeline.get_timeline_interface().play()

#from omni.isaac.dynamic_control import _dynamic_control
import gym
from gym import spaces
import numpy as np
import math                 
import time


class Isaac_envs(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=3000,
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        self.dc            = omni.isaac.dynamic_control._dynamic_control.acquire_dynamic_control_interface()
        from isaac_robots  import isaac_robot
        from isaac_envs    import isaac_envs  
        from omni.isaac.core.objects import VisualCuboid

        env_name    = "random_walk" #random_walk
        robot_name  = "jetbot"
        action_type = "discrete"

        self._env_name    = env_name
        self._action_type = action_type
        self.episode_count = 0
        self.goal_count = 0
        self.step_count = 0
        self.distance_count = 0

        self.isaac_environments = isaac_envs(headless=self.headless)

        if env_name=="random_walk":
            self._my_world = self.isaac_environments.add_environment(env=env_name, name=robot_name)
            self.isaac_environments._generate_vertical_path_map(random=False)
            robot_pos, robot_ori = self.isaac_environments._robot_pose_random_walk(random=True)
            pos_target           = self.isaac_environments._target_pos_random_walk(random=True)
        else:
            self._my_world = self.isaac_environments.add_environment(env=env_name)
            robot_pos  = np.array([0, 0.0, 2.0])
            robot_ori  = np.array([1.0, 0.0, 0.0, 0.0])
            pos_target = np.array([60, 30, 1])


        self.robot = self._my_world.scene.add(
            isaac_robot(
                prim_path="/basic",
                name=robot_name,
                position=robot_pos,
                orientation=robot_ori,
            )
        )

        self.isaac_environments._set_camera(name=self.robot._name, prim_path=self.robot._prim_path, headless=self.headless)
        self.isaac_environments._set_lidar(name=self.robot._name, prim_path=self.robot._prim_path, headless=self.headless)

        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=pos_target,
                size=0.01,
                scale= np.array([5, 5, 5]),
                color=np.array([1.0, 0, 0]),
            )
        )
        self.seed(seed)
        self.sd_helper = None
        self.viewport_window = None
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)

        self.action_space = self.robot.get_action_space(type=action_type)
        self.movements    = self.robot.get_discrete_actions()

        self.state_pos_space = spaces.Box(
			low=np.array([0,-np.pi], dtype=np.float32),
			high=np.array([8,np.pi], dtype=np.float32),
			dtype=np.float32
		)

        self.state_vel_space = spaces.Box(
			low=np.array([0,-np.pi/4], dtype=np.float32),
			high=np.array([10,np.pi/4], dtype=np.float32),
			dtype=np.float32
        )
        
        self.state_IR_space = spaces.Box(low=0, high=1, shape=(1,12), dtype=np.float32)

        self.depth_obs_space = spaces.Box(low=0, high=1, shape=(1, 128, 128), dtype=np.uint8)
        self.rgb_obs_space   = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)

        self.observation_space = spaces.Dict(spaces={
				"IR_raleted" : self.state_IR_space, 
				"pos_raleted": self.state_pos_space,
                "vel_raleted": self.state_vel_space,
                #"rgb_raleted": self.state_h_space,
                #"depth_depth": self.cam_obs_space,
				}
		)
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        previous_jetbot_position, _ = self.robot.get_world_pose()

        ## Robot take action and run a simulation step
        for i in range(self._skip_frame):

            if self._action_type=="continuous":
                if self.robot._is_differential:
                    self.robot.differential_controller(np.array([action[0], action[1]]))
                else:
                    self.robot.holonomic_controller(np.array([action[0], action[1], action[2]]))
            
            elif self._action_type=="discrete":
                selected_action = self.movements[action]
                # print(selected_action)
                if self.robot._is_differential:
                    self.robot.differential_controller(np.array([selected_action[0], selected_action[1]]))
                else:
                    self.robot.holonomic_controller(np.array([selected_action[0], selected_action[1], selected_action[2]]))
            
            self._my_world.step(render=False)
        
        current_jetbot_position, _ = self.robot.get_world_pose()

        self.distance_count += np.linalg.norm(current_jetbot_position - previous_jetbot_position)
        
        ## Observation
        observations = self.get_observations()

        info = {}

        ## Reward function and end condition
        done = False
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
        
        goal_world_position, _ = self.goal.get_world_pose()
        current_dist_to_goal = self.robot.distance_to(goal_world_position)

        depth_points = self.isaac_environments._get_lidar_data()
        depth_points_min = np.amin(depth_points)
        
        step_conservation = 1 - (self._my_world.current_time_step_index/self._max_episode_length)# 1 + (self._my_world.current_time_step_index/self._max_episode_length)#
        landing_reward = 90
        distance_reward = (-current_dist_to_goal)/100 # previous_dist_to_goal - current_dist_to_goal -0.1

        if distance_reward > 0:
            reward = distance_reward*step_conservation
        else:
            reward = distance_reward

        #if depth_points_min <= 0.30 and depth_points_min > 0.15:
        #    #done = True
        #    reward -= (0.6-depth_points_min)

        if depth_points_min <= 0.155:
            done = True
            reward -= 10*(0.6-depth_points_min)
        
        if current_dist_to_goal <= 0.5:
            done = True
            self.goal_count += 1
            print("Arrived!")
            reward += 10 + landing_reward * (1 - (self._my_world.current_time_step_index/self._max_episode_length))

        self.step_count = self._my_world.current_time_step_index

        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        if self._env_name=="random_walk":
            # randomize goal location in circle around robot
            position = self.isaac_environments._target_pos_random_walk(random=True)
            self.goal.set_world_pose(position)
            # randomize robot pose
            robot_pos, robot_ori = self.isaac_environments._robot_pose_random_walk(random=True)
            self.robot.set_robot_pose(position=robot_pos, orientation=robot_ori)

        observations = self.get_observations()

        print("Hit rate: " + str(self.goal_count) +"/"+ str(self.episode_count))
        print("Steps on episode " + str(self.episode_count) + ": " + str(self.step_count))
        print("Seconds on episode " + str(self.episode_count) + ": " + str( (self.step_count)/60 ))
        print("Distance on episode " + str(self.episode_count) + ": " + str(self.distance_count))
        self.episode_count += 1
        self.distance_count = 0
        return observations

    def get_observations(self):
        ## Camera Data
        # rgb_data   = self.isaac_environments._get_cam_data(type="rgb")
        depth_data = self.isaac_environments._get_cam_data(type="depth")

        ## Lidar Data
        lidar_data = self.isaac_environments._get_lidar_data()
        # for transporter uncomment the next line
        # lidar_data2 = self.isaac_environments._get_lidar_data(lidar_selector=2)

        ## Distance and angular differencess
        goal_world_position, _ = self.goal.get_world_pose()
        d = self.robot.distance_to(goal_world_position)
        angle = self.robot.angular_difference_to(goal_world_position)
        target_relative_to_robot_data = np.array([d, angle])

        ## Robot base's velocities
        real_V = self.robot.get_lineal_vel_base()
        real_W = self.robot.get_angular_vel_base()

        vase_vel_data = np.array([ real_V, real_W])

        obs = {"IR_raleted" : lidar_data, "pos_raleted" : target_relative_to_robot_data, "vel_raleted" : vase_vel_data} 
        #obs = {"h_raleted" : h_state, "vel_raleted" : obs_state}

        return obs

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]