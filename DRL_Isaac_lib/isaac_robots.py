from typing import Optional, Tuple
import numpy as np
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import find_nucleus_server
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from omni.isaac.dynamic_control import _dynamic_control
from gym import spaces
import carb


class isaac_robot(Robot):
    """[summary]

        Args:
            stage (Usd.Stage): [description]
            prim_path (str): [description]
            name (str): [jetbot, carter_v1, kaya, transporter]
            usd_path (str, optional): [description]
            position (Optional[np.ndarray], optional): [description]. Defaults to None.
            orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        prim_path: str,
        name: str = "jetbot",
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._name = name
        self._prim_path = prim_path
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self._is_differential = True

        if name=="jetbot":
            usd_path = "/Isaac/Robots/Jetbot/jetbot.usd"
            self._is_differential = True
        elif name=="carter_v1":
            usd_path = "/Isaac/Robots/Carter/carter_v1.usd"
            self._is_differential = True
        elif name=="kaya":
            usd_path = "/Isaac/Robots/Kaya/kaya.usd"
            self._is_differential = False
        elif name=="transporter":
            usd_path = "/Isaac/Robots/Transporter/transporter_sensors.usd"
            self._is_differential = True
        elif name=="quadcopter":
            usd_path = "/Isaac/Robots/Quadcopter/quadcopter.usd"
        else:
            carb.log_error("Could not find robot :(")
        
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            """ result, nucleus_server = find_nucleus_server()
            if result is False:
                carb.log_error("Could not find nucleus server with /Isaac folder")
                return """
            self.nucleus_server = get_assets_root_path()
            if self.nucleus_server is None:
                carb.log_error("Could not find nucleus server with /Isaac folder")
                return
            asset_path = self.nucleus_server + usd_path #"/Isaac/Robots/Jetbot/jetbot.usd" "/Isaac/Robots/Carter/carter_v1.usd"
            prim.GetReferences().AddReference(asset_path)
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )

        if self._name == "jetbot" or self._name == "transporter":
            self._wheel_dof_names = ["left_wheel_joint", "right_wheel_joint"]
            self._wheel_dof_indices = (0, 1)
        elif self._name == "carter_v1":
            self._wheel_dof_names = ["left_wheel", "right_wheel"]
            self._wheel_dof_indices = (0, 1)
        # elif self._name == "carter_v2":
        #     self._wheel_dof_names = ["joint_wheel_left", "joint_wheel_right"]
        #     self._wheel_dof_indices = (5, 6)
        elif self._name == "kaya":
            self._wheel_dof_names = ["axle_2_joint", "axle_1_joint", "axle_0_joint"] # [left, right, back]
            self._wheel_dof_indices = (0, 1, 2)
        elif self._name == "quadcopter":
            self._wheel_dof_names = ["rotor_0", "rotor_1", "rotor_2"] # [left, right, back]
            self._wheel_dof_indices = (0, 1, 2)

        return
    

    def get_wheel_velocities(self) -> Tuple[float, float] or Tuple[float, float, float]:
        """[summary]

        Returns:
            Tuple[np.ndarray, np.ndarray]            : [jetbot, carter_v1, transporter]
            Tuple[np.ndarray, np.ndarray, np.ndarray]: [kaya]
        """
        
        joint_velocities = self.get_joint_velocities()

        if self._name=="jetbot" or self._name=="carter_v1" or self._name == "transporter":
            velocities = joint_velocities[self._wheel_dof_indices[0]], joint_velocities[self._wheel_dof_indices[1]]
        if self._name=="kaya":
            velocities = joint_velocities[self._wheel_dof_indices[0]], joint_velocities[self._wheel_dof_indices[1]], joint_velocities[self._wheel_dof_indices[2]]
        return velocities

    def set_wheel_velocities(self, velocities: Tuple) -> None:
        """[summary]

        Args:
            velocities (Tuple[float, float]): [description]
        """
        
        self.ar = self.dc.get_articulation(self._prim_path)
        if self._name == "kaya":
            self.wheel_left = self.dc.find_articulation_dof(self.ar , self._wheel_dof_names[0])
            self.wheel_right = self.dc.find_articulation_dof(self.ar, self._wheel_dof_names[1])
            self.wheel_back = self.dc.find_articulation_dof(self.ar , self._wheel_dof_names[2])
            self.dc.set_dof_velocity_target(self.wheel_left , velocities[0])
            self.dc.set_dof_velocity_target(self.wheel_right, velocities[1])
            self.dc.set_dof_velocity_target(self.wheel_back , velocities[3])
    
        else:
            self.wheel_left = self.dc.find_articulation_dof(self.ar, self._wheel_dof_names[0])
            self.wheel_right = self.dc.find_articulation_dof(self.ar, self._wheel_dof_names[1])
            self.dc.set_dof_velocity_target(self.wheel_left, velocities[0])
            self.dc.set_dof_velocity_target(self.wheel_right, velocities[1])
        
        return
    
    def differential_controller(self, velocities: np.array = [float, float]):
        """[summary]
        Controller for two wheeled differential robots
        Args:
            velocities: Tuple[float, float]: [Lineal velocity, Yaw velocity].

        Returns:
        """

        wheel_radius = 1
        wheel_base   = 5

        if self._name == "jetbot":
            wheel_radius = 3.25 
            wheel_base   = 11.25
        elif self._name == "carter_v1":
            wheel_radius = 24.5 
            wheel_base   = 49.7
        # elif self._name == "carter_v2":
        #     wheel_radius = 14 
        #     wheel_base   = 41.3
        elif self._name == "transporter":
            wheel_radius = 8.6
            wheel_base   = 16.95
        else:
            carb.log_error("Could not find the robot " + self._name + ", or is not compatible with differential controller, remember, just two wheeled robot allowed!")
            
        # velocities[0] = velocities[0]*10

        joint_velocities = [0.0, 0.0]
        joint_velocities[0] = ((2 * velocities[0]) - (velocities[1] * wheel_base)) / (2 * wheel_radius)
        joint_velocities[1] = ((2 * velocities[0]) + (velocities[1] * wheel_base)) / (2 * wheel_radius)

        self.set_wheel_velocities( (joint_velocities[0], joint_velocities[1]) )
        return
    
    def holonomic_controller(self, velocities: np.array = [float, float, float]):
        """[summary]
        Controller for three wheeled holonomic robots
        Args:
            Tuple[float, float, float]: [longitudinal_velocity, lateral_velocity, yaw_velocity].

        Returns:
        """
        
        if self._name=="kaya":
            wheel_radius = 4.0 
            wheel_base   = 12.5
        else: 
            carb.log_error("Could not find the robot " + self._name + ", or is not compatible with holonomic controller, remember, just three wheeled robot allowed! Gotcha boi?")
        
        wheel_speed = [0.0, 0.0, 0.0]

        # velocities[0] = velocities[0]*10
        # velocities[1] = velocities[1]*10

        wheel_speed[0] = -(-velocities[1] + velocities[2]*wheel_base)/wheel_radius
        wheel_speed[1] = -(velocities[1] - np.sqrt(3)*velocities[0] + 2*velocities[2]*wheel_base)/(2*wheel_radius)
        wheel_speed[2] = -(velocities[1] + np.sqrt(3)*velocities[0] + 2*velocities[2]*wheel_base)/(2*wheel_radius)

        ar = self.dc.get_articulation(self._prim_path)
        self.wheel_a = self.dc.find_articulation_dof(ar, self._wheel_dof_names[0])
        self.wheel_b = self.dc.find_articulation_dof(ar, self._wheel_dof_names[1])
        self.wheel_c = self.dc.find_articulation_dof(ar, self._wheel_dof_names[2])
        self.dc.set_dof_velocity_target(self.wheel_c, wheel_speed[0])
        self.dc.set_dof_velocity_target(self.wheel_a, wheel_speed[1])
        self.dc.set_dof_velocity_target(self.wheel_b, wheel_speed[2])

        return

    def get_lineal_vel_base(self):
        """[summary]
        Returns the linear velocity of the base of the robot.
        Args:

        Returns:
            lineal velocity: [float]
        """
        real_V = np.sqrt( self.get_linear_velocity()[0]**2 + self.get_linear_velocity()[1]**2)
        return real_V

    def get_angular_vel_base(self):
        """[summary]
        Returns the angular velocity (yaw) of the base of the robot.
        Args:

        Returns:
            angular velocity: [float]
        """
        real_W = self.get_angular_velocity()[2]
        return real_W
    
    def distance_to(self, target_pos: Tuple[float, float, float]):
        """[summary]
        Measures the distance from the base of the robot to an object in centimeters
        Args:

        Returns:
            angular velocity: [float]
        """
        current_position, _ = self.get_world_pose()
        d = np.sqrt( ((current_position[0] - target_pos[0]) ** 2) + ((current_position[1] - target_pos[1]) ** 2) + ((current_position[2] - target_pos[2]) ** 2) )
        return d

    def angular_difference_to(self, target_pos: Tuple[float, float, float]):
        """[summary]
        Measures the angular difference between the unitary vector headed to the front of the the robot and
        the vector formeb by the robot and an object in radiands, all in the XY plane.
        Args:

        Returns:
            angular velocity: [float]
        """
        current_position, current_orientation = self.get_world_pose()

        qx = current_orientation[0]
        qy = current_orientation[1]
        qz = current_orientation[2]
        qw = current_orientation[3]

        y = (target_pos[1] - current_position[1])
        x = (target_pos[0] - current_position[0])
        
        theta = np.arctan2(2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qw * qw + qz * qz))
        alpha = np.arctan2(y, x)

        if alpha < 0:
            alpha += 2.0*np.pi
        
        angle = np.arctan2(np.sin(theta-alpha), np.cos(alpha-theta))
        return angle

    def set_robot_pose(self, position: np.array = [float, float, float], orientation: np.array = [float, float, float, float]):
        if self._name=="jetbot":
            chassis_path = self._prim_path+"/chassis"
        
        elif self._name=="carter_v1":
            chassis_path = self._prim_path+"/chassis_link"
        
        elif self._name=="transporter":
            chassis_path = self._prim_path+"/chassis"
        
        elif self._name=="kaya":
            chassis_path = self._prim_path+"/base_link"
        
        else:
            carb.log_error("Could not find the selected sensor, maybe there is a lidar already in this robot :p")
            return
        
        robot = self.dc.get_rigid_body(chassis_path)
        new_pose = _dynamic_control.Transform( position, orientation)
        self.dc.set_rigid_body_pose(robot, new_pose)
        return

    def get_action_space(self, type: str = "continuous"):
        action_space = None

        if type=="continuous":
            if self._name=="jetbot":
                action_space = spaces.Box(
                    low=np.array([0,-np.pi/4], dtype=np.float32),
			        high=np.array([10,np.pi/4], dtype=np.float32),
			        dtype=np.float32)
        
            elif self._name=="carter_v1":
                action_space = spaces.Box(
                    low=np.array([0,-np.pi/2], dtype=np.float32),
			        high=np.array([10,np.pi/2], dtype=np.float32),
			        dtype=np.float32)
        
            elif self._name=="transporter":
                action_space = spaces.Box(
                    low=np.array([0,-np.pi/2], dtype=np.float32),
			        high=np.array([10,np.pi/2], dtype=np.float32),
			        dtype=np.float32)
        
            elif self._name=="kaya":
                action_space = spaces.Box(
                    low=np.array([-10,-10,-np.pi/4], dtype=np.float32),
			        high=np.array([10, 10, np.pi/4], dtype=np.float32),
			        dtype=np.float32)

        elif type=="discrete":
            action_space = spaces.Discrete(3)

        return action_space

    def get_discrete_actions(self):
        movements = None
        if self._name=="jetbot":
            movements = np.array([[15, 0.785], [25, 0], [15, -0.785]])

        elif self._name=="carter_v1":
            movements = np.array([[5, 1.571], [8, 0], [5, -1.571]])
        
        elif self._name=="transporter":
            movements = np.array([[5, 1.571], [8, 0], [5, -1.571]])
        
        elif self._name=="kaya":
            movements = np.array([[3, 0, 0.785], [5, 0, 0], [3, 0, -0.785]])
        
        return movements

    