import carb
import omni
import numpy as np
from typing import Optional, List
from omni.isaac.core.objects import FixedCuboid

class isaac_envs():

    def __init__(
        self,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        headless=True
    
    ) -> None:
        from omni.isaac.core import World
        from omni.isaac.core.utils.nucleus import find_nucleus_server
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.range_sensor import _range_sensor

        # Basic world prims
        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1)
        #result, self.nucleus_server = find_nucleus_server('/Isaac')
        self.nucleus_server = get_assets_root_path()
        if self.nucleus_server is None:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return
        
        self._headless = headless
        self.sd_helper = None
        self._lidar_path  = ""
        self._lidar_path2 = ""
        #self._map_dimension = 5
        #self._map_dist_unit = 50

        self.stage = omni.usd.get_context().get_stage()
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()

        return
    
    
    def add_environment(self, env="grid_default", name: Optional[str] = "jetbot"):
        """[summary]
        Available environments:
         grid_default, grid_black, grid_curved, simple_room, warehause_small_A, warehause_small_B, warehause_small_C, warehause_full,
         hospital, office, random_walk

        Returns:
            World
        """
        from omni.isaac.core.utils.stage import add_reference_to_stage

        asset_path = ""
        if env=="grid_default":
            asset_path = self.nucleus_server + "/Isaac/Environments/Grid/default_environment.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        elif env=="grid_black":
            asset_path = self.nucleus_server + "/Isaac/Environments/Grid/gridroom_black.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        elif env=="grid_curved":
            asset_path = self.nucleus_server + "/Isaac/Environments/Grid/gridroom_curved.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        elif env=="simple_room":
            asset_path = self.nucleus_server + "/Isaac/Environments/Simple_Room/simple_room.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        elif env=="warehause_small_A":
            asset_path = self.nucleus_server + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        elif env=="warehause_small_B":
            asset_path = self.nucleus_server + "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        elif env=="warehause_small_C":
            asset_path = self.nucleus_server + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        elif env=="warehause_full":
            asset_path = self.nucleus_server + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        elif env=="hospital":
            asset_path = self.nucleus_server + "/Isaac/Environments/Hospital/hospital.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        elif env=="office":
            asset_path = self.nucleus_server + "/Isaac/Environments/Office/office.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        elif env=="random_walk":
            asset_path = self.nucleus_server + "/Isaac/Environments/Grid/default_environment.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World")
            self._set_random_walk_env(name=name)
        else:
            carb.log_error("Could not find the selected environment")
        
        return self._my_world
    
    ## Set sensor's configurations

    def _set_camera(self, name: str, prim_path: str, headless: bool, size: Optional[np.ndarray] = np.array( [128, 128]) ):
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        from pxr import Gf, UsdGeom
        
        if name=="jetbot":
            camera_path = prim_path+"/chassis/rgb_camera/jetbot_camera"
        elif name=="carter_v1":
            camera_path= prim_path+"/chassis_link/camera_mount/carter_camera_first_person"
        elif name=="kaya":
            camera_path = prim_path+"/base_link/camera"
            camera_prim = self.stage.DefinePrim(camera_path, "Camera")
        elif name=="transporter":
            camera_path = prim_path+"/camera_mount/transporter_camera_first_person"

        if headless:
            #viewport_handle = omni.kit.viewport.get_viewport_interface()
            viewport_handle = omni.kit.viewport.utility.get_active_viewport() #changed

            viewport_handle.get_viewport_window().set_active_camera(str(camera_path))
            viewport_window = viewport_handle.get_viewport_window()
            self.viewport_window = viewport_window
            viewport_window.set_texture_resolution(size[0], size[1])
            
            if name=="kaya":
                xform = UsdGeom.Xformable(camera_prim)
                transform = xform.AddTransformOp()
                mat = Gf.Matrix4d()
                mat.SetTranslateOnly(Gf.Vec3d(0.0, 11.5, 8.0))
                mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(1,0,0), 70))
                transform.Set(mat)
            
        else:
            #viewport_handle = omni.kit.viewport.get_viewport_interface().create_instance()
            #new_viewport_name = omni.kit.viewport.get_viewport_interface().get_viewport_window_name(viewport_handle)
            #viewport_window = omni.kit.viewport.get_viewport_interface().get_viewport_window(viewport_handle)
            viewport_handle = omni.kit.viewport.utility.get_active_viewport()
            new_viewport_name = omni.kit.viewport.utility.get_active_viewport_camera_string(viewport_handle)
            viewport_window = omni.kit.viewport.utility.get_active_viewport_window(new_viewport_name)

            #viewport_window.set_active_camera(camera_path)
            viewport_window.camera_path = camera_path
            #viewport_window.set_texture_resolution(size[0], size[1])
            viewport_window.resolution = (size[0], size[1])
            
            if name=="kaya":
                xform = UsdGeom.Xformable(camera_prim)
                transform = xform.AddTransformOp()
                mat = Gf.Matrix4d()
                mat.SetTranslateOnly(Gf.Vec3d(0.0, 11.5, 8.0))
                mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(1,0,0), 70))
                transform.Set(mat)
            
            #viewport_window.set_window_size(420, 420)
            #self.viewport_window = viewport_window
            self.viewport_window = viewport_handle
        
        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["depth", "rgb"], viewport_api=self.viewport_window)
        self._my_world.render()
        self.sd_helper.get_groundtruth(["depth", "rgb"], self.viewport_window)
        return
    
    def _set_lidar(self, name: str, prim_path: str, headless: bool, number_lasers: Optional[int] = 12):        
        lidarPath = ""
        parent    = ""
        min_range = 0.1
        max_range = 1
        number_lasers = number_lasers
        
        self.number_lasers = number_lasers

        if name=="jetbot":
            lidarPath = "/lidar"
            parent    = prim_path+"/chassis"
            self._lidar_path = parent+lidarPath
            min_range = 0.15
            max_range = 1
        
        elif name=="carter_v1":
            lidarPath = "/carter_lidar"
            parent    = prim_path+"/chassis_link"
            self._lidar_path = parent+lidarPath
            return
        
        elif name=="transporter":
            lidarPath        = "/lidar"
            lidarPath2       = "/lidar_2"
            parent           = prim_path+"/chassis"
            self._lidar_path = parent+lidarPath
            self._lidar_path = parent+lidarPath2
            return
        
        elif name=="kaya":
            lidarPath = "/lidar"
            parent    = prim_path+"/base_link"
            self._lidar_path = parent+lidarPath
            min_range = 0.15
            max_range = 1
        
        else:
            carb.log_error("Could not find the selected sensor, maybe there is a lidar already in this robot")
            return

        if headless:
            draw_lines = False
        else:
            draw_lines = True
        
        result, prim = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=lidarPath,
            parent=parent,
            min_range=min_range,
            max_range=max_range,
            draw_points=False,
            draw_lines=draw_lines,
            horizontal_fov=360.0,
            vertical_fov=30.0,
            horizontal_resolution=(360/number_lasers),
            vertical_resolution=4.0,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False
        )
        return
    
    ## Return sensor's data
    
    def _get_cam_data(self, type: str = "rgb"):
        self._my_world.render()
        if type=="depth":
             gt = self.sd_helper.get_groundtruth(["depth"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0)
             #img = gt["depth"][:, :, :1]
             img = gt["depth"][:, :]
        else:
            gt = self.sd_helper.get_groundtruth(["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0)
            img = gt["rgb"][:, :, :3]
        return img
    
    def _get_lidar_data(self, lidar_selector: Optional[int] = 1):

        if lidar_selector==1:
            depth_points = self.lidarInterface.get_linear_depth_data(self._lidar_path)
            depth_points = np.resize(depth_points, (1,self.number_lasers))
        if lidar_selector==2:
            depth_points = self.lidarInterface.get_linear_depth_data(self._lidar_path2)
            depth_points = np.resize(depth_points, (1,self.number_lasers))
        return depth_points
    

    ## Random walk env

    def _set_random_walk_env(self, name: str = "jetbot", map_dist_unit=50, map_dimension=5, height=20):
        from pxr import UsdGeom

        if name=="jetbot" or name=="kaya":
            map_dist_unit=0.5#50
            height = 0.2#20

        if name=="carter_v1":
            map_dist_unit=1.1
            height = 0.7
        
        if name=="transporter":
            map_dist_unit=1.3
            height = 0.4
        
        self._map_dimension = map_dimension
        self._map_dist_unit = map_dist_unit
        # Walls
        self.outer_wall_1 = self._my_world.scene.add(
            FixedCuboid(
                prim_path="/outer_wall_1",
                name="wall_1",
                position=np.array([map_dist_unit*(map_dimension-1)/2, -map_dist_unit*(map_dimension-2), height/2]),
                size=1,
                scale=np.array([map_dist_unit*(map_dimension+2), map_dist_unit, height]), #size
                color=np.array([0.3, 0.5, 0.3]),
            )
        )
        self.outer_wall_2 = self._my_world.scene.add(
            FixedCuboid(
                prim_path="/outer_wall_2",
                name="wall_2",
                position=np.array([-map_dist_unit*2, map_dist_unit*(map_dimension-1)/2, height/2]),
                size=1,
                scale=np.array([map_dist_unit, map_dist_unit*(map_dimension+2+4), height]), #size
                color=np.array([0.3, 0.5, 0.3]),
            )
        )
        self.outer_wall_3 = self._my_world.scene.add(
            FixedCuboid(
                prim_path="/outer_wall_3",
                name="wall_3",
                position=np.array([map_dist_unit*(map_dimension+1), map_dist_unit*(map_dimension-1)/2, height/2]),
                size=1,
                scale=np.array([map_dist_unit, map_dist_unit*(map_dimension+2+4), height]), #size
                color=np.array([0.3, 0.5, 0.3]), 
            )
        )
        self.outer_wall_4 = self._my_world.scene.add(
            FixedCuboid(
                prim_path="/outer_wall_4",
                name="wall_4",
                position=np.array([map_dist_unit*(map_dimension-1)/2, map_dist_unit*(map_dimension+2), height/2]),
                size=1,
                scale=np.array([map_dist_unit*(map_dimension+2), map_dist_unit, height]), #size
                color=np.array([0.3, 0.5, 0.30]),
            )
        )

        # Cubes for random map generation
        self.stage = omni.usd.get_context().get_stage()

        self.cuboid_T = {}
        i = 0
        for row in range(map_dimension):
            for column in range(map_dimension):
                self._my_world.scene.add(
                    FixedCuboid(
                        prim_path="/wall_cube_"+str(row)+str(column),
                        name="cube_"+str(row)+str(column),
                        position=np.array([row*map_dist_unit, column*map_dist_unit, height/2]),
                        size=1,
                        scale=np.array([map_dist_unit, map_dist_unit, height]), #size
                        color=np.array([0.3, 0.3, 0.3]),))
                cube_row_column = self.stage.GetPrimAtPath("/wall_cube_"+str(row)+str(column))
                xform = UsdGeom.Xformable(cube_row_column)
                self.cuboid_T["wall_cube{0}".format(i)] =  xform.AddTransformOp()
                i += 1
            i += 1
        return

    def _generate_map(self, max_n_tunel = 40, max_length_tunel = 3, random = True):
        from pxr import Gf
        # Reference: https://www.freecodecamp.org/news/how-to-make-your-own-procedural-dungeon-map-generator-using-the-random-walk-algorithm-e0085c8aa9a/
        map = np.ones((self._map_dimension, self._map_dimension))
        pos = np.random.randint(10, size=2)
        #map[pos[0], pos[1]] = 0
        n_tunel = 0
        length_tunel = 0
        direction = [[-1, 0],
                     [1 , 0], 
                     [0 ,-1], 
                     [0 , 1]]
        

        while n_tunel < max_n_tunel and random:
            length_tunel = np.random.randint(max_length_tunel)
            idx          = np.random.randint(4)
            for i in range (length_tunel):
                pos = pos + direction[idx]
                if pos[0] < 0:
                    pos[0] = 0
  
                if pos[0] >= self._map_dimension:
                    pos[0] = self._map_dimension-1

                if pos[1] < 0:
                    pos[1] = 0
        
                if pos[1] >= self._map_dimension:
                    pos[1] = self._map_dimension-1
        
                map[pos[0], pos[1]] = 0
            n_tunel = n_tunel + 1
        
        i = 0

        if not random:
            map = np.array([[1., 0., 1., 0., 0.], 
                            [0., 0., 0., 0., 0.], 
                            [0., 0., 0., 0., 0.], 
                            [0., 0., 1., 0., 1.], 
                            [1., 0., 1., 0., 1.] ])

            map = np.rot90(map, axes=(0,1))


        for row in range(self._map_dimension):
            for column in range(self._map_dimension):
                mat = Gf.Matrix4d()
                if map[row][column] == 1:
                    mat.SetTranslateOnly(Gf.Vec3d(0, 0, 0))
                else:
                    mat.SetTranslateOnly(Gf.Vec3d(0, 0, -2))
                mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0,1,0), 0))
                self.cuboid_T["wall_cube{0}".format(i)].Set(mat)
                i += 1
            i += 1
        
        return map

    def _generate_vertical_path_map(self, random = True):
        mapita_aux = np.zeros((2, self._map_dimension))
        mapita_core = np.rot90(self._generate_map(random=random), axes=(1,0))
        no_path = True
        flag = False

        while no_path and random:
            for row in range (self._map_dimension):
                if np.sum(mapita_core[row,:]) >= 4:
                    flag = True
            if flag == True:
                flag = False
                mapita_core = np.rot90(self._generate_map(), axes=(1,0))
            else:
                no_path = False
        
        
        map = np.concatenate((mapita_aux, mapita_core, mapita_aux), axis=0)
        return map
    
    def _robot_pose_random_walk(self,random: bool = False):
        if random:
            position = np.array([self._map_dist_unit * np.random.randint(self._map_dimension), self._map_dist_unit * (-1.5), 0])
            # randomize robot orientation expressed in quaternions with cosine-sine trick to constraint rotation in one axis and make the final vector's magnitude equal to 1
            # ref: https://eater.net/quaternions/video/intro
            a  = np.random.rand()*np.pi # junt pi instead of 2*pi 'cause the angle in doubled in quaternion application
            q1 = np.sin(a)*0 # x axis?
            q2 = np.sin(a)*0 # y axis?
            q3 = np.sin(a)   # z axis?
            q4 = np.cos(a)   # 4th dim axis?
            orientation = np.array([q1, q2, q3, q4])
        else:
            position=np.array([self._map_dist_unit * (self._map_dimension-1)/2, self._map_dist_unit * (-1.5), 0])
            orientation = np.array([1.0, 0.0, 0.0, 0.0])
        return position, orientation

    def _target_pos_random_walk(self, random: bool = False):
        if random:
            position=np.array([self._map_dist_unit * np.random.randint(self._map_dimension), self._map_dist_unit * (5.5), 0.25])
        else:
            position=np.array([self._map_dist_unit * (self._map_dimension-1)/2, self._map_dist_unit * (5.5), 0.25])
        return position