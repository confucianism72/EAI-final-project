import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs import Actor
from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent
from scripts.so101 import SO101


# Grid boundaries in meters (from Track1_Simulation_Parameters.md)
# Left Grid: X = 5.1cm to 21.7cm, Y = 17.8cm to 34.2cm
LEFT_GRID = {"x_min": 0.051, "x_max": 0.217, "y_min": 0.178, "y_max": 0.342}
# Mid Grid: X = 23.8cm to 39.4cm, Y = 17.8cm to 34.2cm  
MID_GRID = {"x_min": 0.238, "x_max": 0.394, "y_min": 0.178, "y_max": 0.342}
# Right Grid: X = 41.4cm to 58.0cm, Y = 17.8cm to 34.2cm
RIGHT_GRID = {"x_min": 0.414, "x_max": 0.580, "y_min": 0.178, "y_max": 0.342}


@register_env("Track1-v0", max_episode_steps=200)
class Track1Env(BaseEnv):
    SUPPORTED_ROBOTS = ["so101", ("so101", "so101")]
    agent: SO101

    def __init__(
        self, 
        *args, 
        robot_uids=("so101", "so101"),
        task: str = "lift",  # "lift", "stack", "sort"
        domain_randomization: bool = True,
        **kwargs
    ):
        self.task = task
        self.domain_randomization = domain_randomization
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # Front Camera with optional per-reconfigure pose randomization
        base_pos = [0.316, 0.260, 0.407]
        
        if self.domain_randomization and hasattr(self, 'num_envs') and self.num_envs > 1:
            # Randomize camera pose: ±2.5cm position, ±7.5° rotation
            pose = sapien_utils.look_at(eye=base_pos, target=[0.316, 0.260, 0.0])
            pose = Pose.create(pose)
            pose = pose * Pose.create_from_pq(
                p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
                q=randomization.random_quaternions(
                    n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
                ),
            )
        else:
            pose = sapien.Pose(p=base_pos, q=[1, 0, 0, 0])
        
        return [
            CameraConfig(
                "front_camera",
                pose=pose,
                width=640,
                height=480,
                fov=np.deg2rad(73.63),
                near=0.01,
                far=100,
            ),
        ]

    def _setup_sensors(self, options: dict):
        """Override to add wrist cameras after agents are loaded."""
        super()._setup_sensors(options)
        
        from mani_skill.sensors.camera import Camera
        wrist_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        
        for i, agent in enumerate(self.agent.agents):
            camera_link = agent.robot.links_map.get("camera_link", None)
            if camera_link is not None:
                uid = f"wrist_camera_{i}"
                config = CameraConfig(
                    uid,
                    pose=wrist_pose,
                    width=640,
                    height=480,
                    fov=np.deg2rad(50),
                    near=0.01,
                    far=100,
                    mount=camera_link,
                )
                self._sensors[uid] = Camera(config, self.scene)

    def _load_lighting(self, options: dict):
        """Load lighting with optional randomization."""
        for i, scene in enumerate(self.scene.sub_scenes):
            if self.domain_randomization:
                # Randomize ambient light intensity
                ambient = np.random.uniform(0.2, 0.5, size=3).tolist()
            else:
                ambient = [0.3, 0.3, 0.3]
            scene.ambient_light = ambient
            scene.add_directional_light([0.5, 0, -1], [3.0, 3.0, 3.0], shadow=True, shadow_scale=5, shadow_map_size=2048)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_scene(self, options: dict):
        # Ground
        for scene in self.scene.sub_scenes:
            ground_material = scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
            scene.add_ground(0, material=ground_material)
        
        # Table surface with optional color randomization
        self._build_table()
        self._build_tape_lines()
        self._build_robot_base_visuals()
        
        # Load task objects
        self._load_objects(options)

    def _build_table(self):
        """Build table with optional visual randomization."""
        if self.domain_randomization:
            tables = []
            for i in range(self.num_envs):
                builder = self.scene.create_actor_builder()
                # Randomize table color slightly
                color = [0.9 + np.random.uniform(-0.05, 0.05)] * 3 + [1]
                builder.add_box_visual(
                    half_size=[0.3, 0.3, 0.01], 
                    material=sapien.render.RenderMaterial(base_color=color)
                )
                builder.add_box_collision(half_size=[0.3, 0.3, 0.01])
                builder.initial_pose = sapien.Pose(p=[0.3, 0.3, -0.01])
                builder.set_scene_idxs([i])
                table = builder.build_static(name=f"table_{i}")
                self.scene.remove_from_state_dict_registry(table)
                tables.append(table)
            self.table = Actor.merge(tables, name="table")
            self.scene.add_to_state_dict_registry(self.table)
        else:
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(half_size=[0.3, 0.3, 0.01], material=[0.9, 0.9, 0.9])
            builder.add_box_collision(half_size=[0.3, 0.3, 0.01])
            builder.initial_pose = sapien.Pose(p=[0.3, 0.3, -0.01])
            self.table = builder.build_static(name="table")

    def _build_tape_lines(self):
        """Build black tape lines for grid visualization."""
        tape_material = [0, 0, 0]
        tape_half_width = 0.009
        tape_height = 0.001
        
        tape_specs = [
            # Horizontal lines
            {"half_size": [0.27, tape_half_width, tape_height], "pos": [0.316, 0.178, 0.001], "name": "tape_bottom"},
            {"half_size": [0.27, tape_half_width, tape_height], "pos": [0.316, 0.342, 0.001], "name": "tape_top"},
            # Vertical lines
            {"half_size": [tape_half_width, 0.082, tape_height], "pos": [0.051, 0.26, 0.001], "name": "tape_left_1"},
            {"half_size": [tape_half_width, 0.082, tape_height], "pos": [0.217, 0.26, 0.001], "name": "tape_left_2"},
            {"half_size": [tape_half_width, 0.082, tape_height], "pos": [0.238, 0.26, 0.001], "name": "tape_mid_1"},
            {"half_size": [tape_half_width, 0.082, tape_height], "pos": [0.394, 0.26, 0.001], "name": "tape_mid_2"},
            {"half_size": [tape_half_width, 0.082, tape_height], "pos": [0.414, 0.26, 0.001], "name": "tape_right_1"},
            {"half_size": [tape_half_width, 0.082, tape_height], "pos": [0.580, 0.26, 0.001], "name": "tape_right_2"},
        ]
        
        for spec in tape_specs:
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(half_size=spec["half_size"], material=tape_material)
            builder.initial_pose = sapien.Pose(p=spec["pos"])
            builder.build_static(name=spec["name"])

    def _build_robot_base_visuals(self):
        """Build visual representations of robot bases."""
        base_material = [0.2, 0.2, 0.2]
        base_specs = [
            {"pos": [0.119, 0.10, 0.005], "name": "left_base_visual"},
            {"pos": [0.433, 0.10, 0.005], "name": "right_base_visual"},
        ]
        
        for spec in base_specs:
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(half_size=[0.055, 0.055, 0.005], material=base_material)
            builder.initial_pose = sapien.Pose(p=spec["pos"])
            builder.build_static(name=spec["name"])

    def _load_agent(self, options: dict):
        agent_poses = [
            sapien.Pose(p=[0.119, 0.10, 0]),  # Left Robot
            sapien.Pose(p=[0.433, 0.10, 0])   # Right Robot
        ]
        
        # Enable per-env building for joint randomization
        if self.domain_randomization:
            super()._load_agent(options, agent_poses, build_separate=True)
            self._randomize_robot_properties()
        else:
            super()._load_agent(options, agent_poses)

    def _randomize_robot_properties(self):
        """Randomize robot joint friction and damping for domain randomization."""
        for agent in self.agent.agents:
            for joint in agent.robot.joints:
                for i, obj in enumerate(joint._objs):
                    # Randomize joint properties
                    stiffness = np.random.uniform(800, 1200)
                    damping = np.random.uniform(80, 120)
                    obj.set_drive_properties(stiffness=stiffness, damping=damping, force_limit=100)
                    obj.set_friction(friction=np.random.uniform(0.3, 0.7))

    def _load_objects(self, options: dict):
        """Load task-specific objects."""
        # Red cube is always 3cm
        self.red_cube = self._build_cube(
            name="red_cube",
            half_size=0.015,
            base_color=[1, 0, 0, 1],
            default_pos=[0.497, 0.26, 0.015]
        )
        
        # Green cube size depends on task
        if self.task == "sort":
            # Sort task: green cube is 1cm
            green_half_size = 0.005
            green_z = 0.005
        else:
            # Lift/Stack: green cube is 3cm (only used in Stack)
            green_half_size = 0.015
            green_z = 0.015
        
        self.green_cube = self._build_cube(
            name="green_cube",
            half_size=green_half_size,
            base_color=[0, 1, 0, 1],
            default_pos=[0.497, 0.30, green_z]
        )

    def _build_cube(self, name: str, half_size: float, base_color: list, default_pos: list) -> Actor:
        """Build a cube with optional domain randomization."""
        if self.domain_randomization:
            cubes = []
            for i in range(self.num_envs):
                builder = self.scene.create_actor_builder()
                
                # Randomize color slightly
                color = [
                    base_color[0] + np.random.uniform(-0.1, 0.1),
                    base_color[1] + np.random.uniform(-0.1, 0.1),
                    base_color[2] + np.random.uniform(-0.1, 0.1),
                    1
                ]
                color = [max(0, min(1, c)) for c in color]
                
                builder.add_box_collision(half_size=[half_size] * 3)
                builder.add_box_visual(
                    half_size=[half_size] * 3,
                    material=sapien.render.RenderMaterial(base_color=color)
                )
                builder.initial_pose = sapien.Pose(p=default_pos)
                builder.set_scene_idxs([i])
                cube = builder.build(name=f"{name}_{i}")
                self.scene.remove_from_state_dict_registry(cube)
                cubes.append(cube)
            
            merged = Actor.merge(cubes, name=name)
            self.scene.add_to_state_dict_registry(merged)
            
            # Randomize physical properties
            self._randomize_cube_physics(merged)
            return merged
        else:
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[half_size] * 3)
            builder.add_box_visual(half_size=[half_size] * 3, material=base_color[:3])
            builder.initial_pose = sapien.Pose(p=default_pos)
            return builder.build(name=name)

    def _randomize_cube_physics(self, cube: Actor):
        """Randomize cube mass and friction."""
        for i, obj in enumerate(cube._objs):
            rigid_body: PhysxRigidBodyComponent = obj.find_component_by_type(PhysxRigidBodyComponent)
            if rigid_body is not None:
                rigid_body.mass = np.random.uniform(0.05, 0.2)
                for shape in rigid_body.collision_shapes:
                    shape.physical_material.dynamic_friction = np.random.uniform(0.2, 0.5)
                    shape.physical_material.static_friction = np.random.uniform(0.2, 0.5)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized object positions based on task."""
        with torch.device(self.device):
            b = len(env_idx)
            
            if self.task == "lift":
                # Red cube random in Right Grid
                red_pos = self._random_grid_position(b, RIGHT_GRID, z=0.015)
                self.red_cube.set_pose(Pose.create_from_pq(p=red_pos))
                # Hide green cube (move far away)
                self.green_cube.set_pose(Pose.create_from_pq(p=torch.tensor([[-10, -10, -10]] * b, device=self.device)))
                
            elif self.task == "stack":
                # Both cubes in Right Grid, non-overlapping
                red_pos = self._random_grid_position(b, RIGHT_GRID, z=0.015)
                green_pos = self._random_grid_position(b, RIGHT_GRID, z=0.015)
                # Ensure minimum distance
                for _ in range(10):  # Try a few times to separate them
                    dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
                    overlap = dist < 0.05  # 5cm minimum
                    if not overlap.any():
                        break
                    green_pos[overlap] = self._random_grid_position(overlap.sum().item(), RIGHT_GRID, z=0.015)
                
                self.red_cube.set_pose(Pose.create_from_pq(p=red_pos))
                self.green_cube.set_pose(Pose.create_from_pq(p=green_pos))
                
            elif self.task == "sort":
                # Both cubes in Mid Grid
                red_pos = self._random_grid_position(b, MID_GRID, z=0.015)
                green_pos = self._random_grid_position(b, MID_GRID, z=0.005)  # Smaller green cube
                self.red_cube.set_pose(Pose.create_from_pq(p=red_pos))
                self.green_cube.set_pose(Pose.create_from_pq(p=green_pos))

    def _random_grid_position(self, batch_size: int, grid: dict, z: float) -> torch.Tensor:
        """Generate random positions within a grid."""
        x = torch.rand(batch_size, device=self.device) * (grid["x_max"] - grid["x_min"]) + grid["x_min"]
        y = torch.rand(batch_size, device=self.device) * (grid["y_max"] - grid["y_min"]) + grid["y_min"]
        z_tensor = torch.full((batch_size,), z, device=self.device)
        return torch.stack([x, y, z_tensor], dim=1)

    def evaluate(self):
        """Evaluate success based on task."""
        if self.task == "lift":
            return self._evaluate_lift()
        elif self.task == "stack":
            return self._evaluate_stack()
        elif self.task == "sort":
            return self._evaluate_sort()
        return {}

    def _evaluate_lift(self):
        """Lift: red cube >= 5cm above table."""
        red_z = self.red_cube.pose.p[:, 2]
        success = red_z >= 0.05
        return {"success": success, "red_height": red_z}

    def _evaluate_stack(self):
        """Stack: red cube on top of green cube, stable."""
        red_pos = self.red_cube.pose.p
        green_pos = self.green_cube.pose.p
        
        # Check if red is above green (z difference ~ 3cm = cube size)
        z_diff = red_pos[:, 2] - green_pos[:, 2]
        z_ok = (z_diff > 0.02) & (z_diff < 0.05)
        
        # Check xy alignment (within 1.5cm)
        xy_dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
        xy_ok = xy_dist < 0.015
        
        success = z_ok & xy_ok
        return {"success": success, "z_diff": z_diff, "xy_dist": xy_dist}

    def _evaluate_sort(self):
        """Sort: green in Left Grid, red in Right Grid."""
        red_pos = self.red_cube.pose.p
        green_pos = self.green_cube.pose.p
        
        # Check red in Right Grid
        red_in_right = (
            (red_pos[:, 0] >= RIGHT_GRID["x_min"]) & 
            (red_pos[:, 0] <= RIGHT_GRID["x_max"]) &
            (red_pos[:, 1] >= RIGHT_GRID["y_min"]) & 
            (red_pos[:, 1] <= RIGHT_GRID["y_max"])
        )
        
        # Check green in Left Grid
        green_in_left = (
            (green_pos[:, 0] >= LEFT_GRID["x_min"]) & 
            (green_pos[:, 0] <= LEFT_GRID["x_max"]) &
            (green_pos[:, 1] >= LEFT_GRID["y_min"]) & 
            (green_pos[:, 1] <= LEFT_GRID["y_max"])
        )
        
        success = red_in_right & green_in_left
        return {"success": success, "red_in_right": red_in_right, "green_in_left": green_in_left}

    def _get_obs_extra(self, info: dict):
        """Return extra observations."""
        return {
            "red_cube_pos": self.red_cube.pose.p,
            "green_cube_pos": self.green_cube.pose.p,
        }
