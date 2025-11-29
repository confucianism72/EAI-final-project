import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose
from scripts.so101 import SO101

@register_env("Track1-v0", max_episode_steps=200)
class Track1Env(BaseEnv):
    SUPPORTED_ROBOTS = ["so101"]
    agent: SO101

    def __init__(self, *args, robot_uids="so101", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # Front Camera Parameters from Track1_Simulation_Parameters.md
        # Position: [0.316, 0.260, 0.407]
        # Orientation: Top-down view.
        # Identity quaternion [1, 0, 0, 0] in Sapien means:
        # - Camera looks down (-Z world)
        # - Camera Up is Forward (+Y world)
        # - Camera Right is Right (+X world)
        pose = sapien.Pose(p=[0.316, 0.260, 0.407], q=[1, 0, 0, 0])
        
        return [
            CameraConfig(
                "front_camera",
                pose=pose,
                width=640,
                height=480,
                fov=np.deg2rad(73.63), # Vertical FOV from front_camera.py
                near=0.01,
                far=100,
            )
        ]

    def _load_scene(self, options: dict):
        # 1. Ground
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([0.5, 0, -1], [3.0, 3.0, 3.0], shadow=True)
        
        # Access the underlying sapien scene to add ground
        # Note: This assumes single scene or identical setup for all
        for scene in self.scene.sub_scenes:
            ground_material = scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
            scene.add_ground(0, material=ground_material)
        
        # Visual ground (Table surface)
        # We can just use the ground plane with a color or add a box for the table.
        # Let's add a visual box for the table to match the dimensions 60x60cm
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.3, 0.3, 0.01], material=[0.9, 0.9, 0.9]) # White/Light Gray table
        builder.add_box_collision(half_size=[0.3, 0.3, 0.01])
        builder.initial_pose = sapien.Pose(p=[0.3, 0.3, -0.01]) # Center at 0.3, 0.3
        self.table = builder.build_static(name="table")

        # 2. Tape Lines (Visual)
        # Tape width: 1.8cm = 0.018m. Half width = 0.009m.
        # Grid Height: 16.4cm = 0.164m.
        # Y Center: 26.0cm = 0.26m.
        # Top Y: 0.342m. Bottom Y: 0.178m.
        
        tape_material = [0, 0, 0] # Black
        tape_half_width = 0.009
        tape_height = 0.001
        
        # Horizontal Lines (Top and Bottom)
        # Spanning from Left Grid Left (0.051) to Right Grid Right (0.580) approx.
        # Let's draw separate boxes for each grid or continuous lines?
        # Continuous lines are easier.
        # Total width approx: 0.580 - 0.051 = 0.529. Center X approx 0.3155.
        
        # Bottom Line
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.27, tape_half_width, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.316, 0.178, 0.001])
        builder.build_static(name="tape_bottom")
        
        # Top Line
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.27, tape_half_width, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.316, 0.342, 0.001])
        builder.build_static(name="tape_top")
        
        # Vertical Lines
        # 1. Left of Left Grid: X = 13.4 - 8.3 = 5.1cm = 0.051m
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.051, 0.26, 0.001])
        builder.build_static(name="tape_left_1")
        
        # 2. Right of Left Grid / Left of Mid Grid
        # Left Grid Right: 21.7cm. Mid Grid Left: 23.8cm.
        # Let's place tape in the gap? Or at boundaries?
        # Let's place at boundaries.
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.217, 0.26, 0.001])
        builder.build_static(name="tape_left_2")
        
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.238, 0.26, 0.001])
        builder.build_static(name="tape_mid_1")
        
        # 3. Right of Mid Grid / Left of Right Grid
        # Mid Grid Right: 39.4cm. Right Grid Left: 41.4cm.
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.394, 0.26, 0.001])
        builder.build_static(name="tape_mid_2")
        
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.414, 0.26, 0.001])
        builder.build_static(name="tape_right_1")
        
        # 4. Right of Right Grid: X = 49.7 + 8.3 = 58.0cm = 0.580m
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[tape_half_width, 0.082, tape_height], material=tape_material)
        builder.initial_pose = sapien.Pose(p=[0.580, 0.26, 0.001])
        builder.build_static(name="tape_right_2")

        # 3. Robot Bases (Visual)
        # Left Base: Center X = 0.119m. Width 11cm.
        # Right Base: Center X = 0.433m. Width 11cm.
        # Y Position: Below grid. Grid bottom 0.178. Let's place at Y=0.10.
        # Size: 11cm x 11cm? Or just a plate.
        
        base_material = [0.2, 0.2, 0.2] # Dark Grey
        
        # Left Base
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.055, 0.055, 0.005], material=base_material)
        builder.initial_pose = sapien.Pose(p=[0.119, 0.10, 0.005])
        builder.build_static(name="left_base_visual")
        
        # Right Base (Where our robot is)
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.055, 0.055, 0.005], material=base_material)
        builder.initial_pose = sapien.Pose(p=[0.433, 0.10, 0.005])
        builder.build_static(name="right_base_visual")

    def _load_agent(self, options: dict):
        # Robot Base Position
        # Left Base: Left edge 6.4cm from table left. Width 11cm. Center = 6.4 + 5.5 = 11.9cm = 0.119m
        # Right Base: Left Base End (6.4+11=17.4) + Gap 20.4 = 37.8. Width 11. Center = 37.8 + 5.5 = 43.3cm = 0.433m
        # Y position: "Located in the area below the grid". 
        # Grid bottom is at 26.0 - 8.2 = 17.8cm.
        # Let's place them at Y=10cm for now.
        
        # We only have one robot in the scene for now? 
        # "Task 1: Lift... Right arm grabs red cube".
        # "Task 2: Stack... Right arm".
        # "Task 3: Sort... Green -> Left, Red -> Right".
        # It seems we might need two robots or one robot with two arms?
        # The doc says "Left Arm Base" and "Right Arm Base".
        # SO101 is a single arm.
        # So we likely need TWO agents if we want both arms, or just one for the specific task.
        # For now, let's spawn ONE robot at the RIGHT base position (since Lift/Stack use Right arm).
        # Right Base Center X = 0.433
        
        super()._load_agent(options, sapien.Pose(p=[0.433, 0.05, 0]))

    def _load_objects(self, options: dict):
        # Red Cube
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.015, 0.015, 0.015])
        builder.add_box_visual(half_size=[0.015, 0.015, 0.015], material=[1, 0, 0])
        builder.initial_pose = sapien.Pose(p=[0.497, 0.26, 0.015]) # Default to Right Grid Center
        self.red_cube = builder.build(name="red_cube")

        # Green Cube
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.015, 0.015, 0.015])
        builder.add_box_visual(half_size=[0.015, 0.015, 0.015], material=[0, 1, 0])
        builder.initial_pose = sapien.Pose(p=[0.497, 0.30, 0.015])
        self.green_cube = builder.build(name="green_cube")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Basic initialization
        with torch.device(self.device):
            # Reset robot to rest pose
            # self.agent.reset(self.agent.keyframes["rest"].qpos) # This is done automatically by BaseEnv if keyframes exist?
            # Actually BaseEnv uses the first keyframe or zeros.
            pass

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: dict):
        return {}
