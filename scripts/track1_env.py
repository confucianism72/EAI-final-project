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
        sim_distorted_images: bool = False,
        **kwargs
    ):
        self.task = task
        self.domain_randomization = domain_randomization
        self.sim_distorted_images = sim_distorted_images
        self.grid_bounds = {}  # Will be populated in _compute_grids
            
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        # Precompute distortion maps if needed (after super init so device is ready)
        if self.sim_distorted_images:
            self._setup_distortion_maps()

    def _setup_distortion_maps(self):
        """Precompute torch grid for simulating distortion via F.grid_sample."""
        import cv2
        import torch
        
        # Camera Parameters
        W, H = 640, 480
        
        # Intrinsic Matrix (K)
        self.mtx_intrinsic = np.array([
            [570.21740069, 0., 327.45975405],
            [0., 570.1797441, 260.83642155],
            [0., 0., 1.]
        ], dtype=np.float64)
        
        # Distortion Coefficients (D)
        self.dist_coeffs = np.array([
            -0.735413911, 0.949258417, 0.000189059234, -0.00200351391, -0.864150312
        ], dtype=np.float64)
        
        # 1. Optimal New Camera Matrix (P) for rectification (alpha=1.0)
        image_size = (W, H)
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.mtx_intrinsic, self.dist_coeffs, image_size, 1.0, image_size
        )
        new_mtx[0, 2] = W / 2
        new_mtx[1, 2] = H / 2
        
        # 2. Grid generation for target distorted image
        xs = np.arange(W)
        ys = np.arange(H)
        xx, yy = np.meshgrid(xs, ys)
        
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        points = points.reshape(-1, 1, 2)
        
        # 3. undistortPoints to find source coordinates (in rectified image)
        undistorted_pts = cv2.undistortPoints(
            points, 
            cameraMatrix=self.mtx_intrinsic, 
            distCoeffs=self.dist_coeffs, 
            R=None, 
            P=new_mtx
        )
        
        map_xy = undistorted_pts.reshape(H, W, 2) # (H, W, 2) in pixels
        mapx = map_xy[:, :, 0]
        mapy = map_xy[:, :, 1]
        
        # 4. Normalize to [-1, 1] for F.grid_sample
        # grid_sample treats (-1, -1) as top-left corner
        grid_x = 2.0 * mapx / (W - 1) - 1.0
        grid_y = 2.0 * mapy / (H - 1) - 1.0
        
        # Stack to (H, W, 2)
        grid = np.stack((grid_x, grid_y), axis=2).astype(np.float32)
        
        # Convert to Tensor on device
        # grid_sample expects (N, H, W, 2)
        # We will expand N aka Batch size dynamically or broadcast?
        # F.grid_sample requires N to match or be broadcastable? 
        # Usually N must match input. We'll expand later.
        self.distortion_grid = torch.from_numpy(grid).to(device=self.device) # (H, W, 2)

    def _apply_distortion_to_obs(self, obs):
        """Apply distortion to RGB images in observation using GPU grid_sample."""
        import torch.nn.functional as F
        
        if isinstance(obs, dict) and "image" in obs:
            if "front_camera" in obs["image"]:
                if "rgb" in obs["image"]["front_camera"]:
                    rgb_tensor = obs["image"]["front_camera"]["rgb"] # Expecting Tensor
                    
                    if not isinstance(rgb_tensor, torch.Tensor):
                        # Should not happen in GPU sim typically, but safe fallback logic or skip
                        return obs

                    # Input: (B, H, W, C) or (H, W, C) generally for ManiSkill sensors
                    # grid_sample needs (B, C, H, W)
                    
                    is_batch = len(rgb_tensor.shape) == 4
                    if not is_batch:
                        # Add batch dim: (1, H, W, C)
                        img_in = rgb_tensor.unsqueeze(0)
                    else:
                        img_in = rgb_tensor

                    B, H, W, C = img_in.shape
                    
                    # Permute to (B, C, H, W)
                    img_in = img_in.permute(0, 3, 1, 2).float() # (B, C, H, W)
                    
                    # Prepare grid: Expand (H, W, 2) to (B, H, W, 2)
                    grid = self.distortion_grid.unsqueeze(0).expand(B, -1, -1, -1)
                    
                    # Sample
                    # mode='bilinear': clean interpolation
                    # padding_mode='border': replicates edge pixels (like REPLICATE in cv2)
                    # align_corners=True: matches standard CV coordinates usually
                    distorted = F.grid_sample(img_in, grid, mode='bilinear', padding_mode='border', align_corners=True)
                    
                    # Permute back to (B, H, W, C)
                    distorted = distorted.permute(0, 2, 3, 1)
                    
                    # Convert type back if needed (usually float is fine, typical gym space is uint8 0-255)
                    # ManiSkill usually keeps sensor data as float or uint8?
                    if rgb_tensor.dtype == torch.uint8:
                        distorted = distorted.to(torch.uint8)
                    else:
                        distorted = distorted.to(rgb_tensor.dtype)
                        
                    if not is_batch:
                        obs["image"]["front_camera"]["rgb"] = distorted.squeeze(0)
                    else:
                        obs["image"]["front_camera"]["rgb"] = distorted

        return obs

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.sim_distorted_images:
            obs = self._apply_distortion_to_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.sim_distorted_images:
            obs = self._apply_distortion_to_obs(obs)
        return obs, reward, terminated, truncated, info

    def _get_obs_extra(self, info: dict):
        """Return extra observations."""
        return {
            "red_cube_pos": self.red_cube.pose.p,
            "green_cube_pos": self.green_cube.pose.p,
        }

    @property
    def _default_sensor_configs(self):
        """Front Camera with optional config file override for manual tuning."""
        import os
        import json
        from scipy.spatial.transform import Rotation
        
        # Default parameters
        base_pos = [0.316, 0.260, 0.407]
        pitch, yaw, roll = -90, 0, 0  # degrees
        fov_deg = 73.63
        
        # Check for camera config file (for manual tuning)
        config_path = os.environ.get('CAMERA_CONFIG_PATH', '')
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                base_pos = config.get('position', base_pos)
                pitch = config.get('pitch', pitch)
                yaw = config.get('yaw', yaw)
                roll = config.get('roll', roll)
                fov_deg = config.get('fov', fov_deg)
                print(f"[Track1Env] Loaded camera config from {config_path}")
            except Exception as e:
                print(f"[Track1Env] Warning: Failed to load camera config: {e}")
        
        # Convert Euler angles to quaternion
        rot = Rotation.from_euler('xyz', [pitch, yaw, roll], degrees=True)
        q_scipy = rot.as_quat()  # [x, y, z, w]
        q_sapien = [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]]  # [w, x, y, z]
        
        if self.domain_randomization and hasattr(self, 'num_envs') and self.num_envs > 1:
            base_pose = sapien.Pose(p=base_pos, q=q_sapien)
            pose = Pose.create(base_pose)
            pose = pose * Pose.create_from_pq(
                p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
                q=randomization.random_quaternions(
                    n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
                ),
            )
        else:
            pose = sapien.Pose(p=base_pos, q=q_sapien)
        
        return [
            CameraConfig(
                "front_camera",
                pose=pose,
                width=640,
                height=480,
                fov=np.deg2rad(fov_deg),
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
            # Physical ground removed as per user request (objects exceeding table should fail)
            # scene.add_ground(0, material=ground_material, render=False)
            
            # Add visual ground plane (randomized color)
            if self.domain_randomization:
                # Random earth-tone/dark colors
                color = np.random.uniform(0.1, 0.4, size=3).tolist() + [1]
            else:
                color = [0.1, 0.1, 0.1, 1] # Dark gray/blackish
            
            builder = scene.create_actor_builder()
            builder.add_box_visual(half_size=[2.0, 2.0, 0.1], material=sapien.render.RenderMaterial(base_color=color))
            builder.initial_pose = sapien.Pose(p=[0, 0, -0.11], q=[1, 0, 0, 0]) # Below table (-0.01)
            builder.build_static(name="visual_ground")
        
        # Compute grid layout for this reconfiguration
        self._compute_grids()
        
        # Table surface with optional color randomization
        self._build_table()
        self._build_tape_lines()
        self._build_debug_markers()
        
        # Load task objects
        self._load_objects(options)

    def _build_debug_markers(self):
        """Build debug markers for coordinate system visualization.
        Red at (0,0), Green at (1,0), Blue at (0,1).
        """
        marker_height = 0.005 # Slightly above table/ground
        radius = 0.02
        
        markers = [
            {"pos": [0, 0, marker_height], "color": [1, 0, 0], "name": "debug_origin_red"},
            {"pos": [1, 0, marker_height], "color": [0, 1, 0], "name": "debug_x1_green"},
            {"pos": [0, 1, marker_height], "color": [0, 0, 1], "name": "debug_y1_blue"},
        ]
        
        for marker in markers:
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(radius=radius, material=marker["color"])
            builder.initial_pose = sapien.Pose(p=marker["pos"])
            builder.build_static(name=marker["name"])

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

    def _compute_grids(self):
        """Compute grid coordinates and boundaries with optional randomization."""
        tape_half_width = 0.009
        
        # Base values (Human specified)
        x_1 = 0.204
        x_4 = 0.6
        y_1 = 0.15
        upper_height = 0.164
        
        # Add randomization if enabled
        # if self.domain_randomization:
        #     noise_scale = 0.005 # +/- 5mm
        #     x_1 += np.random.uniform(-noise_scale, noise_scale)
        #     # x_4 (table width) usually fixed or small noise
        #     x_4 += np.random.uniform(-0.002, 0.002) 
        #     y_1 += np.random.uniform(-noise_scale, noise_scale)
        #     upper_height += np.random.uniform(-0.002, 0.002)
        
        # NOTE: User requested independent tape randomization. 
        # We keep the logical grid bounds deterministic (or globally fixed for this episode)
        # so success criteria are consistent, but the Visual Tape will be noisy.
            
        # Calculate derived coordinates
        x = [0.0] * 5
        x[1] = x_1
        x[0] = x[1] - 0.166 - 2 * tape_half_width
        x[4] = x_4
        x[2] = x[4] - 0.204 - 2 * tape_half_width
        x[3] = x[4] - 0.204 + 0.166
        
        y = [0.0] * 3
        y[0] = 0.0
        y[1] = y_1
        y[2] = y_1 + upper_height + 2 * tape_half_width
        
        # Store for _build_tape_lines
        self.grid_points = {"x": x, "y": y, "tape_half_width": tape_half_width}
        
        # Calculate logical boundaries for success/placement (Inner areas excluding tape)
        # Left Grid: between col1(x[0]) and col2(x[1]) ?? 
        # Wait, let's map the user's tape logic to logical areas.
        
        # Tape logic from user:
        # row1: y[1] to y[1]+2w (Separates Bottom and Upper?) No, row1 is y[1]. 
        # row2: y[2] to ...
        
        # Based on user code:
        # Row 1 pos y: y[1] + w.  Size y: w. -> Tape is from y[1] to y[1]+2w.
        # Row 2 pos y: y[2] + w.  Size y: w. -> Tape is from y[2] to y[2]+2w.
        
        # Col 1 pos x: x[0] + w.  Size x: w. -> Tape is from x[0] to x[0]+2w.
        # Col 2 pos x: x[1] + w.  Size x: w. -> Tape is from x[1] to x[1]+2w.
        
        # So the grid "Left" is likely between Col 1 and Col 2, and Row 1 and Row 2.
        # Left Grid Bounds:
        # X: (x[0] + 2w) to x[1]
        # Y: (y[1] + 2w) to y[2] 
        
        w = tape_half_width
        
        self.grid_bounds["left"] = {
            "x_min": x[0] + 2*w, "x_max": x[1],
            "y_min": y[1] + 2*w, "y_max": y[2]
        }
        
        self.grid_bounds["mid"] = {
            "x_min": x[1] + 2*w, "x_max": x[2], # Wait, is there a tape between Left and Mid?
            # User code: col1, col4, col2, col3, col5.
            # col1 @ x[0], col2 @ x[1], col3 @ x[2], col4 @ x[3], col5 @ x[4]?
            # Let's re-read user code logic carefully.
            # col1: x[0]. col2: x[1]. col3: x[2]. col4: x[3]. col5: x[4]... 
            # col4 pos: x[3]+w. 
            
            # Left Grid is between x[0] and x[1].
            # Mid Grid is between x[1] and x[2]? Or x[1] and x[2] are edges?
            # x[2] = x[4] - 0.204 - 2w.
            # x[3] = x[4] - 0.204 + 0.166.
            
            # It seems:
            # Left: x[0]...x[1]
            # Gap?
            # Mid: x[1]...x[2] ?? No, x[1]=0.204. x[2] ~ 0.6-0.2-small = 0.38.
            # Right: x[3]...x[4]? x[3] ~ 0.56. x[4]=0.6. width ~4cm? No.
            
            # Let's trust the areas defined by the columns.
            # Left Grid: Inside col1 and col2.
            "y_min": y[1] + 2*w, "y_max": y[2]
        }
        
        # Re-evaluating Mid/Right based on user's manual "draw correctly" code
        # User X array: x[0], x[1], x[2], x[3], x[4]
        # col1 at x[0]
        # col2 at x[1]
        # col3 at x[2]
        # col4 at x[3]
        # col5 at x[4]
        
        # Left Grid: between col1 and col2.
        # Mid Grid: between col2 and col3.
        self.grid_bounds["mid"] = {
            "x_min": x[1] + 2*w, "x_max": x[2],
            "y_min": y[1] + 2*w, "y_max": y[2]
        }
        
        # Right Grid: between col3 and col4 ? 
        # OR col3 and col5?
        # x[3] = x[4] - 0.204 + 0.166.  = 0.562.  x[4]=0.6. Diff = 0.038. Too small for Right grid.
        # x[2] = x[4] - 0.204 - 2w = 0.378.
        # Gap between x[2] and x[3] = 0.562 - 0.378 = 0.184. This looks like the Right Grid!
        
        # So Right Grid is between col3(x[2]) and col4(x[3]).
        self.grid_bounds["right"] = {
            "x_min": x[2] + 2*w, "x_max": x[3],
            "y_min": y[1] + 2*w, "y_max": y[2]
        }
        
        # Bottom Grid (between robot bases)
        # Usually below Mid.
        # User code: col2 and col3 extend down to y[0]?
        # col2 pos y: (y[2]+y[0])/2. Height: (y[2]-y[0])/2. -> Spans y[0] to y[2].
        # col3 pos y: (y[2]+y[0])/2. -> Spans y[0] to y[2].
        # So col2 and col3 go all the way down.
        # Thus Bottom Grid is between col2 and col3, and between row? (no bottom row tape?)
        # row1 is at y[1].
        # So Bottom Grid is y[0] to y[1].
        self.grid_bounds["bottom"] = {
            "x_min": x[1] + 2*w, "x_max": x[2],
            "y_min": y[0], "y_max": y[1]
        }


    def _build_tape_lines(self):
        """Build black tape lines using computed grid points."""
        tape_material = [0, 0, 0]
        tape_height = 0.001
        
        # Retrieve computed params
        x = self.grid_points["x"]
        y = self.grid_points["y"]
        tape_half_width = self.grid_points["tape_half_width"]
        
        tape_specs = []

        tape_specs.append({
            "half_size": [(x[3]- x[0]) / 2 + tape_half_width, tape_half_width, tape_height],
            "pos": [(x[3] +  x[0]) / 2 + tape_half_width, y[1] + tape_half_width, 0.001],
            "name": "row1"
        })

        tape_specs.append({
            "half_size": [(x[3]- x[0]) / 2 + tape_half_width, tape_half_width, tape_height],
            "pos": [(x[3] +  x[0]) / 2 + tape_half_width, y[2] + tape_half_width, 0.001],
            "name": "row2"
        })


        tape_specs.append({
            "half_size": [tape_half_width, (y[2] - y[1])/2 + tape_half_width , tape_height],
            "pos": [x[0] + tape_half_width, (y[2] + y[1])/2 + tape_half_width, 0.001],
            "name": "col1"
        })

        tape_specs.append({
            "half_size": [tape_half_width, (y[2] - y[1])/2 + tape_half_width , tape_height],
            "pos": [x[3] + tape_half_width, (y[2] + y[1])/2 + tape_half_width, 0.001],
            "name": "col4"
        })

        tape_specs.append({
            "half_size": [tape_half_width, (y[2] - y[0])/2 + tape_half_width , tape_height],
            "pos": [x[1] + tape_half_width, (y[2] + y[0])/2 + tape_half_width, 0.001],
            "name": "col2"
        })
        
        tape_specs.append({
            "half_size": [tape_half_width, (y[2] - y[0])/2 + tape_half_width , tape_height],
            "pos": [x[2] + tape_half_width, (y[2] + y[0])/2 + tape_half_width, 0.001],
            "name": "col3"
        })

        tape_specs.append({
            "half_size": [tape_half_width, 0.6 / 2 , tape_height],
            "pos": [x[4] + tape_half_width, 0.6 / 2, 0.001],
            "name": "col5"
        })
        
        # Build all tape lines
        for spec in tape_specs:
            builder = self.scene.create_actor_builder()
            
            # Apply independent randomization if enabled
            pos = list(spec["pos"])
            half_size = list(spec["half_size"])
            rotation = [1, 0, 0, 0] # Identity quaternion
            
            if self.domain_randomization:
                # 1. Position Noise (x, y)
                pos_noise = np.random.uniform(-0.005, 0.005, size=2) # +/- 5mm
                pos[0] += pos_noise[0]
                pos[1] += pos_noise[1]
                
                # 2. Size Noise (length aka half_size[0] mostly, or width)
                size_noise = np.random.uniform(-0.002, 0.002) # +/- 2mm
                # Don't change thickness (z), maybe slight width/length change
                half_size[0] += size_noise 
                
                # 3. Rotation Noise (Yaw)
                # Small rotation around Z axis
                yaw_noise = np.deg2rad(np.random.uniform(-2, 2)) # +/- 2 degrees
                import transforms3d
                rotation = transforms3d.quaternions.axangle2quat([0, 0, 1], yaw_noise)
                # Transforms3d returns [w, x, y, z], Sapien expects [w, x, y, z] match? 
                # Sapien Pose takes q=[w, x, y, z] or [x, y, z, w]?
                # Sapien uses [w, x, y, z] usually. Let's verify or use Sapien's Rotation.
                # Actually sapien.Pose q is [w, x, y, z].
                # Let's use simple randomization without external lib if possible or check imports.
                # simpler:
                q_z = np.sin(yaw_noise / 2)
                q_w = np.cos(yaw_noise / 2)
                rotation = [q_w, 0, 0, q_z]

            builder.add_box_visual(half_size=half_size, material=tape_material)
            builder.initial_pose = sapien.Pose(p=pos, q=rotation)
            builder.build_static(name=spec["name"])



    def _load_agent(self, options: dict):
        # Rotate robots to face +Y (90 degrees around Z axis)
        # q = [cos(pi/4), 0, 0, sin(pi/4)]
        rotation = [0.7071068, 0, 0, 0.7071068]
        
        agent_poses = [
            sapien.Pose(p=[0.119, 0.055, 0], q=rotation),  # Left Robot
            sapien.Pose(p=[0.481, 0.055, 0], q=rotation)   # Right Robot
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
                red_pos = self._random_grid_position(b, self.grid_bounds["right"], z=0.015)
                self.red_cube.set_pose(Pose.create_from_pq(p=red_pos))
                # Hide green cube (move far away)
                self.green_cube.set_pose(Pose.create_from_pq(p=torch.tensor([[-10, -10, -10]] * b, device=self.device)))
                
            elif self.task == "stack":
                # Both cubes in Right Grid, non-overlapping
                red_pos = self._random_grid_position(b, self.grid_bounds["right"], z=0.015)
                green_pos = self._random_grid_position(b, self.grid_bounds["right"], z=0.015)
                # Ensure minimum distance
                for _ in range(10):  # Try a few times to separate them
                    dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
                    overlap = dist < 0.05  # 5cm minimum
                    if not overlap.any():
                        break
                    green_pos[overlap] = self._random_grid_position(overlap.sum().item(), self.grid_bounds["right"], z=0.015)
                
                self.red_cube.set_pose(Pose.create_from_pq(p=red_pos))
                self.green_cube.set_pose(Pose.create_from_pq(p=green_pos))
                
            elif self.task == "sort":
                # Both cubes in Mid Grid
                red_pos = self._random_grid_position(b, self.grid_bounds["mid"], z=0.015)
                green_pos = self._random_grid_position(b, self.grid_bounds["mid"], z=0.005)  # Smaller green cube
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
        # Common failure check: objects falling off table (z < -0.05)
        # Note: Table surface is at 0.0.
        fallen_threshold = -0.05
        red_fallen = self.red_cube.pose.p[:, 2] < fallen_threshold
        
        # If any object vital to the task falls, it's a fail (success=False)
        # We can return 'fail': True in info if needed, but for now strict success check is key.
        
        if self.task == "lift":
            # Only red cube matters
            if red_fallen.any():
                return {"success": torch.zeros_like(red_fallen, dtype=torch.bool), "fail": red_fallen}
            return self._evaluate_lift()
            
        elif self.task == "stack":
            green_fallen = self.green_cube.pose.p[:, 2] < fallen_threshold
            if red_fallen.any() or green_fallen.any():
                is_fallen = red_fallen | green_fallen
                return {"success": torch.zeros_like(is_fallen, dtype=torch.bool), "fail": is_fallen}
            return self._evaluate_stack()
            
        elif self.task == "sort":
            green_fallen = self.green_cube.pose.p[:, 2] < fallen_threshold
            if red_fallen.any() or green_fallen.any():
                is_fallen = red_fallen | green_fallen
                return {"success": torch.zeros_like(is_fallen, dtype=torch.bool), "fail": is_fallen}
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
            (red_pos[:, 0] >= self.grid_bounds["right"]["x_min"]) & 
            (red_pos[:, 0] <= self.grid_bounds["right"]["x_max"]) &
            (red_pos[:, 1] >= self.grid_bounds["right"]["y_min"]) & 
            (red_pos[:, 1] <= self.grid_bounds["right"]["y_max"])
        )
        
        # Check green in Left Grid
        green_in_left = (
            (green_pos[:, 0] >= self.grid_bounds["left"]["x_min"]) & 
            (green_pos[:, 0] <= self.grid_bounds["left"]["x_max"]) &
            (green_pos[:, 1] >= self.grid_bounds["left"]["y_min"]) & 
            (green_pos[:, 1] <= self.grid_bounds["left"]["y_max"])
        )
        
        success = red_in_right & green_in_left
        return {"success": success, "red_in_right": red_in_right, "green_in_left": green_in_left}

    def _get_obs_extra(self, info: dict):
        """Return extra observations."""
        return {
            "red_cube_pos": self.red_cube.pose.p,
            "green_cube_pos": self.green_cube.pose.p,
        }
