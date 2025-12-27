import numpy as np
import sapien
import sapien.render
import torch
import torch.nn.functional as tFunc
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





@register_env("Track1-v0", max_episode_steps=800)  # Default high, actual limit from config
class Track1Env(BaseEnv):
    SUPPORTED_ROBOTS = ["so101", ("so101", "so101")]
    agent: SO101

    def __init__(
        self, 
        *args, 
        robot_uids=("so101", "so101"),
        task: str = "lift",  # "lift", "stack", "sort"
        domain_randomization: bool = True,
        camera_mode: str = "direct_pinhole",  # "distorted", "distort-twice", "direct_pinhole"
        control_mode: str = "pd_joint_target_delta_pos",  # control mode from Hydra config
        render_scale: int = 3,
        reward_config: dict = None,  # Reward configuration from Hydra
        action_bounds: dict = None,  # Per-joint action bounds override
        camera_extrinsic: list = None,  # Camera extrinsic matrix (4x4 cam2world)
        undistort_alpha: float = 0.25,  # Undistortion alpha for pinhole modes
        obs_normalization: dict = None,  # Obs normalization config (qvel_clip, etc.)
        **kwargs
    ):
        self.task = task
        self.domain_randomization = domain_randomization
        self.camera_extrinsic = camera_extrinsic  # Store for use in _default_sensor_configs
        self.undistort_alpha = undistort_alpha  # For pinhole output modes
        
        # Obs normalization config
        if obs_normalization is None:
            obs_normalization = {}
        self.obs_normalize_enabled = obs_normalization.get("enabled", False)
        # qpos normalization scale (default π)
        self.qpos_scale = obs_normalization.get("qpos_scale", np.pi)
        # Per-joint qvel clip ranges
        self.qvel_clip = obs_normalization.get("qvel_clip", [1.0, 2.5, 2.0, 1.0, 0.6, 1.5])
        # Relative position clip (tcp_to_red_pos, etc.)
        self.relative_pos_clip = obs_normalization.get("relative_pos_clip", 0.5)
        # Whether to include absolute positions in obs
        self.include_abs_pos = obs_normalization.get("include_abs_pos", True)
        # How to include target_qpos: True, False, or "relative" (target_qpos - qpos)
        self.include_target_qpos = obs_normalization.get("include_target_qpos", True)
        # Action bounds for relative target_qpos normalization (synced from control config)
        # Format: dict with joint names as keys (e.g., {"shoulder_pan": 0.044, ...})
        self.obs_action_bounds = obs_normalization.get("action_bounds", None)
        # Position normalization params: (pos - mean) / std
        self.tcp_pos_norm = obs_normalization.get("tcp_pos", {"mean": [0.3, 0.3, 0.2], "std": [0.1, 0.1, 0.1]})
        self.red_cube_pos_norm = obs_normalization.get("red_cube_pos", {"mean": [0.3, 0.3, 0.2], "std": [0.1, 0.1, 0.1]})
        self.green_cube_pos_norm = obs_normalization.get("green_cube_pos", {"mean": [0.3, 0.3, 0.2], "std": [0.1, 0.1, 0.1]})
        # Include is_grasped in observations (0/1 -> -1/1)
        self.include_is_grasped = obs_normalization.get("include_is_grasped", False)
        # Include TCP orientation (quaternion)
        self.include_tcp_orientation = obs_normalization.get("include_tcp_orientation", False)

        self.render_scale = render_scale
        
        # Set SO101 action mode based on task BEFORE agent creation
        # This affects the action bounds used in the controller configs
        if task == "sort":
            SO101.active_mode = "dual"
        else:
            SO101.active_mode = "single"
        
        # Override action bounds if provided from config
        if action_bounds is not None:
            if task == "sort":
                SO101.action_bounds_dual_arm = action_bounds
            else:
                SO101.action_bounds_single_arm = action_bounds
        
        # Setup reward configuration (default values if not provided)
        self._setup_reward_config(reward_config)
        
        # Validate camera_mode
        valid_modes = ["distorted", "distort-twice", "direct_pinhole"]
        if camera_mode not in valid_modes:
            raise ValueError(f"camera_mode must be one of {valid_modes}, got '{camera_mode}'")
        self.camera_mode = camera_mode
        
        self.grid_bounds = {}  # Will be populated in _compute_grids

        # Precompute camera processing maps if needed (after super init so device is ready)
        self._setup_camera_processing_maps()
        
        # Set single_arm_mode BEFORE super init (needed in _setup_sensors)
        self.single_arm_mode = task in ["lift", "stack"]
            
        super().__init__(*args, robot_uids=robot_uids, control_mode=control_mode, **kwargs)

        self._setup_device()
        self._setup_single_arm_action_space()
        
    def _setup_reward_config(self, reward_config):
        """Setup reward configuration with defaults."""
        if reward_config is None:
            reward_config = {}
        
        # Reward type: "parallel" or "staged"
        self.reward_type = reward_config.get("reward_type", "staged")
        
        # Weights (new naming scheme with backward compatibility)
        weights = reward_config.get("weights", {})
        self.reward_weights = {
            # New style
            "approach": weights.get("approach", weights.get("reach", 1.0)),
            "horizontal_displacement": weights.get("horizontal_displacement", 0.0),
            "lift": weights.get("lift", 5.0),
            "success": weights.get("success", 10.0),
            # Legacy (for backward compatibility)
            "reach": weights.get("reach", weights.get("approach", 1.0)),
            "grasp": weights.get("grasp", 0.0),
        }
        
        # Approach reward curve (piecewise linear)
        self.approach_threshold = reward_config.get("approach_threshold", 0.01)  # Full reward within this distance (1cm)
        self.approach_zero_point = reward_config.get("approach_zero_point", 0.20)  # Zero reward at this distance (20cm)
        
        # Legacy scaling (kept for backward compatibility, not used with piecewise)
        self.approach_scale = reward_config.get("approach_scale", reward_config.get("reach_scale", 5.0))
        self.reach_scale = self.approach_scale
        
        # Approach mode: 'tcp_midpoint' (single TCP center) or 'dual_point' (separate fixed/moving jaw)
        self.approach_mode = reward_config.get("approach_mode", "dual_point")
        
        # Gripper reference point offsets (only used in dual_point mode)
        self.gripper_tip_offset = reward_config.get("gripper_tip_offset", 0.015)  # Along jaw, back from tip (1.5cm)
        self.gripper_outward_offset = reward_config.get("gripper_outward_offset", 0.015)  # Perpendicular, outward for cube thickness (1.5cm)
        
        # Stage thresholds
        stages = reward_config.get("stages", {})
        self.stage_thresholds = {
            "approach": stages.get("approach_threshold", stages.get("reach_threshold", 0.05)),
            "reach": stages.get("reach_threshold", stages.get("approach_threshold", 0.05)),
            "grasp": stages.get("grasp_threshold", 0.03),
            "lift": stages.get("lift_target", 0.05),
        }
        
        # Lift success config: cube must stay above lift_target for stable_hold_time
        self.lift_target = reward_config.get("lift_target", 0.05)  # 5cm default
        self.lift_max_height = reward_config.get("lift_max_height", None)  # Cap for lift reward (None = no cap)
        self.stable_hold_time = reward_config.get("stable_hold_time", 0.0)  # 0 = instant success
        # Convert hold time to steps (at control_freq, default 30 Hz)
        control_freq = getattr(self, 'control_freq', 30)
        self.stable_hold_steps = int(self.stable_hold_time * control_freq)
        
        # Fail bounds: cube XY must stay within this rectangle
        fail_bounds = reward_config.get("fail_bounds", None)
        if fail_bounds:
            self.fail_bounds = {
                "x_min": fail_bounds.get("x_min", 0.0),
                "x_max": fail_bounds.get("x_max", 1.0),
                "y_min": fail_bounds.get("y_min", 0.0),
                "y_max": fail_bounds.get("y_max", 1.0),
            }
        else:
            self.fail_bounds = None
        
        # Fail penalty weight
        self.reward_weights["fail"] = weights.get("fail", 0.0)
        
        # Spawn bounds: where the red cube is randomly spawned
        spawn_bounds = reward_config.get("spawn_bounds", None)
        if spawn_bounds:
            self.spawn_bounds = {
                "x_min": spawn_bounds.get("x_min", 0.35),
                "x_max": spawn_bounds.get("x_max", 0.55),
                "y_min": spawn_bounds.get("y_min", 0.15),
                "y_max": spawn_bounds.get("y_max", 0.35),
            }
        else:
            self.spawn_bounds = None  # Will use grid_bounds["right"] as default
        
        # Moving jaw (approach2) reference point config
        self.reward_weights["approach2"] = weights.get("approach2", 0.0)
        self.moving_jaw_tip_offset = reward_config.get("moving_jaw_tip_offset", 0.015)
        self.moving_jaw_outward_offset = reward_config.get("moving_jaw_outward_offset", 0.01)
        self.approach2_threshold = reward_config.get("approach2_threshold", 0.01)
        self.approach2_zero_point = reward_config.get("approach2_zero_point", 0.20)
        
        # Action rate penalty (anti-jitter)
        self.reward_weights["action_rate"] = weights.get("action_rate", 0.0)
        self.prev_action = None  # Will be initialized on first step
        
        # Horizontal displacement threshold: only penalize moves > threshold
        self.horizontal_displacement_threshold = reward_config.get("horizontal_displacement_threshold", 0.0)
        
        # Grasp detection parameters (for is_grasping)
        self.grasp_min_force = reward_config.get("grasp_min_force", 0.5)
        self.grasp_max_angle = reward_config.get("grasp_max_angle", 110)
        
        # Grasp reward weight
        self.reward_weights["grasp"] = weights.get("grasp", 0.0)
        
        # Gated lift reward: only give lift reward when is_grasped=True
        self.gate_lift_with_grasp = reward_config.get("gate_lift_with_grasp", False)

    def _setup_single_arm_action_space(self):
        """For lift/stack tasks, only expose right arm action space."""
        import gymnasium as gym
        from gymnasium.vector.utils import batch_space
        
        # Check if this is a single-arm task
        self.single_arm_mode = self.task in ["lift", "stack"]
        
        if self.single_arm_mode and self.agent is not None:
            # Original action space is Dict({'so101-0': Box, 'so101-1': Box})
            # so101-0 is at X=0.119 (left), so101-1 is at X=0.481 (right)
            # For single-arm tasks, we use the RIGHT arm (so101-1)
            if isinstance(self.single_action_space, gym.spaces.Dict):
                # Right arm is so101-1 (larger X coordinate)
                self._right_arm_key = "so101-1"
                self._left_arm_key = "so101-0"
                
                right_arm_space = self.single_action_space[self._right_arm_key]
                
                # Override action spaces to only expose right arm
                self.single_action_space = right_arm_space
                self.action_space = batch_space(self.single_action_space, n=self.num_envs)
                
                # Store original for step() to reconstruct full action
                self._left_arm_action_dim = self.agent.single_action_space[self._left_arm_key].shape[0]

    def _setup_device(self):
        assert hasattr(self, 'device')
        if hasattr(self, 'distortion_grid'):
            self.distortion_grid = self.distortion_grid.to(self.device)
        if hasattr(self, 'undistortion_grid'):
            self.undistortion_grid = self.undistortion_grid.to(self.device)

    def _setup_camera_processing_maps(self):
        """Precompute torch grids for camera distortion/undistortion processing via tFunc.grid_sample.
        
        Pipeline:
        - Source: Rendered at (640×scale) × (480×scale) with scaled intrinsic matrix
        - Distortion: Maps source -> 640×480 distorted output
        - Undistortion (alpha=0): Maps 640×480 distorted -> 640×480 clean pinhole
        """
        import cv2
        
        # Camera intrinsic parameters (from real camera calibration)
        self.mtx_intrinsic = np.array([
            [570.21740069, 0., 327.45975405],
            [0., 570.1797441, 260.83642155],
            [0., 0., 1.]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.array([
            -0.735413911, 0.949258417, 0.000189059234, -0.00200351391, -0.864150312
        ], dtype=np.float64)
        
        # Scale factor for high-res rendering
        
        # Source image size (high-res pinhole render)
        OUT_W, OUT_H = 640, 480
        SRC_W = OUT_W * self.render_scale
        SRC_H = OUT_H * self.render_scale
        if self.camera_mode in ["distorted", "distort-twice"]:
            self.front_render_width = SRC_W
            self.front_render_height = SRC_H

            # Get the undistorted intrinsic matrix using getOptimalNewCameraMatrix with alpha=1
            # This gives us the intrinsic for a pinhole camera that covers all distorted pixels
            new_mtx_alpha1, _ = cv2.getOptimalNewCameraMatrix(
                self.mtx_intrinsic, self.dist_coeffs, (OUT_W, OUT_H), 1.0, (SRC_W, SRC_H)
            )
            
            # Scale the new_mtx to render resolution
            self.render_intrinsic = new_mtx_alpha1.copy()
            
            # ============ Distortion Grid (SRC -> OUT distorted) ============
            # For each pixel in the 640×480 distorted output, find where it maps to in the source
            
            # Step 1: Generate grid for distorted output image (640×480)
            xs = np.arange(OUT_W)
            ys = np.arange(OUT_H)
            xx, yy = np.meshgrid(xs, ys)
            points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32).reshape(-1, 1, 2)
            
            # Step 2: undistortPoints with P=scaled_intrinsic gives coordinates in render space directly
            undistorted_pts = cv2.undistortPoints(
                points, 
                cameraMatrix=self.mtx_intrinsic, 
                distCoeffs=self.dist_coeffs, 
                R=None, 
                P=self.render_intrinsic  # Project to render camera space
            )
            map_xy_render = undistorted_pts.reshape(OUT_H, OUT_W, 2)
            
            # Step 3: Normalize to [-1, 1] for grid_sample
            grid_x = 2.0 * map_xy_render[:, :, 0] / (SRC_W - 1) - 1.0
            grid_y = 2.0 * map_xy_render[:, :, 1] / (SRC_H - 1) - 1.0
            distortion_grid = np.stack((grid_x, grid_y), axis=2).astype(np.float32)
            self.distortion_grid = torch.from_numpy(distortion_grid)# .to(device=self.device)  # (OUT_H, OUT_W, 2)
            
            # ============ Undistortion Grid ============
            # This maps 640x480 distorted -> 640x480 clean pinhole
        if self.camera_mode in ["distort-twice", "direct_pinhole"]:
            # Get new camera matrix with configurable alpha
            # alpha=0: crop black borders, alpha=1: keep all pixels (shrinks image)
            # alpha=0.25 is optimal for full work area visibility
            alpha = getattr(self, 'undistort_alpha', 0.25)
            new_mtx_undist, _ = cv2.getOptimalNewCameraMatrix(
                self.mtx_intrinsic, self.dist_coeffs, (OUT_W, OUT_H), alpha, (OUT_W, OUT_H)
            )
            
            if self.camera_mode == "direct_pinhole":
                self.front_render_width = OUT_W
                self.front_render_height = OUT_H
                self.render_intrinsic = new_mtx_undist.copy()
                return
            # initUndistortRectifyMap gives us the mapping from undistorted -> distorted source
            # We need the inverse for grid_sample
            map1, map2 = cv2.initUndistortRectifyMap(
                self.mtx_intrinsic, self.dist_coeffs, None, new_mtx_undist, (OUT_W, OUT_H), cv2.CV_32FC1
            )
            
            # map1, map2 are (OUT_H, OUT_W) containing x, y source coordinates
            # Normalize to [-1, 1]
            undist_grid_x = 2.0 * map1 / (OUT_W - 1) - 1.0
            undist_grid_y = 2.0 * map2 / (OUT_H - 1) - 1.0
            undistortion_grid = np.stack((undist_grid_x, undist_grid_y), axis=2).astype(np.float32)
            self.undistortion_grid = torch.from_numpy(undistortion_grid).to(device=self.device)  # (OUT_H, OUT_W, 2)

    def _apply_camera_processing(self, obs):
        """Apply camera processing based on camera_mode.
        
        Modes:
        - direct_pinhole: No processing (already rendered with correct params)
        - distorted: Apply distortion to 1920x1440 source -> 640x480 distorted output
        - distort-twice: distorted -> then undistort (alpha=0) -> 640x480 clean
        """
        
        if self.camera_mode == "direct_pinhole":
            return obs  # No processing needed
        
        # Skip if grids not yet initialized (happens during parent __init__ reset)
        if not hasattr(self, 'distortion_grid'):
            return obs
        
        # Find the RGB tensor - could be in 'sensor_data' or 'image'
        rgb_tensor = None
        obs_key = None
        
        if isinstance(obs, dict):
            if "sensor_data" in obs and "front_camera" in obs["sensor_data"]:
                if "rgb" in obs["sensor_data"]["front_camera"]:
                    rgb_tensor = obs["sensor_data"]["front_camera"]["rgb"]
                    obs_key = "sensor_data"
            elif "image" in obs and "front_camera" in obs["image"]:
                if "rgb" in obs["image"]["front_camera"]:
                    rgb_tensor = obs["image"]["front_camera"]["rgb"]
                    obs_key = "image"
        
        if rgb_tensor is None or not isinstance(rgb_tensor, torch.Tensor):
            return obs

        # Input: (B, SRC_H, SRC_W, C) or (SRC_H, SRC_W, C)
        # For distorted/distort-twice: Source is 1920x1440
        is_batch = len(rgb_tensor.shape) == 4
        if not is_batch:
            img_in = rgb_tensor.unsqueeze(0)
        else:
            img_in = rgb_tensor

        B = img_in.shape[0]
        original_dtype = rgb_tensor.dtype
        
        # Permute to (B, C, H, W) for grid_sample
        img_in = img_in.permute(0, 3, 1, 2).float()
        
        # Ensure grids are on same device as input
        device = img_in.device
        dist_grid = self.distortion_grid.to(device).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Step 1: Apply distortion (1920x1440 -> 640x480)
        distorted = tFunc.grid_sample(img_in, dist_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        if self.camera_mode == "distorted":
            result = distorted
        elif self.camera_mode == "distort-twice":
            # Step 2: Apply undistortion (640x480 distorted -> 640x480 clean)
            undist_grid = self.undistortion_grid.to(device).unsqueeze(0).expand(B, -1, -1, -1)
            result = tFunc.grid_sample(distorted, undist_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        else:
            result = distorted  # Fallback
        
        # Permute back to (B, H, W, C)
        result = result.permute(0, 2, 3, 1)
        
        # Restore dtype
        if original_dtype == torch.uint8:
            result = result.clamp(0, 255).to(torch.uint8)
        else:
            result = result.to(original_dtype)
            
        if not is_batch:
            obs[obs_key]["front_camera"]["rgb"] = result.squeeze(0)
        else:
            obs[obs_key]["front_camera"]["rgb"] = result

        return obs

    @property
    def _default_sensor_configs(self):
        """Front Camera with optional config file override for manual tuning."""
        
        # Use extrinsic matrix from config if provided
        if self.camera_extrinsic is not None:
            extrinsic = np.array(self.camera_extrinsic)
            R = extrinsic[:3, :3]  # Rotation matrix (cam2world)
            eye = extrinsic[:3, 3]  # Camera position
            
            # forward = camera Z axis in world (third column of R)
            forward = R[:, 2]
            
            # up = camera -Y axis in world (images have Y pointing down)
            up = -R[:, 1]
            
            # target = eye + forward * distance (use original distance ~0.407)
            distance = 0.407
            target = eye + forward * distance
            
            pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist(), up=up.tolist())
        else:
            # Default look_at parameters
            pose = sapien_utils.look_at(eye=[0.316, 0.260, 0.407], target=[0.316, 0.260, 0.0], up=[0, -1, 0])
        
        if self.domain_randomization and hasattr(self, 'num_envs') and self.num_envs > 1:
            # base_pose = sapien.Pose(p=base_pos, q=q_sapien)
            # pose = Pose.create(base_pose)
            
            # Note: look_at returns a sapien.Pose (cpu). We need to convert if using batch logic?
            # look_at supports batch if inputs are tensors. Here inputs are lists.
            # So 'pose' is a single sapien.Pose.
            
            # Convert to Maniskill Pose to apply randomization
            pose = Pose.create(pose)
            
            pose = pose * Pose.create_from_pq(
                p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
                q=randomization.random_quaternions(
                    n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
                ),
            )
        
        # Determine resolution and intrinsic based on camera_mode
        if self.camera_mode == "direct_pinhole":
            return [
                CameraConfig(
                    "front_camera",
                    pose=pose,
                    width=self.front_render_width,
                    height=self.front_render_height,
                    intrinsic=self.render_intrinsic,
                    near=0.01,
                    far=100,
                ),
            ]
        else:
            # High-res source for distortion pipeline using scaled intrinsic
            return [
                CameraConfig(
                    "front_camera",
                    pose=pose,
                    width=self.front_render_width,
                    height=self.front_render_height,
                    intrinsic=self.render_intrinsic,
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
            # Skip left arm wrist camera (index 0, so101-0) for single-arm tasks
            # Right arm is index 1 (so101-1)
            if self.single_arm_mode and i == 0:
                continue
                
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

    def render_sensors(self):
        """Override to render only RGB images from sensors (no depth/segmentation).
        
        This reduces rendering compute by only requesting RGB texture from the shader,
        unlike the default which requests all textures (rgb + position + segmentation).
        """
        from mani_skill.utils.visualization.misc import tile_images
        from mani_skill.sensors.camera import Camera
        
        # Hide objects that should be hidden for observation
        for obj in self._hidden_objects:
            obj.hide_visual()
        
        # Update render for sensors only
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.capture_sensor_data()
        
        images = []
        for name, sensor in self._sensors.items():
            if isinstance(sensor, Camera):
                # Request ONLY RGB - this is the key difference that reduces compute
                obs = sensor.get_obs(
                    rgb=True,
                    depth=False,
                    position=False,
                    segmentation=False,
                    apply_texture_transforms=True
                )
                if 'rgb' in obs:
                    images.append(obs['rgb'])
        
        if len(images) == 0:
            # Fallback to default if no RGB found
            return super().render_sensors()
        
        return tile_images(images)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.camera_mode != "direct_pinhole":
            obs = self._apply_camera_processing(obs)
        return obs, info

    def step(self, action):
        # For single-arm mode (lift/stack), convert single-arm action to full multi-arm action
        if getattr(self, 'single_arm_mode', False) and hasattr(self, '_left_arm_action_dim'):
            import torch
            # action is a tensor of shape (num_envs, right_arm_action_dim)
            # We need to convert it to a dict for the multi-agent
            if isinstance(action, (np.ndarray, torch.Tensor)):
                if isinstance(action, np.ndarray):
                    action = torch.tensor(action, device=self.device, dtype=torch.float32)
                
                # Create full action dict with zeros for left arm
                left_arm_zeros = torch.zeros(
                    (action.shape[0], self._left_arm_action_dim), 
                    device=self.device, 
                    dtype=action.dtype
                )
                action = {
                    self._right_arm_key: action,
                    self._left_arm_key: left_arm_zeros,
                }
        
        obs, reward, terminated, truncated, info = super().step(action)
        if self.camera_mode != "direct_pinhole":
            obs = self._apply_camera_processing(obs)
        return obs, reward, terminated, truncated, info

    def _get_obs_state_dict(self, info: dict):
        """Override to filter left arm obs and apply normalization."""
        obs = super()._get_obs_state_dict(info)
        
        if self.single_arm_mode and "agent" in obs:
            # Remove left arm (so101-0) from agent observations
            # Right arm is so101-1 (larger X coordinate)
            if "so101-0" in obs["agent"]:
                del obs["agent"]["so101-0"]
        
        # Apply agent state normalization
        if self.obs_normalize_enabled and "agent" in obs:
            qvel_clip = torch.tensor(self.qvel_clip, device=self.device)
            
            for agent_key in obs["agent"]:
                agent_obs = obs["agent"][agent_key]
                
                # Get raw qpos for potential relative calculation
                raw_qpos = agent_obs.get("qpos", None)
                
                # qpos: divide by π → approximately [-1, 1] for typical joint ranges
                if raw_qpos is not None:
                    agent_obs["qpos"] = raw_qpos / self.qpos_scale
                
                # target_qpos handling based on include_target_qpos setting
                if "controller" in agent_obs and "target_qpos" in agent_obs["controller"]:
                    target_qpos = agent_obs["controller"]["target_qpos"]
                    
                    if self.include_target_qpos == "relative" and raw_qpos is not None:
                        # Replace with tracking error: (target_qpos - qpos) / action_bounds
                        tracking_error = target_qpos - raw_qpos
                        
                        # Normalize by action bounds if available, otherwise fall back to qpos_scale
                        if self.obs_action_bounds is not None:
                            # Dynamically get joint order from robot to avoid scrambling
                            active_joints = self.right_arm.robot.get_active_joints()
                            joint_names = [j.name for j in active_joints]
                            bounds_list = [self.obs_action_bounds.get(j, 0.1) for j in joint_names]
                            bounds = torch.tensor(bounds_list, device=self.device)
                            agent_obs["controller"]["target_qpos"] = tracking_error / bounds
                        else:
                            agent_obs["controller"]["target_qpos"] = tracking_error / self.qpos_scale
                    elif self.include_target_qpos:
                        # Include normalized target_qpos
                        agent_obs["controller"]["target_qpos"] = target_qpos / self.qpos_scale
                    else:
                        # Exclude target_qpos entirely
                        del agent_obs["controller"]["target_qpos"]
                        # Remove empty controller dict
                        if not agent_obs["controller"]:
                            del agent_obs["controller"]
                
                # qvel: clip and normalize
                if "qvel" in agent_obs:
                    qvel = agent_obs["qvel"]
                    qvel_clipped = torch.clamp(qvel, -qvel_clip, qvel_clip)
                    agent_obs["qvel"] = qvel_clipped / qvel_clip
        
        return obs

    def _get_obs_extra(self, info: dict):
        """Return extra observations (state-based).
        
        Includes privileged information like object poses and relative vectors
        to facilitate state-based training.
        """
        obs = dict()
        
        # Helper for position normalization: (pos - mean) / std with optional clipping
        def normalize_pos(pos, norm_config):
            mean = torch.tensor(norm_config["mean"], device=self.device)
            std = torch.tensor(norm_config["std"], device=self.device)
            normalized = (pos - mean) / std
            # Optional clipping to ±clip_with_std
            clip_std = norm_config.get("clip_with_std", None)
            if clip_std is not None:
                normalized = torch.clamp(normalized, -clip_std, clip_std)
            return normalized
        
        # 1. Object State
        red_cube_pos = self.red_cube.pose.p
        obs["red_cube_rot"] = self.red_cube.pose.q
        
        green_cube_pos = None
        if self.green_cube is not None:
            green_cube_pos = self.green_cube.pose.p
            obs["green_cube_rot"] = self.green_cube.pose.q

        # 2. End-Effector (TCP) State (Right Arm) - using agent.tcp_pos (fingertip midpoint)
        agent = self.right_arm
        
        tcp_pos = agent.tcp_pos  # Uses new fingertip-based calculation
        tcp_pose = agent.tcp_pose
        
        # 2a. is_grasped observation (optional, config controlled)
        if self.include_is_grasped:
            is_grasped = agent.is_grasping(
                self.red_cube, 
                min_force=self.grasp_min_force, 
                max_angle=self.grasp_max_angle
            )
            # Convert bool to -1/1 for better neural network input
            obs["is_grasped"] = is_grasped.float() * 2 - 1
        
        # 2b. TCP orientation (quaternion, optional)
        if self.include_tcp_orientation:
            obs["tcp_orientation"] = tcp_pose.q
        
        # 3. Relative State (Critical for RL efficiency)
        # TCP to Red Cube - apply clip + normalize
        tcp_to_red = red_cube_pos - tcp_pos
        if self.obs_normalize_enabled:
            clip_val = self.relative_pos_clip
            tcp_to_red = torch.clamp(tcp_to_red, -clip_val, clip_val) / clip_val
        obs["tcp_to_red_pos"] = tcp_to_red
        
        # Red to Green (for stack task)
        if self.task == "stack" and green_cube_pos is not None:
            red_to_green = green_cube_pos - red_cube_pos
            if self.obs_normalize_enabled:
                clip_val = self.relative_pos_clip
                red_to_green = torch.clamp(red_to_green, -clip_val, clip_val) / clip_val
            obs["red_to_green_pos"] = red_to_green
        
        # 4. Absolute positions (controlled by include_abs_pos: list, bool, or false)
        # Convert to list for uniform handling
        abs_pos_list = self.include_abs_pos
        if abs_pos_list is True:
            abs_pos_list = ["tcp_pos", "red_cube_pos", "green_cube_pos"]
        elif abs_pos_list is False or abs_pos_list is None:
            abs_pos_list = []
        
        if "tcp_pos" in abs_pos_list:
            if self.obs_normalize_enabled:
                obs["tcp_pos"] = normalize_pos(tcp_pos, self.tcp_pos_norm)
            else:
                obs["tcp_pos"] = tcp_pos
                
        if "red_cube_pos" in abs_pos_list:
            if self.obs_normalize_enabled:
                obs["red_cube_pos"] = normalize_pos(red_cube_pos, self.red_cube_pos_norm)
            else:
                obs["red_cube_pos"] = red_cube_pos
                
        if "green_cube_pos" in abs_pos_list and green_cube_pos is not None:
            if self.obs_normalize_enabled:
                obs["green_cube_pos"] = normalize_pos(green_cube_pos, self.green_cube_pos_norm)
            else:
                obs["green_cube_pos"] = green_cube_pos
            
        return obs


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
        
        # Base y-position, randomized if domain_randomization is enabled
        if self.domain_randomization:
            left_y = np.random.uniform(0.01, 0.03)
            right_y = np.random.uniform(0.01, 0.03)
        else:
            left_y = 0.02
            right_y = 0.02
        
        agent_poses = [
            sapien.Pose(p=[0.119, left_y, 0], q=rotation),  # Left Robot
            sapien.Pose(p=[0.481, right_y, 0], q=rotation)   # Right Robot
        ]
        
        # Enable per-env building for joint randomization
        if self.domain_randomization:
            super()._load_agent(options, agent_poses, build_separate=True)
            self._randomize_robot_properties()
        else:
            super()._load_agent(options, agent_poses)
        
        # Create agents dict for easy access by name
        self._agents_dict = {
            "left": self.agent.agents[0],    # so101-0
            "right": self.agent.agents[1],   # so101-1
        }
    
    @property
    def agents_dict(self):
        """Get agents as a dict for intuitive access: agents_dict['left'] or agents_dict['right']."""
        return self._agents_dict
    
    @property
    def right_arm(self):
        """Shortcut to access the right arm agent."""
        return self._agents_dict["right"]
    
    @property
    def left_arm(self):
        """Shortcut to access the left arm agent."""
        return self._agents_dict["left"]

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
        
        # Green cube: only load for stack and sort tasks
        if self.task == "lift":
            # Lift task: no green cube needed
            self.green_cube = None
        elif self.task == "sort":
            # Sort task: green cube is 1cm
            self.green_cube = self._build_cube(
                name="green_cube",
                half_size=0.005,
                base_color=[0, 1, 0, 1],
                default_pos=[0.497, 0.30, 0.005]
            )
        else:  # stack
            # Stack task: green cube is 3cm
            self.green_cube = self._build_cube(
                name="green_cube",
                half_size=0.015,
                base_color=[0, 1, 0, 1],
                default_pos=[0.497, 0.30, 0.015]
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
        """Initialize episode with randomized object positions and robot poses."""
        with torch.device(self.device):
            b = len(env_idx)
            
            # Initialize robot poses with noise around zero
            self._initialize_robot_poses(b, env_idx)
            
            if self.task == "lift":
                # Red cube random in configured spawn_bounds (or default grid)
                spawn_grid = self.spawn_bounds if self.spawn_bounds else self.grid_bounds["right"]
                red_pos = self._random_grid_position(b, spawn_grid, z=0.015)
                self.red_cube.set_pose(Pose.create_from_pq(p=red_pos))
                
                # Store initial cube XY for horizontal penalty in reward
                if not hasattr(self, 'initial_cube_xy'):
                    self.initial_cube_xy = torch.zeros(self.num_envs, 2, device=self.device)
                self.initial_cube_xy[env_idx] = red_pos[:, :2]
                
                # Reset stable hold counter for success condition
                if not hasattr(self, 'lift_hold_counter'):
                    self.lift_hold_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
                self.lift_hold_counter[env_idx] = 0
                
                # Reset prev_action for action rate penalty (set to None to skip first step penalty)
                # Note: prev_action is per-env, so we just clear the whole tensor on any reset
                self.prev_action = None
                
            elif self.task == "stack":
                # Both cubes in Right Grid, non-overlapping
                # Minimum distance: 3cm * sqrt(2) ≈ 4.3cm (diagonal of cube)
                min_dist = 0.043
                red_pos = self._random_grid_position(b, self.grid_bounds["right"], z=0.015)
                green_pos = self._random_grid_position(b, self.grid_bounds["right"], z=0.015)
                # Ensure minimum distance (retry until no collision)
                for _ in range(100):  # Generous retry limit
                    dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
                    overlap = dist < min_dist
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

    def _initialize_robot_poses(self, batch_size: int, env_idx: torch.Tensor):
        """Initialize robot poses with zero + small noise.
        
        Uses zero qpos as base and adds small Gaussian noise to introduce
        variation in initial configurations while keeping the pose valid.
        """
        # Noise standard deviation per joint (radians)
        # Smaller noise for arm joints (0-4), larger for gripper (5)
        qpos_noise_std = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.2], device=self.device)
        
        for agent in self.agent.agents:
            # Zero base pose
            zero_qpos = torch.zeros((batch_size, 6), device=self.device)
            
            # Add Gaussian noise
            noise = torch.randn_like(zero_qpos) * qpos_noise_std
            qpos = zero_qpos + noise
            
            # Clamp to safe joint limits (approximate, from URDF)
            # More conservative than actual limits to ensure valid poses
            qpos_lower = torch.tensor([-0.1, -2.0, -1.5, -1.5, -1.5, -1.0], device=self.device)
            qpos_upper = torch.tensor([1.5, 2.0, 1.5, 1.5, 1.5, 0.5], device=self.device)
            qpos = torch.clamp(qpos, qpos_lower, qpos_upper)
            
            agent.robot.set_qpos(qpos)

    def _random_grid_position(self, batch_size: int, grid: dict, z: float) -> torch.Tensor:
        """Generate random positions within a grid."""
        x = torch.rand(batch_size, device=self.device) * (grid["x_max"] - grid["x_min"]) + grid["x_min"]
        y = torch.rand(batch_size, device=self.device) * (grid["y_max"] - grid["y_min"]) + grid["y_min"]
        z_tensor = torch.full((batch_size,), z, device=self.device)
        return torch.stack([x, y, z_tensor], dim=1)

    def evaluate(self):
        """Evaluate success/fail based on task."""
        if self.task == "lift":
            return self._evaluate_lift()
        elif self.task == "stack":
            return self._evaluate_stack()
        elif self.task == "sort":
            return self._evaluate_sort()
        return {}

    def _evaluate_lift(self):
        """Lift: red cube >= lift_target for stable_hold_time seconds.
        
        If stable_hold_time=0, success is instant (cube just needs to be above threshold).
        Otherwise, cube must stay above threshold for stable_hold_steps consecutive steps.
        """
        red_z = self.red_cube.pose.p[:, 2]
        is_above = red_z >= self.lift_target
        
        if self.stable_hold_steps <= 0:
            # Instant success mode (backward compatible)
            success = is_above
        else:
            # Stable hold mode: increment counter when above, reset when below
            if not hasattr(self, 'lift_hold_counter'):
                self.lift_hold_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
            
            # Increment counter where above threshold
            self.lift_hold_counter = torch.where(
                is_above,
                self.lift_hold_counter + 1,
                torch.zeros_like(self.lift_hold_counter)  # Reset if below
            )
            
            # Success when counter reaches required hold steps
            success = self.lift_hold_counter >= self.stable_hold_steps
            
        # Failure conditions
        fallen_threshold = -0.05
        red_fallen = self.red_cube.pose.p[:, 2] < fallen_threshold
        fail = red_fallen
        
        if self.fail_bounds is not None:
            red_pos = self.red_cube.pose.p
            out_of_bounds = (
                (red_pos[:, 0] < self.fail_bounds["x_min"]) |
                (red_pos[:, 0] > self.fail_bounds["x_max"]) |
                (red_pos[:, 1] < self.fail_bounds["y_min"]) |
                (red_pos[:, 1] > self.fail_bounds["y_max"])
            )
            fail = fail | out_of_bounds
            
        # Ensure success is False if already failed
        success = success & (~fail)
        
        return {
            "success": success, 
            "fail": fail,
            "red_height": red_z, 
            "hold_steps": self.lift_hold_counter if hasattr(self, 'lift_hold_counter') else 0
        }

    def _evaluate_stack(self):
        """Stack: red cube on top of green cube, stable on table."""
        red_pos = self.red_cube.pose.p
        green_pos = self.green_cube.pose.p
        
        # Check green cube is on table (z ~ 1.5cm for 3cm cube)
        green_on_table = (green_pos[:, 2] > 0.010) & (green_pos[:, 2] < 0.020)
        
        # Check if red is above green (z difference ~ 3cm = cube size, allow ±0.5cm)
        z_diff = red_pos[:, 2] - green_pos[:, 2]
        z_ok = (z_diff > 0.025) & (z_diff < 0.035)
        
        # Check xy alignment (within 1.5cm for stability)
        xy_dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
        xy_ok = xy_dist < 0.015
        
        success = green_on_table & z_ok & xy_ok
        # Failure conditions
        fallen_threshold = -0.05
        red_fallen = self.red_cube.pose.p[:, 2] < fallen_threshold
        green_fallen = self.green_cube.pose.p[:, 2] < fallen_threshold
        fail = red_fallen | green_fallen
        
        success = success & (~fail)
        
        return {
            "success": success, 
            "fail": fail,
            "green_on_table": green_on_table, 
            "z_diff": z_diff, 
            "xy_dist": xy_dist
        }

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
        # Failure conditions
        fallen_threshold = -0.05
        red_fallen = self.red_cube.pose.p[:, 2] < fallen_threshold
        green_fallen = self.green_cube.pose.p[:, 2] < fallen_threshold
        fail = red_fallen | green_fallen
        
        success = success & (~fail)
        
        return {
            "success": success, 
            "fail": fail,
            "red_in_right": red_in_right, 
            "green_in_left": green_in_left
        }
    # ==================== Dense Reward Functions ====================
    
    def compute_dense_reward(self, obs, action, info):
        """Compute dense reward based on task."""
        if self.task == "lift":
            return self._compute_lift_dense_reward(info, action)
        elif self.task == "stack":
            return self._compute_stack_dense_reward(info)
        elif self.task == "sort":
            return self._compute_sort_dense_reward(info)
        else:
            return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        """Compute normalized dense reward (scaled to roughly [0, 1])."""
        reward = self.compute_dense_reward(obs, action, info)
        # Normalize by maximum expected reward
        max_reward = 10.0  # Approximate max for success bonus
        return reward / max_reward

    def _get_gripper_pos(self):
        """Get the gripper reference position for the right arm.
        
        Applies two offsets to gripper_frame_link (fixed jaw tip):
        1. gripper_tip_offset: back along jaw (towards gripper_link)
        2. gripper_outward_offset: perpendicular, towards the moving jaw (where cube should be)
        
        Returns position where cube center should be when properly grasped.
        """
        # Right arm is agents[1] (so101-1, at X=0.481)
        right_agent = self.agent.agents[1]
        gripper_link = right_agent.robot.links_map.get("gripper_link")
        gripper_frame = right_agent.robot.links_map.get("gripper_frame_link")
        moving_jaw = right_agent.robot.links_map.get("moving_jaw_so101_v1_link")
        
        if gripper_frame is None:
            # Fallback
            if gripper_link is not None:
                return gripper_link.pose.p
            return right_agent.robot.links[-1].pose.p
        
        # Start from jaw tip
        ref_pos = gripper_frame.pose.p.clone()
        
        # Apply tip offset (back along jaw direction)
        if hasattr(self, 'gripper_tip_offset') and self.gripper_tip_offset != 0 and gripper_link is not None:
            jaw_direction = gripper_frame.pose.p - gripper_link.pose.p
            jaw_length = torch.norm(jaw_direction, dim=1, keepdim=True)
            jaw_unit = jaw_direction / (jaw_length + 1e-6)
            ref_pos = ref_pos - jaw_unit * self.gripper_tip_offset
        
        # Apply outward offset (towards moving jaw, perpendicular to fixed jaw)
        if hasattr(self, 'gripper_outward_offset') and self.gripper_outward_offset != 0 and moving_jaw is not None:
            outward_direction = moving_jaw.pose.p - gripper_frame.pose.p
            outward_length = torch.norm(outward_direction, dim=1, keepdim=True)
            outward_unit = outward_direction / (outward_length + 1e-6)
            ref_pos = ref_pos + outward_unit * self.gripper_outward_offset
        
        return ref_pos

    def _get_moving_jaw_pos(self):
        """Get the moving jaw reference position for the right arm.
        
        Uses calibrated local direction: (-0.2, -1, 0.23) normalized, scaled by 1.8
        Then applies offsets:
        - moving_jaw_tip_offset: back along the jaw direction
        - moving_jaw_outward_offset: along local -X towards cube center
        
        Returns position where cube center should be when properly grasped.
        """
        right_agent = self.agent.agents[1]
        moving_jaw = right_agent.robot.links_map.get("moving_jaw_so101_v1_link")
        
        if moving_jaw is None:
            # Fallback to fixed jaw
            return self._get_gripper_pos()
        
        # Get moving jaw base position and orientation
        moving_jaw_base = moving_jaw.pose.p  # [num_envs, 3]
        moving_jaw_quat = moving_jaw.pose.q  # [num_envs, 4] - SAPIEN uses [w, x, y, z]
        
        # Calibrated local direction to jaw tip: (-0.2, -1, 0.23) normalized
        import torch
        x_comp, y_comp, z_comp = -0.2, -1.0, 0.23
        local_forward = torch.tensor([x_comp, y_comp, z_comp], device=self.device, dtype=torch.float32)
        local_forward = local_forward / torch.norm(local_forward)
        
        # Convert quaternion to rotation matrix and apply to local_forward
        # SAPIEN quaternion: [w, x, y, z]
        w, x, y, z = moving_jaw_quat[:, 0], moving_jaw_quat[:, 1], moving_jaw_quat[:, 2], moving_jaw_quat[:, 3]
        
        # Rotation matrix from quaternion
        R00 = 1 - 2*(y**2 + z**2)
        R01 = 2*(x*y - w*z)
        R02 = 2*(x*z + w*y)
        R10 = 2*(x*y + w*z)
        R11 = 1 - 2*(x**2 + z**2)
        R12 = 2*(y*z - w*x)
        R20 = 2*(x*z - w*y)
        R21 = 2*(y*z + w*x)
        R22 = 1 - 2*(x**2 + y**2)
        
        # Apply rotation to local_forward: world_dir = R @ local_forward
        jaw_direction = torch.stack([
            R00 * local_forward[0] + R01 * local_forward[1] + R02 * local_forward[2],
            R10 * local_forward[0] + R11 * local_forward[1] + R12 * local_forward[2],
            R20 * local_forward[0] + R21 * local_forward[1] + R22 * local_forward[2],
        ], dim=1)  # [num_envs, 3]
        
        # Scale to reach jaw tip (calibrated: 1.8 * 0.045 = 0.081)
        scale = 1.8
        tip_dist = 0.045 * scale
        ref_pos = moving_jaw_base + jaw_direction * tip_dist
        
        # Apply tip offset (back along jaw direction)
        if hasattr(self, 'moving_jaw_tip_offset') and self.moving_jaw_tip_offset != 0:
            ref_pos = ref_pos - jaw_direction * self.moving_jaw_tip_offset
        
        # Apply outward offset (along local -X, towards cube center)
        if hasattr(self, 'moving_jaw_outward_offset') and self.moving_jaw_outward_offset != 0:
            local_minus_x = torch.tensor([-1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
            outward_dir = torch.stack([
                R00 * local_minus_x[0] + R01 * local_minus_x[1] + R02 * local_minus_x[2],
                R10 * local_minus_x[0] + R11 * local_minus_x[1] + R12 * local_minus_x[2],
                R20 * local_minus_x[0] + R21 * local_minus_x[1] + R22 * local_minus_x[2],
            ], dim=1)
            ref_pos = ref_pos + outward_dir * self.moving_jaw_outward_offset
        
        return ref_pos

    def _compute_lift_dense_reward(self, info, action=None):
        """Dense reward for Lift task.
        
        Components:
        1. approach: Encourage fixed jaw to approach cube center
        2. approach2: Encourage moving jaw to approach cube center
        3. horizontal_displacement: Cube XY displacement from initial position (meters)
        4. lift: Reward proportional to cube height
        5. action_rate: Penalize action changes (anti-jitter)
        6. success: Bonus when cube is lifted above threshold
        """
        # Get config values
        w = self.reward_weights
        thresholds = self.stage_thresholds
        
        cube_pos = self.red_cube.pose.p
        cube_height = cube_pos[:, 2]
        
        # Approach reward calculation based on approach_mode
        threshold = self.approach_threshold
        zero_point = self.approach_zero_point
        
        if self.approach_mode == "tcp_midpoint":
            # TCP midpoint mode: single distance from TCP center to cube (like reference implementation)
            tcp_pos = self.right_arm.tcp_pos
            distance = torch.norm(tcp_pos - cube_pos, dim=1)
            
            approach_reward = torch.where(
                distance < threshold,
                torch.ones_like(distance),
                torch.clamp(1.0 - (distance - threshold) / (zero_point - threshold), min=0.0)
            )
            # No approach2 in tcp_midpoint mode
            approach2_reward = torch.zeros_like(approach_reward)
        else:
            # Dual-point mode: separate fixed jaw and moving jaw approach rewards
            # 1. Approach reward (fixed jaw)
            gripper_pos = self._get_gripper_pos()
            distance = torch.norm(gripper_pos - cube_pos, dim=1)
            
            approach_reward = torch.where(
                distance < threshold,
                torch.ones_like(distance),
                torch.clamp(1.0 - (distance - threshold) / (zero_point - threshold), min=0.0)
            )
            
            # 2. Approach2 reward (moving jaw)
            moving_jaw_pos = self._get_moving_jaw_pos()
            distance2 = torch.norm(moving_jaw_pos - cube_pos, dim=1)
            threshold2 = self.approach2_threshold
            zero_point2 = self.approach2_zero_point
            
            approach2_reward = torch.where(
                distance2 < threshold2,
                torch.ones_like(distance2),
                torch.clamp(1.0 - (distance2 - threshold2) / (zero_point2 - threshold2), min=0.0)
            )
        
        # 3. Horizontal displacement: cube XY displacement from initial position (positive value)
        # initial_cube_xy is set during episode initialization
        # Only penalize displacement beyond threshold (allows small movements during grasping)
        if hasattr(self, 'initial_cube_xy'):
            raw_displacement = torch.norm(cube_pos[:, :2] - self.initial_cube_xy, dim=1)
            threshold = self.horizontal_displacement_threshold
            horizontal_displacement = torch.clamp(raw_displacement - threshold, min=0.0)
        else:
            horizontal_displacement = torch.zeros(self.num_envs, device=self.device)
        
        # 4. Lift reward: height of cube above baseline (subtract initial resting height)
        # Cube starts at half_size height (0.015m for 3cm cube)
        cube_baseline_height = 0.015  # half_size of red cube
        lift_height = cube_height - cube_baseline_height
        if self.lift_max_height is not None and self.lift_max_height > 0:
            lift_reward = torch.clamp(lift_height, min=0.0, max=self.lift_max_height) / self.lift_max_height
        else:
            lift_reward = torch.clamp(lift_height, min=0.0)
        
        # 5. Action rate penalty: ||action_t - action_{t-1}||^2
        if action is not None and w.get("action_rate", 0.0) != 0:
            # Convert action to tensor if needed
            if isinstance(action, dict):
                # Dict action space - concatenate all values
                action_tensor = torch.cat([v.flatten(start_dim=1) for v in action.values()], dim=1)
            else:
                action_tensor = action.flatten(start_dim=1) if action.dim() > 1 else action.unsqueeze(0)
            
            if self.prev_action is None:
                # First step: no penalty
                action_rate = torch.zeros(self.num_envs, device=self.device)
            else:
                # Compute squared L2 norm of action difference
                action_diff = action_tensor - self.prev_action
                action_rate = torch.sum(action_diff ** 2, dim=1)
            
            # Update prev_action for next step
            self.prev_action = action_tensor.clone()
        else:
            action_rate = torch.zeros(self.num_envs, device=self.device)
        
        # 6. Success bonus: cube lifted above threshold for stable_hold_time
        success = info.get("success", torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
        success_bonus = success.float()
        
        # 7. Fail penalty: cube out of bounds or fallen
        fail = info.get("fail", torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
        fail_penalty = fail.float()
        
        # 8. Grasp reward: bonus for successfully grasping the cube
        agent = self.right_arm
        
        is_grasped = agent.is_grasping(
            self.red_cube,
            min_force=self.grasp_min_force,
            max_angle=self.grasp_max_angle
        )
        grasp_reward = is_grasped.float()  # 0 or 1
        
        # 9. Apply gated lift reward if configured
        # Only give lift reward when grasping the cube
        if self.gate_lift_with_grasp:
            effective_lift_reward = lift_reward * is_grasped.float()
        else:
            effective_lift_reward = lift_reward
        
        # Weighted sum (use negative weights for penalties)
        # In dual_point mode, approach weight is used for both approach and approach2
        reward = (w["approach"] * approach_reward +
                  w["approach"] * approach2_reward +  # Same weight as approach in dual_point mode
                  w.get("grasp", 0.0) * grasp_reward +
                  w["horizontal_displacement"] * horizontal_displacement +
                  w["lift"] * effective_lift_reward + 
                  w.get("action_rate", 0.0) * action_rate +
                  w["success"] * success_bonus +
                  w["fail"] * fail_penalty)
        
        # Store reward components for logging (keep as GPU tensors to avoid sync)
        # Runner will call .item() only when logging is needed
        info["reward_components"] = {
            "approach": (w["approach"] * approach_reward).mean(),
            "grasp": (w.get("grasp", 0.0) * grasp_reward).mean(),
            "horizontal_displacement": (w["horizontal_displacement"] * horizontal_displacement).mean(),
            "lift": (w["lift"] * effective_lift_reward).mean(),
            "action_rate": (w.get("action_rate", 0.0) * action_rate).mean(),
        }
        # Only log approach2 in dual_point mode
        if self.approach_mode == "dual_point":
            info["reward_components"]["approach2"] = (w["approach"] * approach2_reward).mean()
        # Track success/fail/grasp counts (keep as GPU tensors)
        info["success_count"] = success.sum()
        info["fail_count"] = fail.sum()
        info["grasp_count"] = is_grasped.sum()
        
        return reward

    def _compute_stack_dense_reward(self, info):
        """Dense reward for Stack task.
        
        Components:
        1. reach_red: Approach red cube
        2. grasp_red: Grasp red cube
        3. lift_red: Lift red cube above green
        4. align: Align red over green (xy distance)
        5. place: Place red on green
        6. success_bonus
        """
        w_reach = 1.0
        w_grasp = 2.0
        w_lift = 2.0
        w_align = 3.0
        w_place = 5.0
        w_success = 10.0
        
        gripper_pos = self._get_gripper_pos()
        red_pos = self.red_cube.pose.p
        green_pos = self.green_cube.pose.p
        
        # 1. Reach reward
        dist_to_red = torch.norm(gripper_pos - red_pos, dim=1)
        reach_reward = 1.0 - torch.tanh(dist_to_red * 5.0)
        
        # 2. Grasp reward
        is_grasping = dist_to_red < 0.03
        grasp_reward = is_grasping.float()
        
        # 3. Lift reward (red above green)
        z_diff = red_pos[:, 2] - green_pos[:, 2]
        lift_reward = torch.clamp(z_diff, min=0.0, max=0.05)  # Cap at 5cm
        
        # 4. Alignment reward (xy distance between red and green)
        xy_dist = torch.norm(red_pos[:, :2] - green_pos[:, :2], dim=1)
        align_reward = 1.0 - torch.tanh(xy_dist * 10.0)
        
        # 5. Place reward (red on top of green)
        is_stacked = (z_diff > 0.025) & (z_diff < 0.04) & (xy_dist < 0.02)
        place_reward = is_stacked.float()
        
        # 6. Success bonus
        success = info.get("success", torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
        success_bonus = success.float()
        
        reward = (w_reach * reach_reward +
                  w_grasp * grasp_reward +
                  w_lift * lift_reward +
                  w_align * align_reward +
                  w_place * place_reward +
                  w_success * success_bonus)
        
        return reward

    def _compute_sort_dense_reward(self, info):
        """Dense reward for Sort task (placeholder - needs both arms)."""
        # For sort task, we need more complex logic with both arms
        # For now, use sparse reward
        success = info.get("success", torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
        return success.float() * 10.0
