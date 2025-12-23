import numpy as np
import copy
import sapien
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.sensors.camera import CameraConfig

@register_agent()
class SO101(BaseAgent):
    uid = "so101"
    # Use absolute path to the asset provided in the workspace
    urdf_path = "/home/admin/Desktop/eai-final-project/eai-2025-fall-final-project-reference-scripts/assets/SO101/so101.urdf"
    
    # Per-joint action bounds (radians) - can be overridden before environment creation
    # Default values based on trajectory analysis (99th percentile + buffer)
    # Keys must match joint names: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    action_bounds_single_arm = {
        "shoulder_pan": 0.044,    # ~2.5 degrees
        "shoulder_lift": 0.087,   # ~5.0 degrees
        "elbow_flex": 0.070,      # ~4.0 degrees
        "wrist_flex": 0.044,      # ~2.5 degrees
        "wrist_roll": 0.026,      # ~1.5 degrees
        "gripper": 0.070,         # ~4.0 degrees
    }
    
    action_bounds_dual_arm = {
        "shoulder_pan": 0.070,    # ~4.0 degrees
        "shoulder_lift": 0.122,   # ~7.0 degrees
        "elbow_flex": 0.122,      # ~7.0 degrees
        "wrist_flex": 0.070,      # ~4.0 degrees
        "wrist_roll": 0.044,      # ~2.5 degrees
        "gripper": 0.113,         # ~6.5 degrees
    }
    
    # Active mode: 'single' or 'dual' - set by environment before agent creation
    active_mode = "single"
    
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)
        ),
        link=dict(
            Fixed_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            Moving_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, -1.5708, 1.5708, 0.66, 0, -1.1]),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 6),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
    )

    arm_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    gripper_joint_names = [
        "gripper",
    ]
    
    @property
    def _active_action_bounds(self):
        """Get active action bounds based on mode."""
        if self.active_mode == "dual":
            return self.action_bounds_dual_arm
        return self.action_bounds_single_arm

    @property
    def _sensor_configs(self):
        print("DEBUG: SO101._sensor_configs called")
        return [
            CameraConfig(
                "wrist_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=640,
                height=480,
                fov=np.deg2rad(50),
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"]
            )
        ]

    @property
    def _controller_configs(self):
        # Get joint names
        all_joint_names = [joint.name for joint in self.robot.active_joints]
        arm_joint_names = all_joint_names[:5]  # First 5 joints are arm
        gripper_joint_names = all_joint_names[5:]  # Last joint is gripper
        
        # Get per-joint action bounds based on active mode
        bounds = self._active_action_bounds
        joint_order = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        lower = [-bounds[j] for j in joint_order]
        upper = [bounds[j] for j in joint_order]
        
        # Absolute position control (no normalization)
        pd_joint_pos = PDJointPosControllerConfig(
            all_joint_names,
            lower=None,
            upper=None,
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            normalize_action=False,
        )

        # Delta position control from CURRENT position (using configurable bounds)
        pd_joint_delta_pos = PDJointPosControllerConfig(
            all_joint_names,
            lower,
            upper,
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            use_delta=True,
            use_target=False,
            normalize_action=True,
        )

        # Delta position control from TARGET position (recommended for sim2real)
        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        # End effector position control (arm) + joint control (gripper)
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            arm_joint_names,
            pos_lower=-0.02,
            pos_upper=0.02,
            stiffness=[1e3] * 5,
            damping=[1e2] * 5,
            force_limit=100,
            ee_link="gripper_link",
            urdf_path=self.urdf_path,
            use_delta=True,
            use_target=True,
            normalize_action=True,
        )
        
        gripper_pd_joint_delta_pos = PDJointPosControllerConfig(
            gripper_joint_names,
            lower=[-0.2],
            upper=[0.2],
            stiffness=[1e3],
            damping=[1e2],
            force_limit=100,
            use_delta=True,
            normalize_action=True,
        )
        
        pd_ee_delta_pos = dict(
            arm=arm_pd_ee_delta_pos,
            gripper=gripper_pd_joint_delta_pos,
        )

        controller_configs = dict(
            pd_joint_pos=pd_joint_pos,
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
            pd_ee_delta_pos=pd_ee_delta_pos,
        )
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        # Map links for gripper logic, assuming similar structure to SO100
        # If names differ in URDF, this might need adjustment
        try:
            self.gripper_link_names = [
                self.robot.links_map["gripper_link"].name,
                self.robot.links_map["moving_jaw_so101_v1_link"].name,
            ]
        except KeyError:
            print("Warning: Could not find gripper links (gripper_link, moving_jaw_so101_v1_link) in SO101 URDF.")
            pass
