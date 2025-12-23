#!/usr/bin/env python3
"""
Joint Optimizer for Robot Base Pose and Wrist Camera Hand-Eye Extrinsics.
Uses a dataset of frames and joint angles to align simulation with real-world.

Optimized Parameters (12 DoF):
- Base Pose w.r.t World (6 DoF)
- Hand-Eye Extrinsic w.r.t End-Effector (6 DoF)
"""

import os
import sys
import json
import numpy as np
import cv2
import argparse
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# Add project root to path for imports
sys.path.append(os.getcwd())

try:
    import gymnasium as gym
    from scripts.track1_env import Track1Env
    import torch
    from mani_skill.utils.structs.pose import Pose
except ImportError:
    print("Error: Could not import environment scripts or dependencies.")
    sys.exit(1)

# Wrist Camera Intrinsics (from SO101 FOV=50)
W, H = 640, 480
FOV = np.deg2rad(50)
F = (W / 2) / np.tan(FOV / 2)
K_WRIST = np.array([
    [F, 0, W/2],
    [0, F, H/2],
    [0, 0, 1]
], dtype=np.float64)

# Coordinate Conversion: Sapien to OpenCV
R_SAPIEN2OPENCV = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
], dtype=np.float64)

def get_grid_3d_points(num_samples_per_line: int = 40):
    """Same 3D points as used in front camera optimizer."""
    tape_half_width = 0.009
    w = tape_half_width
    x1, x2 = 0.204, 0.378
    y1, y2 = 0.150, 0.332
    z = 0.001
    
    points = []
    # Use center lines of the tapes
    # Row at y1
    for t in np.linspace(x1, x2 + 2*w, num_samples_per_line):
        points.append([t, y1 + w, z])
    # Col at x1
    for t in np.linspace(y1, y2 + 2*w, num_samples_per_line):
        points.append([x1 + w, t, z])
    # Col at x2
    for t in np.linspace(y1, y2 + 2*w, num_samples_per_line):
        points.append([x2 + w, t, z])
    # Row at y2 (partial left side)
    for t in np.linspace(x1, (x1+x2)/2, num_samples_per_line // 2):
        points.append([t, y2 + w, z])
        
    return np.array(points, dtype=np.float64)

def pose_to_matrix(pos, rot_euler):
    """6-DoF to 4x4 matrix."""
    T = np.eye(4)
    T[:3, 3] = pos
    R = Rotation.from_euler('xyz', rot_euler).as_matrix()
    T[:3, :3] = R
    return T

def matrix_to_pose(T):
    """4x4 matrix to 6-DoF."""
    pos = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3]).as_euler('xyz')
    return pos, rot

class WristOptimizer:
    def __init__(self, data_dir, visualize=False):
        self.data_dir = data_dir
        self.visualize = visualize
        
        # Load metadata
        meta_path = os.path.join(data_dir, 'frames_meta.json')
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Preprocess each image (Distance Transform)
        self.dt_maps = []
        for item in self.metadata:
            img_path = os.path.join(data_dir, item['image_file'])
            img = cv2.imread(img_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Black tape mask
            mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 75]))
            # Distance Transform
            dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
            self.dt_maps.append(dist)
            
        self.grid_points = get_grid_3d_points()
        
        # Setup environment for FK
        self.env = gym.make('Track1-v0', obs_mode='rgb', render_mode=None)
        self.env.reset()
        u_env = self.env.unwrapped
        
        # Explicitly find so101-0 (Right Robot)
        self.agent = None
        if hasattr(u_env.agent, 'agents'):
            for a in u_env.agent.agents:
                if a.uid == 'so101-0':
                    self.agent = a
                    break
            if self.agent is None:
                self.agent = u_env.agent.agents[0]
        else:
            self.agent = u_env.agent
            
        self.robot = self.agent.robot
        # Camera sensor (to get relative mount pose in Sapien)
        self.cam_sensor = u_env._sensors.get('wrist_camera_0') or self.agent.sensors.get('wrist_camera')
        
    def get_fk_matrices(self, qpos_list):
        """Precompute FK matrices for the wrist camera link relative to robot base."""
        fk_mats = []
        with torch.no_grad():
            for q in qpos_list:
                # Convert DEGREES from dataset to RADIANS for simulation
                q_rad = np.deg2rad(q)
                new_q = torch.tensor(q_rad, dtype=torch.float32, device=self.robot.get_qpos().device).reshape(1, 6)
                self.robot.set_qpos(new_q)
                # We need the mount pose RELATIVE to the robot base
                # Sapien link poses are in world frame.
                # Since robot base is currently at World(0,0,0) in our dummy env reset:
                mount_pose = self.cam_sensor.camera.get_global_pose()
                fk_mats.append(mount_pose.to_transformation_matrix().squeeze(0).cpu().numpy())
        return fk_mats

    def objective(self, params):
        # params: [base_x, base_y, base_z, base_r, base_p, base_y, 
        #          he_x, he_y, he_z, he_r, he_p, he_y]
        base_pos, base_rot = params[:3], params[3:6]
        he_pos, he_rot = params[6:9], params[9:12]
        
        T_base = pose_to_matrix(base_pos, base_rot)
        T_hand_eye = pose_to_matrix(he_pos, he_rot)
        
        total_loss = 0
        
        for i, T_fk in enumerate(self.fk_mats):
            # T_cam_world = T_base * T_fk * T_hand_eye
            T_cam_world = T_base @ T_fk @ T_hand_eye
            
            # World to Cam
            T_w2c = np.linalg.inv(T_cam_world)
            
            # Map Sapien Camera to OpenCV
            T_w2c_cv = np.eye(4)
            T_w2c_cv[:3, :3] = R_SAPIEN2OPENCV @ T_w2c[:3, :3]
            T_w2c_cv[:3, 3] = R_SAPIEN2OPENCV @ T_w2c[:3, 3]
            
            R_cv = T_w2c_cv[:3, :3]
            t_cv = T_w2c_cv[:3, 3]
            rvec, _ = cv2.Rodrigues(R_cv)
            
            # Project
            projs, _ = cv2.projectPoints(self.grid_points, rvec, t_cv, K_WRIST, None)
            projs = projs.reshape(-1, 2)
            
            # Distance Transform Loss
            dt = self.dt_maps[i]
            loss_i = 0
            valid_pts = 0
            for pt in projs:
                u, v = int(round(pt[0])), int(round(pt[1]))
                if 0 <= u < W and 0 <= v < H:
                    # Penalty: dist^2
                    loss_i += dt[v, u]**2
                    valid_pts += 1
            
            if valid_pts > 0:
                total_loss += loss_i / valid_pts
            else:
                total_loss += 10000 # Large penalty for off-screen
                
        return total_loss / len(self.fk_mats)

    def optimize(self):
        # Initial guess
        # Base: Robot is likely near (0,0,0) or (0.1, 0.25, 0)
        # Hand-eye: Identity (0,0,0, 0,0,0)
        initial_params = np.zeros(12)
        initial_params[1] = 0.25 # Assume centered on Y table
        
        print(f"Precomputing FK for {len(self.metadata)} frames...")
        qpos_list = [item['qpos'] for item in self.metadata]
        self.fk_mats = self.get_fk_matrices(qpos_list)
        
        print("Starting optimization (12 DoF)...")
        res = minimize(self.objective, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'disp': True})
        
        print("\nOptimization Results:")
        final_params = res.x
        base_pos, base_rot = final_params[:3], final_params[3:6]
        he_pos, he_rot = final_params[6:9], final_params[9:12]
        
        print(f"Base Pose (X,Y,Z): {base_pos}")
        print(f"Base Rot (R,P,Y): {base_rot}")
        print(f"Hand-Eye Post (X,Y,Z): {he_pos}")
        print(f"Hand-Eye Rot (R,P,Y): {he_rot}")
        print(f"Final Loss: {res.fun}")
        
        self.final_params = final_params
        
        # Save results
        results = {
            "base_pose": {
                "position": base_pos.tolist(),
                "rotation_euler_xyz": base_rot.tolist()
            },
            "hand_eye": {
                "position": he_pos.tolist(),
                "rotation_euler_xyz": he_rot.tolist()
            },
            "loss": res.fun
        }
        with open(os.path.join(self.data_dir, 'optimization_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        if self.visualize:
            self.visualize_results(final_params)

    def visualize_results(self, params):
        base_pos, base_rot = params[:3], params[3:6]
        he_pos, he_rot = params[6:9], params[9:12]
        T_base = pose_to_matrix(base_pos, base_rot)
        T_hand_eye = pose_to_matrix(he_pos, he_rot)
        
        out_dir = os.path.join(self.data_dir, 'optimization_visuals')
        os.makedirs(out_dir, exist_ok=True)
        
        for i, item in enumerate(self.metadata):
            img = cv2.imread(os.path.join(self.data_dir, item['image_file']))
            T_fk = self.fk_mats[i]
            T_cam = T_base @ T_fk @ T_hand_eye
            T_w2c = np.linalg.inv(T_cam)
            T_w2c_cv = np.eye(4)
            T_w2c_cv[:3, :3] = R_SAPIEN2OPENCV @ T_w2c[:3, :3]
            T_w2c_cv[:3, 3] = R_SAPIEN2OPENCV @ T_w2c[:3, 3]
            rvec, _ = cv2.Rodrigues(T_w2c_cv[:3, :3])
            tvec = T_w2c_cv[:3, 3]
            
            projs, _ = cv2.projectPoints(self.grid_points, rvec, tvec, K_WRIST, None)
            for pt in projs.reshape(-1, 2):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
                
            cv2.imwrite(os.path.join(out_dir, f"result_{i:04d}.png"), img)
        print(f"Visuals saved to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="wrist_calibration_data")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    
    opt = WristOptimizer(args.data, visualize=args.visualize)
    opt.optimize()

if __name__ == "__main__":
    main()
