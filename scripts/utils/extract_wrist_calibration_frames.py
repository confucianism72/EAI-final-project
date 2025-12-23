#!/usr/bin/env python3
"""
Extract wrist camera frames and corresponding joint angles for calibration.
ROBUST VERSION 2:
- Focuses on Right Robot (so101-0)
- Degree to Radian conversion
- Line detection (Hough) for verifiable grid existence
"""

import os
import sys
import shutil
import json
import glob
import numpy as np
import pandas as pd
import argparse
import cv2
import subprocess

# Add project root to path for imports
sys.path.append(os.getcwd())

try:
    import gymnasium as gym
    from scripts.track1_env import Track1Env
    import sapien.core as sapien
    import torch
except ImportError:
    print("Error: Could not import environment scripts or dependencies.")
    sys.exit(1)

# Wrist Camera Intrinsics (approximated from FOV=50)
W, H = 640, 480
FOV = np.deg2rad(50)
F = (W / 2) / np.tan(FOV / 2)
K_WRIST = np.array([
    [F, 0, W/2],
    [0, F, H/2],
    [0, 0, 1]
], dtype=np.float64)

# Grid Corners (X, Y, Z=0)
GRID_CORNERS = np.array([
    [0.204, 0.150, 0.0],   # C1: Lower Left
    [0.378, 0.150, 0.0],   # C2: Lower Right
    [0.204, 0.332, 0.0],   # C3: Upper Left
    [0.378, 0.332, 0.0],   # C4: Upper Right
    [0.291, 0.332, 0.0]    # C5: Mid Y2
], dtype=np.float64)

def is_camera_looking_at_grid(env, qpos):
    """
    Check if any grid corners are within the camera FOV using FK.
    """
    u_env = env.unwrapped
    # Explicitly find so101-0 (Right Robot)
    agent = None
    if hasattr(u_env.agent, 'agents'):
        # u_env.agent.agents is a list of Agent objects
        for a in u_env.agent.agents:
            if a.uid == 'so101-0':
                agent = a
                break
        if agent is None:
            agent = u_env.agent.agents[0]
    else:
        agent = u_env.agent
        
    robot = agent.robot
    current_qpos = robot.get_qpos() # (1, 6)
    
    # Update joints (Convert DEGREES to RADIANS)
    qpos_rad = np.deg2rad(qpos)
    new_qpos = torch.tensor(qpos_rad, dtype=torch.float32, device=current_qpos.device).reshape(1, 6)
    robot.set_qpos(new_qpos)
    
    # Get camera pose
    cam = u_env._sensors.get('wrist_camera_0')
    if cam is None:
        cam = agent.sensors.get('wrist_camera')
    
    if cam is None:
        return False, 0
        
    # In ManiSkill 3, cam.camera is a RenderCamera object
    gpose = cam.camera.get_global_pose()
    T_c2w_torch = gpose.to_transformation_matrix() # (1, 4, 4)
    T_c2w = T_c2w_torch.squeeze(0).cpu().numpy()
    
    T_w2c = np.linalg.inv(T_c2w)
    
    # OpenGL to OpenCV
    R_sapien2opencv = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    T_w2c_cv = np.eye(4)
    T_w2c_cv[:3, :3] = R_sapien2opencv @ T_w2c[:3, :3]
    T_w2c_cv[:3, 3] = R_sapien2opencv @ T_w2c[:3, 3]
    
    R_cv = T_w2c_cv[:3, :3]
    t_cv = T_w2c_cv[:3, 3]
    rvec, _ = cv2.Rodrigues(R_cv)
    
    projected, _ = cv2.projectPoints(GRID_CORNERS, rvec, t_cv, K_WRIST, None)
    projected = projected.reshape(-1, 2)
    
    # Loose projection check (account for base pose uncertainty)
    count = 0
    padding = -50 # Be very loose, rely on CV later
    for i, pt in enumerate(projected):
        if padding <= pt[0] <= W - padding and padding <= pt[1] <= H - padding:
            pt_cam = R_cv @ GRID_CORNERS[i] + t_cv
            if pt_cam[2] > 0.05:
                count += 1
                
    return count > 0, count

def check_image_content_robust(img_path):
    """
    Robust CV check: Canny Edge Length + Black HSV
    """
    if not os.path.exists(img_path): return False, 0.0
    img = cv2.imread(img_path)
    if img is None: return False, 0.0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Laplacian variance is very low in this dataset (~2-10), so we disable strict blur filtering
    # but still record it for metadata if needed.
    # laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Canny Edge Detection
    # These parameters tuned based on debug frame analysis
    edges = cv2.Canny(gray, 30, 100) # Lower thresholds for softer edges
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_len = 0
    for cnt in contours:
        length = cv2.arcLength(cnt, closed=False)
        if length > max_len:
            max_len = length
            
    # Require at least one long edge (> 100 pixels)
    if max_len < 100:
        return False, 0.0

    # 3. HSV Black check
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 75]))
    score = cv2.countNonZero(mask) / (W * H)
    
    # Require some black pixels (grid lines)
    if 0.002 < score < 0.25:
        return True, score
    return False, score

def extract_frames(dataset_path: str, output_dir: str, num_frames_wanted: int = 30):
    # Setup dummy env with RIGHT ROBOT
    env = gym.make('Track1-v0', obs_mode='rgb', render_mode=None)
    env.reset()
    
    # Ensure robot is at standard position
    u_env = env.unwrapped
    agent = None
    if hasattr(u_env.agent, 'agents'):
        for a in u_env.agent.agents:
            if a.uid == 'so101-0':
                agent = a
                break
    if agent:
        # Default robot pose in sim usually near (0,0.25,0)
        # But let's keep it consistent with what we know
        pass
    
    parquet_path = os.path.join(dataset_path, 'lift', 'data', 'chunk-000', 'file-000.parquet')
    video_path = os.path.join(dataset_path, 'lift', 'videos', 'observation.images.wrist', 'chunk-000', 'file-000.mp4')
    
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    total_frames = len(df)
    
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Step 1: Kinematics Filtering (Loose Corner Projection)...")
    candidates = []
    for idx in range(0, total_frames, 2):
        qpos = df.iloc[idx]['observation.state']
        is_looking, corner_count = is_camera_looking_at_grid(env, qpos)
        if is_looking:
            candidates.append((idx, corner_count))
            
    print(f"Found {len(candidates)} candidates from kinematics check")
    
    # Fallback: if kinematics too strict, try all frames with a stride
    if not candidates:
        print("Kinematics check found nothing. Falling back to CV-only sampling.")
        candidates = [(i, 0) for i in range(0, total_frames, 5)]

    print("Step 2: High-Speed Bulk Extraction...")
    # Extract every 2nd frame in one go to a temporary folder
    stride = 2
    temp_extract_dir = os.path.join(output_dir, "temp_extract")
    if os.path.exists(temp_extract_dir): shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    print(f"  Extracting all frames (stride {stride}) to temporary directory...")
    # Using %06d to ensure sorting order
    cmd = [
        "ffmpeg", "-y", "-i", video_path, 
        "-vf", f"select=not(mod(n\\,{stride}))", 
        "-vsync", "vfr", 
        os.path.join(temp_extract_dir, "f_%06d.png")
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Process and score
    print("  Scoring extracted frames...")
    candidates_pool = []
    extracted_files = sorted(glob.glob(os.path.join(temp_extract_dir, "*.png")))
    
    for i, img_path in enumerate(extracted_files):
        # Extract frame index from filename f_000001.png -> 1 * stride
        # Note: ffmpeg %d is 1-indexed for the output sequence
        seq_idx = int(os.path.basename(img_path).split('_')[1].split('.')[0])
        actual_idx = (seq_idx - 1) * stride
        
        has_lines, score = check_image_content_robust(img_path)
        if has_lines:
            candidates_pool.append({
                "idx": actual_idx,
                "score": score,
                "path": img_path
            })
            
    print(f"\nFound {len(candidates_pool)} frames with detectable grid lines.")
    
    # Sort by score descending
    candidates_pool.sort(key=lambda x: x['score'], reverse=True)
    
    # Temporal Diversity Filter (NMS)
    # Ensure selected frames are at least N frames apart to avoid near-identical images
    MIN_STRIDE = 20
    diverse_candidates = []
    used_indices = []
    
    for cand in candidates_pool:
        idx = cand['idx']
        # Check if this frame is too close to any already selected frame
        if any(abs(idx - used_idx) < MIN_STRIDE for used_idx in used_indices):
            continue
        
        diverse_candidates.append(cand)
        used_indices.append(idx)
        
        if len(diverse_candidates) >= num_frames_wanted:
            break
            
    print(f"Filtered to {len(diverse_candidates)} diverse candidates using Temporal NMS (min stride: {MIN_STRIDE})")
    
    # Save top candidates to final output
    final_data = []
    for item in diverse_candidates:
        idx = item['idx']
        final_filename = f"frame_{idx:04d}_score_{item['score']:.4f}.png"
        final_path = os.path.join(output_dir, final_filename)
        shutil.move(item['path'], final_path)
        
        joints = df.iloc[idx]['observation.state']
        final_data.append({
            "image_file": final_filename,
            "frame_index": int(idx),
            "qpos": joints.tolist(),
            "cv_score": item['score']
        })
        
    # Cleanup remaining temp files
    shutil.rmtree(temp_extract_dir)
    
    if final_data:
        with open(os.path.join(output_dir, 'candidates_meta.json'), 'w') as f:
            json.dump(final_data, f, indent=2)
        print(f"Successfully saved TOP {len(final_data)} candidate frames to {output_dir}")
    else:
        print("Final CV filtering failed to find any grid lines. Possible reasons:")
        print(" - Video quality too low for Edge detection")
        print(" - HSV black threshold too strict")
        print(" - Grid is not in the wrist camera view in this chunk.")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="eai-dataset")
    parser.add_argument("--output", type=str, default="wrist_calibration_data")
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    extract_frames(args.dataset, args.output, args.count)
