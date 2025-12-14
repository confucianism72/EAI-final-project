#!/usr/bin/env python3
"""Visualize camera output with distortion processing.

This script captures sensor images and displays them with proper distortion processing,
which cannot be done through the native GUI viewer.

Usage:
    python -m scripts.view_camera --camera-mode distorted
    python -m scripts.view_camera --camera-mode distort-twice
    python -m scripts.view_camera --camera-mode direct_pinhole
"""

import argparse
import gymnasium as gym
import cv2
import numpy as np
from scripts.track1_env import Track1Env


def main():
    parser = argparse.ArgumentParser(description="View processed camera output")
    parser.add_argument("--task", type=str, default="lift", 
                        choices=["lift", "stack", "sort"],
                        help="Task type")
    parser.add_argument("--camera-mode", type=str, default="distorted",
                        choices=["distorted", "distort-twice", "direct_pinhole"],
                        help="Camera output mode")
    parser.add_argument("--save", type=str, default=None,
                        help="Save image to file instead of displaying")
    args = parser.parse_args()
    
    print(f"Creating environment with camera_mode={args.camera_mode}")
    
    env = gym.make(
        "Track1-v0",
        render_mode=None,  # No native rendering
        obs_mode="rgbd",   # Include RGB in observations
        reward_mode="none",
        task=args.task,
        domain_randomization=False,
        camera_mode=args.camera_mode,
        num_envs=1,
    )
    
    obs, _ = env.reset()
    
    # Extract front_camera RGB (could be in 'image' or 'sensor_data')
    rgb = None
    if "sensor_data" in obs and "front_camera" in obs["sensor_data"]:
        if "rgb" in obs["sensor_data"]["front_camera"]:
            rgb = obs["sensor_data"]["front_camera"]["rgb"]
    elif "image" in obs and "front_camera" in obs["image"]:
        if "rgb" in obs["image"]["front_camera"]:
            rgb = obs["image"]["front_camera"]["rgb"]
    
    if rgb is not None:
        # Convert to numpy if tensor
        if hasattr(rgb, 'cpu'):
            rgb = rgb.cpu().numpy()
        
        # Remove batch dimension if present
        if len(rgb.shape) == 4:
            rgb = rgb[0]
        
        # Convert to uint8 if float
        if rgb.dtype in [np.float32, np.float64]:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        print(f"Image shape: {bgr.shape}")
        
        if args.save:
            cv2.imwrite(args.save, bgr)
            print(f"Saved to {args.save}")
        else:
            # Display with OpenCV
            window_name = f"Camera View ({args.camera_mode})"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)
            
            print("Press 'q' to quit, 'r' to reset, 's' to save")
            
            while True:
                cv2.imshow(window_name, bgr)
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    obs, _ = env.reset()
                    if "sensor_data" in obs:
                        rgb = obs["sensor_data"]["front_camera"]["rgb"]
                    else:
                        rgb = obs["image"]["front_camera"]["rgb"]
                    if hasattr(rgb, 'cpu'):
                        rgb = rgb.cpu().numpy()
                    if len(rgb.shape) == 4:
                        rgb = rgb[0]
                    if rgb.dtype in [np.float32, np.float64]:
                        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    print("Reset!")
                elif key == ord('s'):
                    filename = f"camera_{args.camera_mode}.png"
                    cv2.imwrite(filename, bgr)
                    print(f"Saved to {filename}")
            
            cv2.destroyAllWindows()
    else:
        print("Error: front_camera RGB not found in observations")
        print(f"Observation keys: {obs.keys()}")
    
    env.close()


if __name__ == "__main__":
    main()
