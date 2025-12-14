#!/usr/bin/env python3
"""GUI viewer script for Track1 environment inspection.

Usage:
    python scripts/view_env.py --task lift
    python scripts/view_env.py --task lift --camera-mode distorted
    python scripts/view_env.py --task lift --camera-mode distort-twice
    python scripts/view_env.py --task lift --camera-mode direct_pinhole
    python scripts/view_env.py --task lift --domain-randomization
"""

import argparse
import gymnasium as gym
from scripts.track1_env import Track1Env


def main():
    parser = argparse.ArgumentParser(description="View Track1 environment in GUI")
    parser.add_argument("--task", type=str, default="lift", 
                        choices=["lift", "stack", "sort"],
                        help="Task type to visualize")
    parser.add_argument("--domain-randomization", action="store_true",
                        help="Enable domain randomization")
    parser.add_argument("--camera-mode", type=str, default="direct_pinhole",
                        choices=["distorted", "distort-twice", "direct_pinhole"],
                        help="Camera output mode: distorted (raw fisheye), distort-twice (rectified), direct_pinhole (efficient render)")
    args = parser.parse_args()
    
    print(f"Starting Track1 environment with task={args.task}, camera_mode={args.camera_mode}")
    print("Controls:")
    print("  - Mouse: Rotate camera view")
    print("  - Scroll: Zoom in/out")
    print("  - R: Reset episode")
    print("  - Q: Quit")
    
    env = gym.make(
        "Track1-v0",
        render_mode="human",
        obs_mode="sensor_data",
        reward_mode="none",  # Disable reward computation
        task=args.task,
        domain_randomization=args.domain_randomization,
        camera_mode=args.camera_mode,
        num_envs=1,
    )
    
    obs, _ = env.reset()
    print("Environment ready! Use viewer to inspect.")
    print("Close the window or press Ctrl+C to exit.")
    
    # Keep viewer open without stepping (static inspection)
    try:
        while True:
            env.render()
    except (KeyboardInterrupt, AttributeError) as e:
        print(f"\nExiting view_env... ({e})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
