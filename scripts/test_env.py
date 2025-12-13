import gymnasium as gym
import mani_skill.envs
import scripts.track1_env
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Enable GUI viewer")
    parser.add_argument("--task", type=str, default="lift", choices=["lift", "stack", "sort"], help="Task type")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--no_dr", action="store_true", help="Disable domain randomization")
    args = parser.parse_args()

    render_mode = "human" if args.gui else "rgb_array"
    env = gym.make(
        "Track1-v0", 
        render_mode=render_mode, 
        obs_mode="sensor_data", 
        reward_mode="none",
        task=args.task,
        domain_randomization=not args.no_dr,
        num_envs=args.num_envs,
    )
    
    print(f"Task: {args.task}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Domain randomization: {not args.no_dr}")
    
    obs, _ = env.reset()
    
    print("Keys in observation:", obs.keys())
    if 'sensor_data' in obs:
        print("Keys in sensor_data:", obs['sensor_data'].keys())
        
        # Save front camera image
        if 'front_camera' in obs['sensor_data']:
            print("Keys in front_camera:", obs['sensor_data']['front_camera'].keys())
            rgb = obs['sensor_data']['front_camera']['Color']
            if len(rgb.shape) == 4:
                rgb = rgb[0]
            rgb = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else rgb
            rgb = (rgb).astype(np.uint8)
            img = Image.fromarray(rgb)
            img.save(f"track1_{args.task}.png")
            print(f"Saved track1_{args.task}.png")
        
        # Note wrist cameras
        for key in obs['sensor_data'].keys():
            if 'wrist' in key:
                print(f"  {key}: {obs['sensor_data'][key].keys()}")
    else:
        print("sensor_data not found in observation")

    # Check evaluation
    info = env.unwrapped.get_info()
    print("\nEvaluation info:", info)

    if args.gui:
        print("Press 'q' in the viewer to exit (or close the window).")
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated.any() if hasattr(terminated, 'any') else terminated:
                obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
