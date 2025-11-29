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
    args = parser.parse_args()

    render_mode = "human" if args.gui else "rgb_array"
    env = gym.make("Track1-v0", render_mode=render_mode, obs_mode="sensor_data", reward_mode="none")
    obs, _ = env.reset()
    
    # Render the front camera view
    # In ManiSkill, sensors are part of the observation if configured.
    # We configured "front_camera" in _default_sensor_configs.
    # obs['sensor_data']['front_camera']['rgb'] should contain the image.
    
    print("Keys in observation:", obs.keys())
    if 'sensor_data' in obs:
        print("Keys in sensor_data:", obs['sensor_data'].keys())
        if 'front_camera' in obs['sensor_data']:
            print("Keys in front_camera:", obs['sensor_data']['front_camera'].keys())
            rgb = obs['sensor_data']['front_camera']['Color']
            # Color is likely [H, W, 4] (RGBA) or [H, W, 3] (RGB)
            # If batched: [1, H, W, 4]
            # rgb is likely a torch tensor or numpy array [H, W, 3] or [1, H, W, 3]
            if len(rgb.shape) == 4:
                rgb = rgb[0]
            
            rgb = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else rgb
            rgb = (rgb).astype(np.uint8)
            
            img = Image.fromarray(rgb)
            img.save("track1_env_init.png")
            print("Saved track1_env_init.png")
        else:
            print("front_camera not found in sensor_data")
    else:
        print("sensor_data not found in observation")

    if args.gui:
        print("Press 'q' in the viewer to exit (or close the window).")
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    main()
