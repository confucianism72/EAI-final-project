
import os
import torch
import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
from scripts.training.common import make_env
from scripts.training.runner import PPORunner

def verify_names():
    # 1. Setup exact config as used in training (using real yaml if possible)
    # We'll use a representative one for Lift
    cfg = OmegaConf.create({
        "env": {
            "env_id": "Track1-v0",
            "task": "lift",
            "control_mode": "pd_joint_delta_pos",
            "camera_mode": "direct_pinhole",
            "obs_mode": "state",
        },
        "training": {
            "num_envs": 1,
            "num_eval_envs": 1,
            "total_timesteps": 1000000,
            "num_steps": 50,
        },
        "ppo": {
            "learning_rate": 3e-4, 
            "gamma": 0.95, 
            "gae_lambda": 0.9,
            "num_minibatches": 2,
            "update_epochs": 4,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "obs": {
            "enabled": False,
            "include_abs_pos": ["tcp_pos", "red_cube_pos"],
            "include_target_qpos": "relative", # This is what single_arm.yaml uses
            "include_is_grasped": True,
            "include_tcp_orientation": True,
            "include_cube_displacement": True,
            "relative_pos_clip": 0.5,
        },
        "reward": {
            "reward_mode": "dense",
        },
        "seed": 0,
        "capture_video": False,
        "n_steps": 50,
        "n_envs": 1,
        "checkpoint": None,
        "save_model": False,
        "log_obs_stats": True,
        "normalize_obs": True,
    })
    
    # 2. Create actual environment
    print("Creating environment...")
    envs = make_env(cfg, 1)
    # Get raw dict observation from unwrapped env to see structure
    raw_obs = envs.unwrapped._get_obs_state_dict({})
    
    def print_keys(d, indent=""):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{indent}{k}:")
                print_keys(v, indent + "  ")
            elif isinstance(v, torch.Tensor):
                print(f"{indent}{k}: shape {v.shape}")
            else:
                print(f"{indent}{k}: {type(v)}")

    print("Raw observation structure:")
    print_keys(raw_obs)
    
    obs = envs.reset()[0]
    n_obs = obs.shape[-1]
    print(f"Actual flattened observation size: {n_obs}")

    # 3. Initialize Runner
    print("Initializing Runner...")
    runner = PPORunner(cfg)
    obs_names = runner.obs_names
    
    print(f"Generated observation names count: {len(obs_names)}")
    
    # Check for mismatch
    if len(obs_names) != n_obs:
        print(f"!!! MISMATCH DETECTED: {len(obs_names)} names vs {n_obs} flattened dims")
        print("Detailed names generated:")
        for i, name in enumerate(obs_names):
            print(f"  {i:2d}: {name}")
    else:
        print("SUCCESS: Observation names count matches flattened size!")
        print("Sample names including displacement:")
        for name in obs_names:
            if "displacement" in name:
                print(f"  {name}")
        for name in obs_names:
            if "tcp_pos" in name:
                print(f"  {name}")

    envs.close()

if __name__ == "__main__":
    verify_names()
