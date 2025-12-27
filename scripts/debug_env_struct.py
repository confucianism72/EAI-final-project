
import os
import torch
import gymnasium as gym
from omegaconf import OmegaConf
from scripts.training.common import make_env

def debug_env():
    # Load a representative config
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
        },
        "seed": 0,
        "capture_video": False,
    })
    
    envs = make_env(cfg, 1)
    unwrapped = envs.unwrapped
    
    print(f"Unwrapped type: {type(unwrapped)}")
    if hasattr(unwrapped, "agent"):
        print(f"Agent type: {type(unwrapped.agent)}")
        print(f"Agent content (if dict): {list(unwrapped.agent.keys()) if isinstance(unwrapped.agent, dict) else 'Not a dict'}")
        
        # If it's a MultiAgent, it has a controllers property which is a dict
        if hasattr(unwrapped.agent, "controllers"):
            print(f"Agent.controllers keys: {list(unwrapped.agent.controllers.keys())}")
            for k, v in unwrapped.agent.controllers.items():
                print(f"  Controller[{k}] type: {type(v)}")
                if hasattr(v, "joints"):
                     print(f"  Controller[{k}] joints: {[j.name for j in v.joints]}")
        
        # Check what 'controller' property returns
        if hasattr(unwrapped.agent, "controller"):
            ctrl = unwrapped.agent.controller
            print(f"Agent.controller type: {type(ctrl)}")
            if isinstance(ctrl, dict):
                print(f"Agent.controller keys: {list(ctrl.keys())}")
                for k, v in ctrl.items():
                    print(f"  Controller value[{k}] type: {type(v)}")
    print("-" * 20)
    print("Testing PPORunner Fixed Naming Logic...")
    from scripts.training.runner import PPORunner
    # Mocking a minimal config needed for PPORunner init
    runner_cfg = cfg.copy()
    runner_cfg.ppo = OmegaConf.create({
        "learning_rate": 3e-4, "gamma": 0.95, "gae_lambda": 0.9, 
        "clip_coef": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, 
        "max_grad_norm": 0.5, "num_minibatches": 2, "update_epochs": 4
    })
    runner_cfg.training = OmegaConf.create({
        "num_steps": 50, "num_envs": 1, 
        "total_timesteps": 100000, "num_eval_envs": 1
    })
    runner_cfg.n_steps = 50
    runner_cfg.n_envs = 1
    runner_cfg.checkpoint = None
    runner_cfg.save_model = False
    runner_cfg.log_obs_stats = True
    runner_cfg.normalize_obs = True
    runner_cfg.reward = cfg.reward if "reward" in cfg else OmegaConf.create({"reward_mode": "sparse"})
    
    runner = PPORunner(runner_cfg)
    obs_names = runner.obs_names
    
    print(f"Total names: {len(obs_names)}")
    print(f"First 10 names:")
    for n in obs_names[:10]:
        print(f"  {n}")
    
    print(f"TCP/Cube samples:")
    found_special = False
    for n in obs_names:
        if "tcp_pos" in n or "red_cube_pos" in n:
             print(f"  {n}")
             found_special = True
    if not found_special:
        print("  None found (check task/obs_mode)")

    envs.close()

if __name__ == "__main__":
    debug_env()
