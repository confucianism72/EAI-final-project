
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
                    if hasattr(v, "joints"):
                        print(f"  Controller value[{k}] joints: {[j.name for j in v.joints]}")
    else:
        print("Unwrapped has no 'agent' attribute")

    envs.close()

if __name__ == "__main__":
    debug_env()
