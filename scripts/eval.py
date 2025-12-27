"""
Standalone evaluation script for trained PPO checkpoints.
Reuses the same evaluation logic as the training runner (DRY).

Usage:
    uv run -m scripts.eval checkpoint=/path/to/ckpt.pt
    uv run -m scripts.eval checkpoint=/path/to/ckpt.pt training.num_eval_envs=16
"""
import os
import sys
from pathlib import Path

import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Set float32 matmul precision for speed
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Run evaluation on a checkpoint."""
    
    # Validate checkpoint
    if not cfg.checkpoint:
        print("ERROR: Must specify checkpoint path")
        print("Usage: uv run -m scripts.eval checkpoint=/path/to/ckpt.pt")
        sys.exit(1)
    
    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(OmegaConf.to_yaml(cfg))
    
    # Seeding
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    
    # Import here to avoid slow imports when just checking usage
    from scripts.training.common import make_env
    from scripts.training.agent import Agent
    from scripts.training.runner import PPORunner
    
    # Output directory for eval videos
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    video_dir = output_dir / "eval_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"Eval videos will be saved to: {video_dir}")
    
    # Create eval environment
    num_eval_envs = cfg.training.num_eval_envs
    eval_envs = make_env(cfg, num_eval_envs, for_eval=True, video_dir=str(video_dir))
    
    # Get observation and action dimensions
    obs_sample, _ = eval_envs.reset()
    if isinstance(obs_sample, dict):
        n_obs = sum(v.shape[-1] for v in obs_sample.values())
    else:
        n_obs = obs_sample.shape[-1]
    n_act = eval_envs.single_action_space.shape[0]
    
    print(f"Obs dim: {n_obs}, Action dim: {n_act}")
    print(f"Num eval envs: {num_eval_envs}")
    
    # Create agent and load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(n_obs, n_act, device=device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint)
    agent.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Create a minimal evaluator that reuses PPORunner's _evaluate method
    # This is a lightweight wrapper that only has the methods needed for eval
    evaluator = EvalRunner(cfg, eval_envs, agent, device, video_dir)
    
    # Run evaluation
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50 + "\n")
    
    evaluator.run_eval()
    
    # Cleanup
    eval_envs.close()
    print("\nEvaluation complete!")


class EvalRunner:
    """Minimal runner for standalone evaluation.
    
    Shares _evaluate and _flatten_obs methods with PPORunner.
    """
    
    def __init__(self, cfg, eval_envs, agent, device, video_dir):
        self.cfg = cfg
        self.eval_envs = eval_envs
        self.agent = agent
        self.device = device
        self.video_dir = video_dir
        self.eval_count = 0
        self.global_step = 0  # For compatibility with wandb logging
    
    def _flatten_obs(self, obs):
        """Flatten dictionary observations into a single tensor.
        
        Copied from PPORunner for DRY - same logic.
        """
        if isinstance(obs, torch.Tensor):
            return obs
        if isinstance(obs, dict):
            return torch.cat([self._flatten_obs(obs[k]) for k in sorted(obs.keys())], dim=-1)
        return torch.as_tensor(obs, device=self.device)
    
    def run_eval(self):
        """Run evaluation episodes."""
        # Reuse the exact same eval logic as PPORunner._evaluate
        # This is essentially a copy of _evaluate but without wandb logging
        
        print("Running evaluation episodes...")
        
        # Flush video wrapper
        self.eval_envs.call("flush_video", save=False)
        
        eval_obs, _ = self.eval_envs.reset()
        eval_returns = []
        eval_successes = []
        eval_fails = []
        episode_rewards = torch.zeros(self.cfg.training.num_eval_envs, device=self.device)
        
        # Track reward components
        eval_reward_components = {}
        eval_component_count = 0
        
        # Compute max_steps
        base = self.cfg.env.episode_steps.get("base", 296)
        multiplier = self.cfg.env.episode_steps.get("multiplier", 1.2)
        hold_steps = 0
        if "reward" in self.cfg and "stable_hold_time" in self.cfg.reward:
            hold_steps = int(self.cfg.reward.stable_hold_time * self.cfg.env.get("control_freq", 30))
        
        training_steps = int(base * multiplier) + hold_steps
        eval_multiplier = self.cfg.training.get("eval_step_multiplier", 1.0)
        max_steps = int(training_steps * eval_multiplier) + 2
        
        print(f"Running for max {max_steps} steps per episode...")
        
        for step in range(max_steps):
            # CRITICAL: Flatten obs like train does
            obs_flat = self._flatten_obs(eval_obs)
            with torch.no_grad():
                eval_action = self.agent.get_action(obs_flat, deterministic=True)
            eval_obs, reward, terminated, truncated, eval_infos = self.eval_envs.step(eval_action)
            
            episode_rewards += reward
            
            # Accumulate reward components
            reward_comps = None
            if "reward_components" in eval_infos:
                reward_comps = eval_infos["reward_components"]
            elif "final_info" in eval_infos and "reward_components" in eval_infos["final_info"]:
                reward_comps = eval_infos["final_info"]["reward_components"]
            
            if reward_comps is not None:
                for k, v in reward_comps.items():
                    val = v.item() if hasattr(v, 'item') else v
                    eval_reward_components[k] = eval_reward_components.get(k, 0) + val
                eval_component_count += 1
            
            # Check for episode completion
            done = terminated | truncated
            if done.any():
                for idx in torch.where(done)[0]:
                    eval_returns.append(episode_rewards[idx].item())
                    episode_rewards[idx] = 0.0
                    
                    if "final_info" in eval_infos:
                        if "success" in eval_infos["final_info"]:
                            success = eval_infos["final_info"]["success"][idx].item()
                            eval_successes.append(bool(success))
                        if "fail" in eval_infos["final_info"]:
                            fail = eval_infos["final_info"]["fail"][idx].item()
                            eval_fails.append(bool(fail))
            
            # Stop when we have enough episodes
            if len(eval_returns) >= self.cfg.training.num_eval_envs:
                break
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        if eval_returns:
            mean_return = np.mean(eval_returns)
            std_return = np.std(eval_returns)
            success_rate = np.mean(eval_successes) if eval_successes else 0.0
            fail_rate = np.mean(eval_fails) if eval_fails else 0.0
            
            print(f"Episodes completed: {len(eval_returns)}")
            print(f"Mean return: {mean_return:.4f} ± {std_return:.4f}")
            print(f"Success rate: {success_rate:.2%}")
            print(f"Fail rate: {fail_rate:.2%}")
            
            if eval_component_count > 0:
                print("\nReward components (avg):")
                for name, total in sorted(eval_reward_components.items()):
                    print(f"  {name}: {total / eval_component_count:.4f}")
        else:
            print("No episodes completed!")
        
        # Save video and split into individual env videos
        if self.video_dir is not None:
            self.eval_envs.call("flush_video", save=True)
            print(f"\nVideos saved to: {self.video_dir}")
            self._split_videos()
        
        print("="*50)
    
    def _split_videos(self):
        """Split tiled eval videos into individual env videos.
        
        Same logic as PPORunner._async_split_videos but runs synchronously.
        """
        import subprocess
        
        video_dir = Path(self.video_dir)
        if not video_dir.exists():
            return
        
        # Find mp4 files
        mp4_files = list(video_dir.glob("*.mp4"))
        if not mp4_files:
            return
        
        print("Splitting videos into individual env videos...")
        
        # Run split_video.py
        cmd = [
            sys.executable, "scripts/utils/split_video.py",
            str(video_dir),
            "--num_envs", str(self.cfg.training.num_eval_envs),
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("Video splitting complete!")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Video splitting failed: {e}")


if __name__ == "__main__":
    main()
