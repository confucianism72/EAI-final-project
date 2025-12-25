"""
LeanRL-style PPO Runner with tensordict, torch.compile, and CudaGraphModule.
Based on LeanRL/cleanrl ppo_continuous_action_torchcompile.py
"""
import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import random
import time
from collections import deque
from functools import partial
from pathlib import Path
import sys
import subprocess

import gymnasium as gym
import hydra
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from omegaconf import OmegaConf
from tensordict import from_module
from tensordict.nn import CudaGraphModule

from scripts.training.agent import Agent
from scripts.training.common import make_env
from scripts.training.ppo_utils import optimized_gae, make_ppo_update_fn

class PPORunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        self.num_envs = cfg.training.num_envs
        self.num_steps = cfg.training.num_steps
        self.total_timesteps = cfg.training.total_timesteps
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // cfg.ppo.num_minibatches
        self.num_iterations = self.total_timesteps // self.batch_size
        
        # Compile settings
        self.compile = cfg.get("compile", True)
        self.cudagraphs = cfg.get("cudagraphs", True)
        self.anneal_lr = cfg.get("anneal_lr", True)
        
        # Seeding
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        
        # Env setup
        self.envs = make_env(cfg, self.num_envs)
        
        # Eval env with video recording
        self.video_dir = None
        if cfg.capture_video:
            output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            self.video_dir = str(output_dir / "videos")
        self.eval_envs = make_env(cfg, cfg.training.num_eval_envs, for_eval=True, video_dir=self.video_dir)
        
        # Determine observation/action dimensions
        obs_space = self.envs.single_observation_space
        act_space = self.envs.single_action_space
        print(f"Observation space: {obs_space}")
        print(f"Action space: {act_space}")
        
        # Handle different observation modes (Dict vs Box)
        if hasattr(obs_space, "shape") and obs_space.shape is not None:
            self.n_obs = math.prod(obs_space.shape)
        elif isinstance(obs_space, gym.spaces.Dict):
            self.n_obs = sum(math.prod(s.shape) for s in obs_space.values())
        else:
            self.n_obs = sum(math.prod(s.shape) for s in obs_space.spaces.values()) if hasattr(obs_space, "spaces") else 0
            
        if hasattr(act_space, "shape") and act_space.shape is not None:
            self.n_act = math.prod(act_space.shape)
        else:
            self.n_act = sum(math.prod(s.shape) for s in act_space.spaces.values()) if hasattr(act_space, "spaces") else 0
        
        print(f"n_obs: {self.n_obs}, n_act: {self.n_act}")
        
        # Agent setup
        self.agent = Agent(self.n_obs, self.n_act, device=self.device)
        
        # Create inference agent only when using CudaGraphModule
        # (CudaGraph captures weights, so we need a separate copy for inference)
        if self.cudagraphs:
            self.agent_inference = Agent(self.n_obs, self.n_act, device=self.device)
            from_module(self.agent).data.to_module(self.agent_inference)
        else:
            self.agent_inference = None  # Not needed without CudaGraph
        
        # Optimizer with fused and capturable for maximum performance
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=torch.tensor(cfg.ppo.learning_rate, device=self.device),
            eps=1e-5,
            fused=True,
            capturable=self.cudagraphs and not self.compile,
        )
        
        if cfg.checkpoint:
            print(f"Loading checkpoint from {cfg.checkpoint}")
            self.agent.load_state_dict(torch.load(cfg.checkpoint))
            if self.cudagraphs:
                from_module(self.agent).data.to_module(self.agent_inference)
        
        # Setup compiled functions
        self._setup_compiled_functions()
        
        # Runtime vars
        self.global_step = 0
        self.avg_returns = deque(maxlen=20)
        self.reward_component_sum = {}  # Accumulated reward components
        self.reward_component_count = 0  # Step count for averaging
        self.success_count = 0  # Success events during rollout
        self.fail_count = 0  # Fail events during rollout
        
        # Termination tracking for logging
        self.terminated_count = 0
        self.truncated_count = 0
        
        # Episode return tracking (per-env accumulator)
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        
        # Handle timeout termination (if True, truncated episodes bootstrap)
        self.handle_timeout_termination = cfg.ppo.get("handle_timeout_termination", True)
        
        # Reward mode for logging
        self.reward_mode = cfg.reward.get("reward_mode", "sparse")
        self.staged_reward = self.reward_mode == "staged_dense"

    def _setup_compiled_functions(self):
        """Setup torch.compile and CudaGraphModule."""
        cfg = self.cfg
        
        # 1. Policy (Inference)
        # Use agent_inference for CudaGraph (needs separate copy for weight sync)
        # Use agent directly for torch.compile (dynamic, no capture)
        inference_agent = self.agent_inference if self.cudagraphs else self.agent
        self.policy = inference_agent.get_action_and_value
        self.get_value = inference_agent.get_value
        
        # 2. GAE: Use functools.partial to bind gamma/gae_lambda
        self.gae_fn = partial(
            optimized_gae,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda
        )

        # 3. Update: Use factory function from ppo_utils
        self.update_fn = make_ppo_update_fn(self.agent, self.optimizer, cfg)
        
        if self.compile:
            print("Compiling functions...")
            if self.cudagraphs:
                # When using CudaGraphModule, use default compile mode (not reduce-overhead)
                # reduce-overhead internally uses CUDA graphs which conflicts with CudaGraphModule
                self.policy = torch.compile(self.policy)
            else:
                # When not using CudaGraphModule, reduce-overhead is safe
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
            # get_value: compile for consistency (called once per iteration)
            self.get_value = torch.compile(self.get_value, mode="reduce-overhead")
            # Update: Always use reduce-overhead (no CudaGraphModule on update)
            self.update_fn = torch.compile(self.update_fn, mode="reduce-overhead")
        
        if self.cudagraphs:
            print("Applying CudaGraphModule to Policy (Inference Only)...")
            self.policy = CudaGraphModule(self.policy)

    def _flatten_obs(self, obs):
        """Flatten dictionary observations into a single tensor."""
        if isinstance(obs, torch.Tensor):
            return obs
        if isinstance(obs, dict):
            # Sort keys to ensure consistent order
            return torch.cat([self._flatten_obs(obs[k]) for k in sorted(obs.keys())], dim=-1)
        return torch.as_tensor(obs, device=self.device)

    def _step_env(self, action):
        """Execute environment step.
        
        Returns:
            next_obs, reward, terminated, truncated, done, info
            where done = terminated | truncated (for episode boundary tracking)
        """
        next_obs, reward, terminations, truncations, info = self.envs.step(action)
        done = terminations | truncations
        return next_obs, reward, terminations, truncations, done, info

    def _rollout(self, obs, bootstrap_mask):
        """Collect trajectories with pre-allocated storage.
        
        Args:
            obs: Current observations
            bootstrap_mask: Mask for GAE bootstrap. If handle_timeout_termination=True,
                           this is `terminated`. If False, this is `done` (terminated|truncated).
        """
        # 1. Pre-allocate TensorDict (Zero-copy optimization)
        storage = tensordict.TensorDict({
            "obs": torch.zeros((self.num_steps, self.num_envs, self.n_obs), device=self.device, dtype=obs.dtype),
            "bootstrap_mask": torch.zeros((self.num_steps, self.num_envs), device=self.device, dtype=torch.bool),
            "vals": torch.zeros((self.num_steps, self.num_envs), device=self.device),
            "actions": torch.zeros((self.num_steps, self.num_envs, self.n_act), device=self.device),
            "logprobs": torch.zeros((self.num_steps, self.num_envs), device=self.device),
            "rewards": torch.zeros((self.num_steps, self.num_envs), device=self.device),
        }, batch_size=[self.num_steps, self.num_envs], device=self.device)

        for step in range(self.num_steps):
            # 2. In-place write (CleanRL style: store BEFORE step)
            # IMPORTANT: Use storage['key'][step] NOT storage[step]['key'] for TensorDict
            storage["obs"][step] = obs
            storage["bootstrap_mask"][step] = bootstrap_mask  # Store PRE-step mask for GAE
            
            # Inference (no gradients during rollout)
            with torch.no_grad():
                action, logprob, _, value = self.policy(obs=obs)
            storage["vals"][step] = value.flatten()
            storage["actions"][step] = action
            storage["logprobs"][step] = logprob
            
            # Environment Step
            next_obs, reward, next_terminated, next_truncated, next_done, infos = self._step_env(action)
            storage["rewards"][step] = reward
            
            # Accumulate reward components for logging (mean over entire rollout)
            # On auto_reset, ManiSkillVectorEnv moves original info to 'final_info'
            reward_comps = None
            if "reward_components" in infos:
                reward_comps = infos["reward_components"]
            elif "final_info" in infos and "reward_components" in infos["final_info"]:
                reward_comps = infos["final_info"]["reward_components"]
            if reward_comps is not None:
                for k, v in reward_comps.items():
                    self.reward_component_sum[k] = self.reward_component_sum.get(k, 0) + v
                self.reward_component_count += 1
            # Accumulate success/fail counts
            # On auto_reset, ManiSkillVectorEnv moves original info to 'final_info'
            # Check both locations to capture counts from both regular and reset steps
            if "success_count" in infos:
                self.success_count += infos["success_count"]
            elif "final_info" in infos and "success_count" in infos["final_info"]:
                self.success_count += infos["final_info"]["success_count"]
            if "fail_count" in infos:
                self.fail_count += infos["fail_count"]
            elif "final_info" in infos and "fail_count" in infos["final_info"]:
                self.fail_count += infos["final_info"]["fail_count"]
            next_obs_flat = self._flatten_obs(next_obs)
            # Accumulate episode returns
            self.episode_returns += reward
            
            # Log episode info when episodes end
            done = next_terminated | next_truncated
            if done.any():
                for idx in torch.where(done)[0]:
                    # Track termination reason
                    if next_terminated[idx].item():
                        self.terminated_count += 1
                    else:
                        self.truncated_count += 1
                    
                    # Record completed episode return
                    ep_return = self.episode_returns[idx].item()
                    self.avg_returns.append(ep_return)
                    self.episode_returns[idx] = 0.0  # Reset for next episode
            
            obs = next_obs_flat
            # Choose bootstrap mask based on config
            if self.handle_timeout_termination:
                bootstrap_mask = next_terminated  # Only true terminations stop bootstrap
            else:
                bootstrap_mask = next_done  # CleanRL default: both stop bootstrap
        
        # Apply reward scale at the end of rollout (or before GAE)
        storage["rewards"] *= self.cfg.ppo.reward_scale
        
        return obs, bootstrap_mask, storage

    def train(self):
        print(f"\n{'='*60}")
        print(f"Training PPO (LeanRL-style) on Track1 {self.cfg.env.task}")
        print(f"Device: {self.device}, Compile: {self.compile}, CudaGraphs: {self.cudagraphs}")
        print(f"Total Timesteps: {self.total_timesteps}, Batch Size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        # Initial reset
        next_obs, _ = self.envs.reset(seed=self.cfg.seed)
        next_obs = self._flatten_obs(next_obs).to(self.device)
        next_bootstrap_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        pbar = tqdm.tqdm(range(1, self.num_iterations + 1))
        global_step_burnin = None
        training_time = 0.0  # Accumulated training time (excludes eval)
        measure_burnin = 3
        iter_start_time = None
        
        for iteration in pbar:
            if iteration == measure_burnin:
                global_step_burnin = self.global_step
            
            # Start timing this iteration (training only)
            iter_start_time = time.time()
            
            # LR Annealing
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.cfg.ppo.learning_rate
                self.optimizer.param_groups[0]["lr"].copy_(lrnow)
            
            # Mark step for cudagraph
            torch.compiler.cudagraph_mark_step_begin()
            
            # Rollout
            next_obs, next_bootstrap_mask, container = self._rollout(next_obs, next_bootstrap_mask)
            self.global_step += container.numel()
            
            # GAE Calculation
            # bootstrap_mask controls when to stop bootstrapping:
            # - handle_timeout_termination=True: only terminated stops bootstrap (truncated continues)
            # - handle_timeout_termination=False: both stop bootstrap (CleanRL default)
            with torch.no_grad():
                next_value = self.get_value(next_obs)
            
            advs, rets = self.gae_fn(
                container["rewards"],
                container["vals"],
                container["bootstrap_mask"],
                next_value,
                next_bootstrap_mask
            )
            container["advantages"] = advs
            container["returns"] = rets
            
            # Flatten for PPO Update
            container_flat = container.view(-1)
            
            # PPO Update (with clipfrac accumulation like CleanRL)
            clipfracs = []
            for epoch in range(self.cfg.ppo.update_epochs):
                b_inds = torch.randperm(container_flat.shape[0], device=self.device).split(self.minibatch_size)
                for b in b_inds:
                    container_local = container_flat[b]
                    out = self.update_fn(container_local, tensordict_out=tensordict.TensorDict())
                    clipfracs.append(out["clipfrac"].item())
                    
                    if self.cfg.ppo.target_kl is not None and out["approx_kl"] > self.cfg.ppo.target_kl:
                        break
                else:
                    continue
                break
            
            # Sync params to inference agent (only needed for CudaGraph)
            if self.cudagraphs:
                from_module(self.agent).data.to_module(self.agent_inference)
            
            # Logging (every iteration after burnin)
            if global_step_burnin is not None and training_time > 0:
                speed = (self.global_step - global_step_burnin) / training_time
                avg_return = np.array(self.avg_returns).mean() if self.avg_returns else 0
                lr = self.optimizer.param_groups[0]["lr"]
                if isinstance(lr, torch.Tensor):
                    lr = lr.item()
                
                # Compute explained variance
                y_pred = container["vals"].flatten().cpu().numpy()
                y_true = container["returns"].flatten().cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                
                pbar.set_description(
                    f"SPS: {speed:.0f}, return: {avg_return:.2f}, lr: {lr:.2e}"
                )
                
                logs = {
                    "charts/SPS": speed,
                    "charts/learning_rate": lr,
                    "charts/terminated_count": self.terminated_count,
                    "charts/truncated_count": self.truncated_count,
                    "losses/value_loss": out["v_loss"].item(),
                    "losses/policy_loss": out["pg_loss"].item(),
                    "losses/entropy": out["entropy_loss"].item(),
                    "losses/approx_kl": out["approx_kl"].item(),
                    "losses/old_approx_kl": out["old_approx_kl"].item(),
                    "losses/clipfrac": np.mean(clipfracs),  # Average over all minibatches (CleanRL style)
                    "losses/explained_variance": explained_var,
                    "losses/grad_norm": out["gn"].item() if isinstance(out["gn"], torch.Tensor) else out["gn"],
                    "rollout/ep_return_mean": avg_return,
                    "rollout/rewards_mean": container["rewards"].mean().item(),
                    "rollout/rewards_max": container["rewards"].max().item(),
                }
                
                # Add reward components (averaged over entire rollout)
                if self.reward_component_count > 0:
                    for name, total in self.reward_component_sum.items():
                        logs[f"reward/{name}"] = total / self.reward_component_count
                    # Add success/fail counts (total in this rollout)
                    logs["reward/success_count"] = self.success_count
                    logs["reward/fail_count"] = self.fail_count
                    # Reset for next rollout
                    self.reward_component_sum = {}
                    self.reward_component_count = 0
                    self.success_count = 0
                    self.fail_count = 0
                
                if self.cfg.wandb.enabled:
                    wandb.log(logs, step=self.global_step)
                    
            # Accumulate training time for this iteration
            training_time += time.time() - iter_start_time
            
            # Evaluation (timed separately)
            if iteration % self.cfg.training.eval_freq == 0:
                eval_start = time.time()
                self._evaluate()
                eval_duration = time.time() - eval_start
                print(f"  Eval took {eval_duration:.2f}s")
                if self.cfg.wandb.enabled:
                    wandb.log({"charts/eval_time": eval_duration}, step=self.global_step)
                self._save_checkpoint(iteration)
        
        self.envs.close()
        self.eval_envs.close()
        print("Training complete!")

    def _evaluate(self):
        """Run evaluation episodes."""
        print("Running evaluation...")
        
        # Optimization: Instead of recreating env, flush the video wrapper state
        # directly in the existing eval_envs. save=False ignores pre-eval trash.
        self.eval_envs.call("flush_video", save=False)
        
        eval_obs, _ = self.eval_envs.reset()
        eval_returns = []
        eval_successes = []
        eval_fails = []
        episode_rewards = torch.zeros(self.cfg.training.num_eval_envs, device=self.device)
        
        # Track reward components during eval
        eval_reward_components = {}
        eval_component_count = 0
        
        # Per-step reward data for CSV export (list of dicts per env)
        # Structure: {env_idx: [{step, reward, component1, component2, ...}, ...]}
        step_reward_data = {i: [] for i in range(self.cfg.training.num_eval_envs)}
        
        # Compute max_steps consistently: (base * multiplier) + hold_steps
        base = self.cfg.env.episode_steps.get("base", 296)
        multiplier = self.cfg.env.episode_steps.get("multiplier", 1.2)
        hold_steps = 0
        if "reward" in self.cfg and "stable_hold_time" in self.cfg.reward:
            hold_steps = int(self.cfg.reward.stable_hold_time * self.cfg.env.get("control_freq", 30))
        
        training_steps = int(base * multiplier) + hold_steps
        eval_multiplier = self.cfg.training.get("eval_step_multiplier", 1.0)
        max_steps = int(training_steps * eval_multiplier)
        
        # Add a small buffer for safety
        max_steps += 2
        
        for step in range(max_steps):
            with torch.no_grad():
                eval_action = self.agent.get_action(eval_obs, deterministic=True)
            eval_obs, reward, terminated, truncated, eval_infos = self.eval_envs.step(eval_action)
            
            episode_rewards += reward
            
            # Accumulate reward components
            if "reward_components" in eval_infos:
                for k, v in eval_infos["reward_components"].items():
                    eval_reward_components[k] = eval_reward_components.get(k, 0) + v
                eval_component_count += 1
                
                # Collect per-step data for CSV export
                for env_idx in range(self.cfg.training.num_eval_envs):
                    step_data = {
                        "step": step,
                        "reward": reward[env_idx].item(),
                    }
                    # Add each reward component
                    for k, v in eval_infos["reward_components"].items():
                        step_data[k] = v  # These are already averaged across envs
                    step_reward_data[env_idx].append(step_data)
            
            # Check for episode completion
            done = terminated | truncated
            if done.any():
                for idx in torch.where(done)[0]:
                    eval_returns.append(episode_rewards[idx].item())
                    episode_rewards[idx] = 0.0  # Reset for next episode
                    
                    # Check success/fail from final_info (ManiSkill provides this)
                    # Note: success/fail are tensors, use .item() to get Python bool
                    if "final_info" in eval_infos:
                        if "success" in eval_infos["final_info"]:
                            success = eval_infos["final_info"]["success"][idx].item()
                            eval_successes.append(bool(success))
                        if "fail" in eval_infos["final_info"]:
                            fail = eval_infos["final_info"]["fail"][idx].item()
                            eval_fails.append(bool(fail))
            
            # Stop after collecting enough episodes
            if len(eval_returns) >= self.cfg.training.num_eval_envs:
                break
        
        if eval_returns:
            mean_return = np.mean(eval_returns)
            success_rate = np.mean(eval_successes) if eval_successes else 0.0
            fail_rate = np.mean(eval_fails) if eval_fails else 0.0
            print(f"  eval/return = {mean_return:.4f}, success_rate = {success_rate:.2%}, fail_rate = {fail_rate:.2%} (n={len(eval_returns)})")
            
            # Build log dict
            eval_logs = {
                "eval/return": mean_return,
                "eval/success_rate": success_rate,
                "eval/fail_rate": fail_rate,
            }
            
            # Add eval reward components
            if eval_component_count > 0:
                for name, total in eval_reward_components.items():
                    eval_logs[f"eval_reward/{name}"] = total / eval_component_count
            
            if self.cfg.wandb.enabled:
                wandb.log(eval_logs, step=self.global_step)
        
        if self.video_dir is not None:
            # Finalize the evaluation video
            self.eval_envs.call("flush_video", save=True)
            self._async_split_videos()
            
            # Save per-step reward data as CSV alongside videos
            import csv
            for env_idx, steps in step_reward_data.items():
                if steps:  # Only save if we have data
                    csv_path = self.video_dir / f"step_{self.global_step}_env{env_idx}_rewards.csv"
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=steps[0].keys())
                        writer.writeheader()
                        writer.writerows(steps)

    def _save_checkpoint(self, iteration):
        """Save model checkpoint."""
        if self.cfg.save_model:
            output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            model_path = output_dir / f"ckpt_{iteration}.pt"
            torch.save(self.agent.state_dict(), model_path)
            torch.save(self.agent.state_dict(), output_dir / "latest.pt")

    def _async_split_videos(self):
        """Asynchronously split tiled eval videos into individual env videos."""
        import subprocess
        
        video_dir = Path(self.video_dir)
        if not video_dir.exists():
            return
        
        # Find any mp4 files that haven't been split yet
        mp4_files = list(video_dir.glob("*.mp4"))
        if not mp4_files:
            return
        
        # Spawn background process to split videos
        # Uses the split_video.py script with --rgb_only flag
        cmd = [
            sys.executable, "scripts/utils/split_video.py",
            str(video_dir),
            "--num_envs", str(self.cfg.training.num_eval_envs),
        ]
        
        # Run in background (non-blocking)
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Detach from parent process
        )
