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
import threading
import copy

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
        self.eval_count = 0  # Counter for eval runs (eval0, eval1, eval2, ...)
        
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
        logstd_init = cfg.ppo.get("logstd_init", 0.0)
        self.agent = Agent(self.n_obs, self.n_act, device=self.device, logstd_init=logstd_init)
        
        # Create inference agent only when using CudaGraphModule
        # (CudaGraph captures weights, so we need a separate copy for inference)
        if self.cudagraphs:
            self.agent_inference = Agent(self.n_obs, self.n_act, device=self.device, logstd_init=logstd_init)
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
        self.success_count = torch.tensor(0, device=self.device, dtype=torch.float32)  # GPU accumulator
        self.fail_count = torch.tensor(0, device=self.device, dtype=torch.float32)  # GPU accumulator
        
        # Termination tracking for logging
        self.terminated_count = 0
        self.truncated_count = 0
        
        # Episode return tracking (per-env accumulator)
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        
        # Handle timeout termination (if True, truncated episodes bootstrap)
        self.handle_timeout_termination = cfg.ppo.get("handle_timeout_termination", True)
        
        # Observation statistics logging (for monitoring normalization quality)
        self.log_obs_stats = cfg.get("log_obs_stats", False)
        if self.log_obs_stats:
            # EMA for mean/var (no accumulation overflow, smooth updates)
            self.obs_ema_tau = cfg.get("obs_stats_tau", 0.01)  # EMA decay rate from config
            self.obs_ema_mean = torch.zeros(self.n_obs, device=self.device)
            self.obs_ema_var = torch.ones(self.n_obs, device=self.device)
            
            # Fetch observation names for granular logging
            # Traverse wrappers to find the original Dict observation space
            curr_env = self.envs
            original_space = None
            while hasattr(curr_env, "env"):
                if hasattr(curr_env, "single_observation_space") and isinstance(curr_env.single_observation_space, gym.spaces.Dict):
                    original_space = curr_env.single_observation_space
                    break
                curr_env = curr_env.env
            
            if original_space:
                self.obs_names = self._get_obs_names(original_space)
            else:
                self.obs_names = [f"obs_{i}" for i in range(self.n_obs)]
            print(f"Dynamic observation names for logging (count: {len(self.obs_names)})")
        
        # Reward mode for logging
        self.reward_mode = cfg.reward.get("reward_mode", "sparse")
        self.staged_reward = self.reward_mode == "staged_dense"
        
        if hasattr(self.envs.unwrapped, "agent"):
            agent = self.envs.unwrapped.agent
            if hasattr(agent, "controller"):
                ctrl = agent.controller
                if isinstance(ctrl, dict):
                    # MultiAgent setup: agent.controller is a dict of sub-controllers
                    all_joint_names = []
                    for agent_id, sub_ctrl in ctrl.items():
                        if hasattr(sub_ctrl, "joints"):
                            all_joint_names.extend([f"{agent_id}/{j.name}" for j in sub_ctrl.joints])
                    
                    if len(all_joint_names) == self.n_act:
                        self.joint_names = all_joint_names
                    elif self.n_act == 6 and len(all_joint_names) == 12:
                        # Specific heuristic for Track1Env single-arm tasks (which use so101-1)
                        if "so101-1" in ctrl:
                            self.joint_names = [j.name for j in ctrl["so101-1"].joints]
                        else:
                            # Fallback to the last agent's joints if n_act matches
                            last_ctrl = list(ctrl.values())[-1]
                            self.joint_names = [j.name for j in last_ctrl.joints]
                    else:
                        # Fallback to generic names if ambiguous
                        self.joint_names = [f"act_{i}" for i in range(self.n_act)]
                elif hasattr(ctrl, "joints"):
                    self.joint_names = [j.name for j in ctrl.joints]
                else:
                    self.joint_names = [f"act_{i}" for i in range(self.n_act)]
            else:
                self.joint_names = [f"act_{i}" for i in range(self.n_act)]
        else:
            self.joint_names = [f"act_{i}" for i in range(self.n_act)]
        
        # Async eval infrastructure
        self.async_eval = cfg.get("async_eval", True)  # Enable async eval by default
        self.eval_thread = None
        self.eval_stream = torch.cuda.Stream() if self.async_eval else None
        # Create separate eval agent to avoid race condition with training agent
        if self.async_eval:
            self.eval_agent = Agent(self.n_obs, self.n_act, device=self.device)
        else:
            self.eval_agent = None

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

    def _get_obs_names(self, space, prefix=""):
        """Recursively get observation feature names from a space."""
        names = []
        if isinstance(space, gym.spaces.Dict):
            # Sort keys to match FlattenStateWrapper / _flatten_obs order
            for k in sorted(space.keys()):
                new_prefix = f"{prefix}/{k}" if prefix else k
                names.extend(self._get_obs_names(space[k], new_prefix))
        elif hasattr(space, "shape") and space.shape:
            flat_size = np.prod(space.shape)
            if flat_size > 1:
                for i in range(flat_size):
                    names.append(f"{prefix}_{i}")
            else:
                names.append(prefix)
        else:
            names.append(prefix)
        return names

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
                    # Handle GPU tensors (convert to scalar if needed)
                    val = v.item() if hasattr(v, 'item') else v
                    self.reward_component_sum[k] = self.reward_component_sum.get(k, 0) + val
                self.reward_component_count += 1
            # Accumulate success/fail counts on GPU (no sync until logging)
            # Check both top-level and final_info (ManiSkill moves info there on auto_reset)
            success_val = None
            fail_val = None
            if "success_count" in infos:
                success_val = infos["success_count"]
            elif "final_info" in infos and "success_count" in infos["final_info"]:
                success_val = infos["final_info"]["success_count"]
            if "fail_count" in infos:
                fail_val = infos["fail_count"]
            elif "final_info" in infos and "fail_count" in infos["final_info"]:
                fail_val = infos["final_info"]["fail_count"]
            
            if success_val is not None:
                self.success_count += success_val if isinstance(success_val, torch.Tensor) else success_val
            if fail_val is not None:
                self.fail_count += fail_val if isinstance(fail_val, torch.Tensor) else fail_val
            next_obs_flat = self._flatten_obs(next_obs)
            # Accumulate episode returns
            self.episode_returns += reward
            
            # Update observation EMA statistics (for monitoring normalization)
            if self.log_obs_stats:
                # Compute batch mean and var
                batch_mean = obs.mean(dim=0)
                batch_var = obs.var(dim=0, unbiased=False)
                # EMA update (lerp_ is in-place, no new tensor allocation)
                self.obs_ema_mean.lerp_(batch_mean, self.obs_ema_tau)
                self.obs_ema_var.lerp_(batch_var, self.obs_ema_tau)
            
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
        measure_burnin = 2
        iter_start_time = None
        
        for iteration in pbar:
            if iteration == measure_burnin:
                global_step_burnin = self.global_step
                training_time = 0.0  # Reset timer to skip initialization/warmup overhead
            
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
                    # Add success/fail counts (sync to CPU only here)
                    logs["reward/success_count"] = self.success_count.item() if hasattr(self.success_count, 'item') else self.success_count
                    logs["reward/fail_count"] = self.fail_count.item() if hasattr(self.fail_count, 'item') else self.fail_count
                    # Reset for next rollout (keep as GPU tensors)
                    self.reward_component_sum = {}
                    self.reward_component_count = 0
                    self.success_count = torch.tensor(0, device=self.device, dtype=torch.float32)
                    self.fail_count = torch.tensor(0, device=self.device, dtype=torch.float32)
                
                # Log observation statistics (for monitoring normalization quality)
                # Ideal: mean ≈ 0, std ≈ 1 indicates good normalization
                if self.log_obs_stats:
                    obs_std = torch.sqrt(torch.clamp(self.obs_ema_var, min=1e-8))
                    
                    # Log overall summary statistics
                    logs["obs/mean_avg"] = self.obs_ema_mean.mean().item()
                    logs["obs/std_avg"] = obs_std.mean().item()
                    
                    # Log granular per-feature EMA statistics
                    for i, name in enumerate(self.obs_names):
                        if i < len(self.obs_ema_mean):
                            logs[f"obs_mean_ema/{name}"] = self.obs_ema_mean[i].item()
                            logs[f"obs_std_ema/{name}"] = obs_std[i].item()
                
                # Log per-joint action std (for monitoring policy convergence)
                with torch.no_grad():
                    action_std_vec = torch.exp(self.agent.actor_logstd).flatten()
                    for i, name in enumerate(self.joint_names):
                        if i < len(action_std_vec):
                            logs[f"action_std/{name}"] = action_std_vec[i].item()

                if self.cfg.wandb.enabled:
                    wandb.log(logs, step=self.global_step)
                    
            # Accumulate training time for this iteration
            training_time += time.time() - iter_start_time
            
            # Evaluation (async or sync based on config)
            if iteration % self.cfg.training.eval_freq == 0:
                if self.async_eval:
                    # Async eval: launch in background, don't block training
                    if self.eval_thread is not None and self.eval_thread.is_alive():
                        # Wait for previous eval to finish before starting new one
                        self.eval_thread.join()
                    
                    # Copy current weights to separate eval_agent (no race condition)
                    # Note: We don't bother syncing logstd_init here as eval_agent is just for inference
                    self.eval_agent.load_state_dict(self.agent.state_dict())
                    
                    # Launch async eval
                    self.eval_thread = threading.Thread(
                        target=self._evaluate_async,
                        args=(iteration,),
                        daemon=True
                    )
                    self.eval_thread.start()
                    print(f"  [Async] Eval launched in background (iteration {iteration})")
                else:
                    # Sync eval (blocking)
                    eval_start = time.time()
                    self._evaluate()
                    eval_duration = time.time() - eval_start
                    print(f"  Eval took {eval_duration:.2f}s")
                    if self.cfg.wandb.enabled:
                        wandb.log({"charts/eval_time": eval_duration}, step=self.global_step)
                    self._save_checkpoint(iteration)
        
        # Wait for any running async eval to complete before cleanup
        if self.async_eval and self.eval_thread is not None and self.eval_thread.is_alive():
            print("Waiting for async eval to complete...")
            self.eval_thread.join()
        
        self.envs.close()
        self.eval_envs.close()
        print("Training complete!")

    def _evaluate(self, agent=None):
        """Run evaluation episodes.
        
        Args:
            agent: Agent to use for evaluation. Defaults to self.agent.
                   For async eval, pass self.eval_agent to avoid race conditions.
        """
        if agent is None:
            agent = self.agent
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
            # CRITICAL: Flatten obs like train does, otherwise agent gets wrong input format
            obs_flat = self._flatten_obs(eval_obs)
            with torch.no_grad():
                eval_action = agent.get_action(obs_flat, deterministic=True)
            eval_obs, reward, terminated, truncated, eval_infos = self.eval_envs.step(eval_action)
            
            episode_rewards += reward
            
            # Accumulate reward components (check both top-level and final_info)
            reward_comps = None
            if "reward_components" in eval_infos:
                reward_comps = eval_infos["reward_components"]
            elif "final_info" in eval_infos and "reward_components" in eval_infos["final_info"]:
                reward_comps = eval_infos["final_info"]["reward_components"]
            
            if reward_comps is not None:
                for k, v in reward_comps.items():
                    # Handle GPU tensors (convert to scalar if needed)
                    val = v.item() if hasattr(v, 'item') else v
                    eval_reward_components[k] = eval_reward_components.get(k, 0) + val
                eval_component_count += 1
                
                # Collect per-step data for CSV export
                for env_idx in range(self.cfg.training.num_eval_envs):
                    step_data = {
                        "step": step,
                        "reward": reward[env_idx].item(),
                    }
                    # Add each reward component
                    for k, v in reward_comps.items():
                        # Handle GPU tensors
                        val = v.item() if hasattr(v, 'item') else v
                        step_data[k] = val
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
            
            # Save per-step reward data as CSV in eval-specific subfolder
            import csv
            from pathlib import Path
            video_dir_path = Path(self.video_dir)
            eval_subfolder = video_dir_path / f"eval{self.eval_count}"
            eval_subfolder.mkdir(exist_ok=True)
            
            for env_idx, steps in step_reward_data.items():
                if steps:  # Only save if we have data
                    csv_path = eval_subfolder / f"env{env_idx}_rewards.csv"
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=steps[0].keys())
                        writer.writeheader()
                        writer.writerows(steps)
            
            self.eval_count += 1  # Increment for next eval

    def _evaluate_async(self, iteration):
        """Run evaluation asynchronously in a background thread.
        
        Uses a separate CUDA stream and a dedicated eval_agent to avoid
        race conditions with the training agent.
        """
        eval_start = time.time()
        
        # Run all CUDA operations on a separate stream
        with torch.cuda.stream(self.eval_stream):
            # Use eval_agent (which has a copy of weights from when eval was triggered)
            # This is completely isolated from self.agent, no race conditions
            self._evaluate(agent=self.eval_agent)
        
        # Sync this stream before logging (ensure eval is complete)
        self.eval_stream.synchronize()
        
        eval_duration = time.time() - eval_start
        print(f"  [Async] Eval completed (iteration {iteration}, took {eval_duration:.2f}s)")
        
        if self.cfg.wandb.enabled:
            # Log from background thread (wandb is thread-safe)
            wandb.log({"charts/eval_time": eval_duration}, step=self.global_step)
        
        # Save checkpoint after eval completes
        self._save_checkpoint(iteration)

    def _save_checkpoint(self, iteration):
        """Save model checkpoint."""
        if self.cfg.save_model:
            output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            model_path = output_dir / f"ckpt_{iteration}.pt"
            torch.save(self.agent.state_dict(), model_path)
            torch.save(self.agent.state_dict(), output_dir / "latest.pt")

    def _async_split_videos(self):
        """Asynchronously split tiled eval videos into individual env videos."""
        from scripts.utils.split_video import split_videos_in_dir
        
        if not self.video_dir:
            return
        
        # Run in background thread (non-blocking)
        def split_task():
            split_videos_in_dir(
                self.video_dir,
                self.cfg.training.num_eval_envs,
                rgb_only=True
            )
        
        thread = threading.Thread(target=split_task, daemon=True)
        thread.start()
