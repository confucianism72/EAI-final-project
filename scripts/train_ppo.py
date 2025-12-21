#!/usr/bin/env python3
"""PPO Training Script for Track1 Environment.

Based on ManiSkill's ppo_rgb.py baseline, adapted for Track1 tasks.

Usage:
    python scripts/train_ppo.py --task lift --num-envs 128
    python scripts/train_ppo.py --task lift --num-envs 512 --track  # with wandb
"""

from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Must run as: python -m scripts.train_ppo
from .track1_env import Track1Env


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Track1-PPO"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""
    save_model: bool = True
    """whether to save model"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint"""

    # Track1 specific arguments
    task: str = "lift"
    """task to train: lift, stack, or sort"""
    control_mode: str = "pd_joint_target_delta_pos"
    """control mode for the robot"""
    camera_mode: str = "direct_pinhole"
    """camera mode: direct_pinhole, distorted"""
    include_state: bool = True
    """whether to include proprioception state in observations"""

    # PPO hyperparameters
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    num_steps: int = 50
    """the number of steps per rollout"""
    num_eval_steps: int = 100
    """the number of steps for evaluation"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for GAE"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""

    # Runtime computed
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DictArray:
    """Helper class for handling dictionary observations in buffer."""
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                             torch.uint8 if v.dtype == np.uint8 else
                             torch.int32 if v.dtype == np.int32 else v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        else:
            for k, v in value.items():
                self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


class NatureCNN(nn.Module):
    """CNN feature extractor for RGB observations."""
    def __init__(self, sample_obs):
        super().__init__()
        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute flattened size
        with torch.no_grad():
            n_flatten = self.cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        self.out_features += feature_size

        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            self.state_fc = nn.Linear(state_size, 256)
            self.out_features += 256
        else:
            self.state_fc = None

    def forward(self, observations) -> torch.Tensor:
        encoded = []
        # Process RGB
        rgb = observations["rgb"].float().permute(0, 3, 1, 2) / 255.0
        encoded.append(self.fc(self.cnn(rgb)))
        
        # Process state if available
        if self.state_fc is not None and "state" in observations:
            encoded.append(self.state_fc(observations["state"]))
        
        return torch.cat(encoded, dim=1)


class Agent(nn.Module):
    """PPO Agent with actor-critic architecture."""
    def __init__(self, envs, sample_obs):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        
        action_dim = np.prod(envs.unwrapped.single_action_space.shape)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_value(self, x):
        return self.critic(self.feature_net(x))

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def make_env(args, num_envs, for_eval=False):
    """Create Track1 environment with proper wrappers."""
    env_kwargs = dict(
        task=args.task,
        control_mode=args.control_mode,
        camera_mode=args.camera_mode,
        obs_mode="rgb",
        reward_mode="sparse",  # TODO: implement dense reward
        render_mode="all",
        sim_backend="physx_cuda",
    )
    
    reconfiguration_freq = 1 if for_eval else None
    
    env = gym.make(
        "Track1-v0",
        num_envs=num_envs,
        reconfiguration_freq=reconfiguration_freq,
        **env_kwargs
    )
    
    # Flatten observations
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=args.include_state)
    
    return env


def main():
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        args.exp_name = f"track1_{args.task}"
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Create environments
    print(f"Creating {args.num_envs} training environments...")
    envs = make_env(args, args.num_envs)
    
    print(f"Creating {args.num_eval_envs} evaluation environments...")
    eval_envs = make_env(args, args.num_eval_envs, for_eval=True)

    # Add video recording for eval
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        print(f"Saving eval videos to {eval_output_dir}")
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=False,
            max_steps_per_video=args.num_eval_steps,
            video_fps=30
        )

    # Wrap with ManiSkillVectorEnv
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=True, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=True, record_metrics=True)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "Only continuous action space supported"

    # Setup logging
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )

    # Initialize agent
    print("Initializing agent...")
    next_obs, _ = envs.reset(seed=args.seed)
    agent = Agent(envs, sample_obs=next_obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        agent.load_state_dict(torch.load(args.checkpoint))

    # Storage setup
    obs_buffer = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Training loop
    global_step = 0
    start_time = time.time()
    next_done = torch.zeros(args.num_envs, device=device)

    print(f"\n{'='*60}")
    print(f"Training PPO on Track1 {args.task}")
    print(f"num_iterations={args.num_iterations}, num_envs={args.num_envs}")
    print(f"batch_size={args.batch_size}, minibatch_size={args.minibatch_size}")
    print(f"{'='*60}\n")

    for iteration in range(1, args.num_iterations + 1):
        print(f"Iteration {iteration}/{args.num_iterations}, global_step={global_step}")
        
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()

        # Evaluation
        if iteration % args.eval_freq == 1:
            print("Running evaluation...")
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_action = agent.get_action(eval_obs, deterministic=True)
                eval_obs, _, _, _, eval_infos = eval_envs.step(eval_action)
                
                if "final_info" in eval_infos:
                    mask = eval_infos["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)
            
            print(f"  Evaluated {args.num_eval_steps * args.num_eval_envs} steps, {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean_val = torch.stack(v).float().mean().item()
                writer.add_scalar(f"eval/{k}", mean_val, global_step)
                print(f"  eval/{k} = {mean_val:.4f}")

        # Save model
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        # Rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buffer[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = (terminations | truncations).float()
            rewards[step] = reward * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    writer.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, done_mask] = agent.get_value(infos["final_observation"]).flatten()

        # Compute advantages using GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + args.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            
            returns = advantages + values

        # Flatten batch
        b_obs = obs_buffer.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.flatten()
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Logging
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"  SPS: {sps}")

    # Save final model
    if args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"Final model saved to {model_path}")

    envs.close()
    eval_envs.close()
    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
