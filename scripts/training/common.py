"""
Common utilities for PPO training.
GPU-native normalization wrappers and environment setup.
"""
import gymnasium as gym
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Import Track1 environment
try:
    from scripts.track1_env import Track1Env
except ImportError:
    import sys
    import os
    sys.path.append(os.getcwd())
    from scripts.track1_env import Track1Env


class RunningMeanStd:
    """GPU-compatible running mean and standard deviation tracker."""
    
    def __init__(self, shape=(), device=None, epsilon=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.device = device
    
    def update(self, x: torch.Tensor):
        """Update statistics with a batch of observations."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class NormalizeObservationGPU(gym.Wrapper):
    """GPU-native observation normalization wrapper."""
    
    def __init__(self, env, device=None, epsilon=1e-8, clip=10.0):
        super().__init__(env)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.clip = clip
        
        # Determine observation shape
        obs_shape = env.single_observation_space.shape
        self.rms = RunningMeanStd(shape=obs_shape, device=self.device)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info
    
    def _normalize(self, obs):
        self.rms.update(obs)
        normalized = (obs - self.rms.mean) / torch.sqrt(self.rms.var + self.epsilon)
        return torch.clamp(normalized, -self.clip, self.clip)


class NormalizeRewardGPU(gym.Wrapper):
    """GPU-native reward normalization wrapper using discounted return variance."""
    
    def __init__(self, env, device=None, gamma=0.99, epsilon=1e-8, clip=10.0):
        super().__init__(env)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.rms = RunningMeanStd(shape=(), device=self.device)
        self.returns = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize returns tracker
        num_envs = obs.shape[0]
        self.returns = torch.zeros(num_envs, device=self.device)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update discounted returns
        self.returns = self.returns * self.gamma + reward
        self.rms.update(self.returns.unsqueeze(1))
        
        # Normalize reward
        normalized_reward = reward / torch.sqrt(self.rms.var + self.epsilon)
        normalized_reward = torch.clamp(normalized_reward, -self.clip, self.clip)
        
        # Reset returns for done environments
        done = terminated | truncated
        self.returns = self.returns * (~done).float()
        
        return obs, normalized_reward, terminated, truncated, info


class FlattenStateWrapper(gym.ObservationWrapper):
    """Flattens the dict observation into a single vector (for State mode).
    Handles GPU tensors efficiently.
    """
    def __init__(self, env):
        super().__init__(env)
        self.flat_dim = self._count_dim(env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.flat_dim,), 
            dtype=np.float32
        )
        
    def _count_dim(self, space):
        d = 0
        if isinstance(space, gym.spaces.Dict):
            for v in space.values():
                d += self._count_dim(v)
        elif isinstance(space, gym.spaces.Box):
            d += np.prod(space.shape)
        return d

    def observation(self, observation):
        return self._flatten_recursive(observation)

    def _flatten_recursive(self, obs):
        tensors = []
        if isinstance(obs, dict):
            for k in sorted(obs.keys()):
                tensors.append(self._flatten_recursive(obs[k]))
        else:
            if obs.ndim > 2:
                v = obs.flatten(start_dim=1)
            else:
                v = obs
            tensors.append(v)
        return torch.cat(tensors, dim=-1)


def make_env(cfg: DictConfig, num_envs: int, for_eval: bool = False, video_dir: str = None):
    """Create Track1 environment with proper wrappers."""
    reward_config = OmegaConf.to_container(cfg.reward, resolve=True) if "reward" in cfg else None
    
    # Get action_bounds from control config if available
    action_bounds = None
    if "control" in cfg and "action_bounds" in cfg.control:
        action_bounds = OmegaConf.to_container(cfg.control.action_bounds, resolve=True)
    
    # Get simulation frequencies (must match real robot for sim2real)
    sim_freq = cfg.env.get("sim_freq", 120)      # Physics simulation frequency
    control_freq = cfg.env.get("control_freq", 30)  # Control frequency (real robot is 30 Hz)
    
    # Build sim_config dict for ManiSkill
    sim_config = {
        "sim_freq": sim_freq,
        "control_freq": control_freq,
    }
    
    # Get camera config options
    camera_extrinsic = None
    undistort_alpha = 0.25  # Default
    if "camera" in cfg.env:
        if "extrinsic" in cfg.env.camera:
            camera_extrinsic = OmegaConf.to_container(cfg.env.camera.extrinsic, resolve=True)
        if "undistort_alpha" in cfg.env.camera:
            undistort_alpha = cfg.env.camera.undistort_alpha
    
    env_kwargs = dict(
        task=cfg.env.task,
        control_mode=cfg.env.control_mode,
        camera_mode=cfg.env.camera_mode,
        obs_mode=cfg.env.obs_mode,
        reward_mode=cfg.reward.reward_mode if "reward" in cfg else "sparse",
        reward_config=reward_config,
        action_bounds=action_bounds,
        camera_extrinsic=camera_extrinsic,
        undistort_alpha=undistort_alpha,
        sim_config=sim_config,
        render_mode="all",
        sim_backend="physx_cuda",
    )
    
    reconfiguration_freq = 1 if for_eval else None
    
    # Compute max_episode_steps from base * multiplier (or use legacy format)
    if "episode_steps" in cfg.env:
        base = cfg.env.episode_steps.get("base", 296)
        multiplier = cfg.env.episode_steps.get("multiplier", 1.2)
        max_episode_steps = int(base * multiplier)
    else:
        max_episode_steps = cfg.env.get("max_episode_steps", None)
    
    env = gym.make(
        cfg.env.env_id,
        num_envs=num_envs,
        reconfiguration_freq=reconfiguration_freq,
        max_episode_steps=max_episode_steps,
        **env_kwargs
    )
    
    # Video Recording
    if for_eval and video_dir and cfg.capture_video:
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            max_steps_per_video=cfg.training.num_eval_steps,
            video_fps=30
        )

    # Flatten observations
    if cfg.env.obs_mode == "state":
        env = FlattenStateWrapper(env)
    else:
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=cfg.env.include_state)
    
    # Wrap with ManiSkillVectorEnv
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)
    
    return env
