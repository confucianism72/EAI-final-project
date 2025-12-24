"""
Benchmark environment initialization and reset overhead.
Compares Domain Randomization ON vs OFF at different num_envs scales.
"""
import time
import torch
import gymnasium as gym
import numpy as np

import sys
sys.path.insert(0, "/home/admin/Desktop/eai-final-project")

# Register environment
from scripts.track1_env import Track1Env


def benchmark_env_creation(num_envs: int, domain_randomization: bool, obs_mode: str = "state"):
    """Benchmark environment creation time."""
    print(f"\n--- num_envs={num_envs}, DR={domain_randomization}, obs={obs_mode} ---")
    
    env_kwargs = dict(
        task="lift",
        control_mode="pd_joint_target_delta_pos",
        camera_mode="direct_pinhole",
        obs_mode=obs_mode,
        domain_randomization=domain_randomization,
        reward_mode="dense",
        render_mode="all",
        sim_backend="physx_cuda",
    )
    
    # Measure creation time
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    env = gym.make(
        "Track1-v0",
        num_envs=num_envs,
        max_episode_steps=355,
        **env_kwargs
    )
    
    torch.cuda.synchronize()
    creation_time = time.perf_counter() - start
    print(f"  Creation time: {creation_time:.2f}s")
    
    # Measure first reset time
    torch.cuda.synchronize()
    start = time.perf_counter()
    obs, info = env.reset()
    torch.cuda.synchronize()
    first_reset_time = time.perf_counter() - start
    print(f"  First reset:   {first_reset_time:.2f}s")
    
    # Warmup steps
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    
    # Measure subsequent reset time (average of 5)
    reset_times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        env.reset()
        torch.cuda.synchronize()
        reset_times.append(time.perf_counter() - start)
    
    avg_reset_time = np.mean(reset_times)
    print(f"  Avg reset:     {avg_reset_time*1000:.2f}ms")
    
    # Measure step time
    torch.cuda.synchronize()
    start = time.perf_counter()
    num_steps = 100
    for _ in range(num_steps):
        action = env.action_space.sample()
        env.step(action)
    torch.cuda.synchronize()
    step_time = (time.perf_counter() - start) / num_steps * 1000
    sps = num_envs / (step_time / 1000)
    print(f"  Step time:     {step_time:.2f}ms ({sps:,.0f} SPS)")
    
    env.close()
    
    return {
        "num_envs": num_envs,
        "domain_randomization": domain_randomization,
        "obs_mode": obs_mode,
        "creation_time": creation_time,
        "first_reset_time": first_reset_time,
        "avg_reset_time": avg_reset_time,
        "step_time_ms": step_time,
        "sps": sps,
    }


def main():
    print("=" * 60)
    print("Environment Initialization & Reset Benchmark")
    print("=" * 60)
    
    results = []
    
    # Test different configurations
    configs = [
        # Small scale
        (128, True, "state"),
        (128, False, "state"),
        # Medium scale
        (512, True, "state"),
        (512, False, "state"),
        # Large scale
        (2048, True, "state"),
        (2048, False, "state"),
        # RGB comparison (expensive)
        # (128, False, "rgb"),
    ]
    
    for num_envs, dr, obs_mode in configs:
        try:
            result = benchmark_env_creation(num_envs, dr, obs_mode)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Envs':<8} {'DR':<6} {'Obs':<8} {'Create(s)':<12} {'Reset(ms)':<12} {'Step(ms)':<10} {'SPS':<12}")
    print("-" * 80)
    
    for r in results:
        dr_str = "ON" if r["domain_randomization"] else "OFF"
        print(f"{r['num_envs']:<8} {dr_str:<6} {r['obs_mode']:<8} "
              f"{r['creation_time']:<12.2f} {r['avg_reset_time']*1000:<12.2f} "
              f"{r['step_time_ms']:<10.2f} {r['sps']:<12,.0f}")
    
    # Speedup analysis
    print("\n" + "=" * 60)
    print("SPEEDUP ANALYSIS (DR OFF vs ON)")
    print("=" * 60)
    
    for num_envs in [128, 512, 2048]:
        dr_on = next((r for r in results if r["num_envs"] == num_envs and r["domain_randomization"]), None)
        dr_off = next((r for r in results if r["num_envs"] == num_envs and not r["domain_randomization"]), None)
        
        if dr_on and dr_off:
            creation_speedup = dr_on["creation_time"] / dr_off["creation_time"]
            step_speedup = dr_on["step_time_ms"] / dr_off["step_time_ms"]
            print(f"num_envs={num_envs}: Creation {creation_speedup:.2f}x faster, Step {step_speedup:.2f}x faster")


if __name__ == "__main__":
    main()
