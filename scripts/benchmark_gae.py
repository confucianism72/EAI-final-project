"""
Benchmark script to compare JIT Script vs torch.compile for GAE function.
"""
import time
import torch

# ============================================================
# GAE Implementation (Pure Python, no decoration)
# ============================================================
def gae_python(
    rewards: torch.Tensor,
    vals: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float
):
    """Standard GAE calculation (pure Python)."""
    num_steps = rewards.shape[0]
    next_value = next_value.reshape(-1)
    
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    
    nextvalues = next_value
    
    for t in range(num_steps - 1, -1, -1):
        non_terminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * nextvalues * non_terminal - vals[t]
        lastgaelam = delta + gamma * gae_lambda * non_terminal * lastgaelam
        advantages[t] = lastgaelam
        nextvalues = vals[t]
        
    return advantages, advantages + vals


# ============================================================
# JIT Script version
# ============================================================
@torch.jit.script
def gae_jit(
    rewards: torch.Tensor,
    vals: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float
):
    """Standard GAE calculation (JIT Script)."""
    num_steps: int = rewards.shape[0]
    next_value = next_value.reshape(-1)
    
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    
    nextvalues = next_value
    
    for t in range(num_steps - 1, -1, -1):
        non_terminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * nextvalues * non_terminal - vals[t]
        lastgaelam = delta + gamma * gae_lambda * non_terminal * lastgaelam
        advantages[t] = lastgaelam
        nextvalues = vals[t]
        
    return advantages, advantages + vals


def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Parameters matching training config
    num_steps = 50
    num_envs = 2048*16
    gamma = 0.8
    gae_lambda = 0.9
    
    # Create test data
    rewards = torch.randn(num_steps, num_envs, device=device)
    vals = torch.randn(num_steps, num_envs, device=device)
    dones = torch.zeros(num_steps, num_envs, device=device, dtype=torch.bool)
    next_value = torch.randn(1, num_envs, device=device)
    next_done = torch.zeros(num_envs, device=device, dtype=torch.bool)
    
    # Warmup iterations and benchmark iterations
    warmup_iters = 10
    bench_iters = 100
    
    print(f"\nConfig: num_steps={num_steps}, num_envs={num_envs}")
    print(f"Warmup: {warmup_iters}, Benchmark: {bench_iters} iterations")
    print("=" * 60)
    
    # ============================================================
    # 1. JIT Script (already compiled at decoration time)
    # ============================================================
    print("\n[1] JIT Script")
    
    # JIT compilation happens at first call (or at decoration)
    # Time the first call which includes any remaining compilation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = gae_jit(rewards, vals, dones, next_value, next_done, gamma, gae_lambda)
    torch.cuda.synchronize()
    jit_compile_time = time.perf_counter() - t0
    print(f"  First call (incl. compile): {jit_compile_time*1000:.2f} ms")
    
    # Warmup
    for _ in range(warmup_iters):
        _ = gae_jit(rewards, vals, dones, next_value, next_done, gamma, gae_lambda)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        _ = gae_jit(rewards, vals, dones, next_value, next_done, gamma, gae_lambda)
    torch.cuda.synchronize()
    jit_time = (time.perf_counter() - t0) / bench_iters * 1000
    print(f"  Avg runtime: {jit_time:.3f} ms/iter")
    
    # ============================================================
    # 2. torch.compile (default mode)
    # ============================================================
    print("\n[2] torch.compile (default mode)")
    
    gae_compiled = torch.compile(gae_python)
    
    # First call triggers compilation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = gae_compiled(rewards, vals, dones, next_value, next_done, gamma, gae_lambda)
    torch.cuda.synchronize()
    compile_time = time.perf_counter() - t0
    print(f"  First call (incl. compile): {compile_time*1000:.2f} ms")
    
    # Warmup
    for _ in range(warmup_iters):
        _ = gae_compiled(rewards, vals, dones, next_value, next_done, gamma, gae_lambda)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        _ = gae_compiled(rewards, vals, dones, next_value, next_done, gamma, gae_lambda)
    torch.cuda.synchronize()
    compiled_time = (time.perf_counter() - t0) / bench_iters * 1000
    print(f"  Avg runtime: {compiled_time:.3f} ms/iter")
    
    # ============================================================
    # 3. torch.compile (reduce-overhead mode)
    # ============================================================
    print("\n[3] torch.compile (reduce-overhead mode)")
    
    gae_reduce_overhead = torch.compile(gae_python, mode="reduce-overhead")
    
    # First call triggers compilation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = gae_reduce_overhead(rewards, vals, dones, next_value, next_done, gamma, gae_lambda)
    torch.cuda.synchronize()
    reduce_compile_time = time.perf_counter() - t0
    print(f"  First call (incl. compile): {reduce_compile_time*1000:.2f} ms")
    
    # Warmup
    for _ in range(warmup_iters):
        _ = gae_reduce_overhead(rewards, vals, dones, next_value, next_done, gamma, gae_lambda)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        _ = gae_reduce_overhead(rewards, vals, dones, next_value, next_done, gamma, gae_lambda)
    torch.cuda.synchronize()
    reduce_time = (time.perf_counter() - t0) / bench_iters * 1000
    print(f"  Avg runtime: {reduce_time:.3f} ms/iter")
    
    # ============================================================
    # 4. Pure Python (baseline)
    # ============================================================
    print("\n[4] Pure Python (no optimization)")
    
    # Warmup
    for _ in range(warmup_iters):
        _ = gae_python(rewards, vals, dones, next_value, next_done, gamma, gae_lambda)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        _ = gae_python(rewards, vals, dones, next_value, next_done, gamma, gae_lambda)
    torch.cuda.synchronize()
    python_time = (time.perf_counter() - t0) / bench_iters * 1000
    print(f"  Avg runtime: {python_time:.3f} ms/iter")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'Compile (ms)':<15} {'Runtime (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    print(f"{'Pure Python':<30} {'N/A':<15} {python_time:<15.3f} {'1.00x':<10}")
    print(f"{'JIT Script':<30} {jit_compile_time*1000:<15.2f} {jit_time:<15.3f} {python_time/jit_time:<10.2f}x")
    print(f"{'torch.compile (default)':<30} {compile_time*1000:<15.2f} {compiled_time:<15.3f} {python_time/compiled_time:<10.2f}x")
    print(f"{'torch.compile (reduce-overhead)':<30} {reduce_compile_time*1000:<15.2f} {reduce_time:<15.3f} {python_time/reduce_time:<10.2f}x")


if __name__ == "__main__":
    benchmark()
