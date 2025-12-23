#!/usr/bin/env python3
"""
Analyze trajectory data from eai-dataset to determine:
1. Episode lengths (steps and time)
2. Action deltas per step (for action space reference)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

# Dataset root
DATASET_ROOT = Path("/home/admin/Desktop/eai-final-project/eai-dataset")

def load_all_parquets(task_dir: Path) -> pd.DataFrame:
    """Load all parquet files from a task directory."""
    data_dir = task_dir / "data"
    all_dfs = []
    for parquet_file in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def analyze_task(task_name: str) -> dict:
    """Analyze a single task's trajectory data."""
    task_dir = DATASET_ROOT / task_name
    
    # Load meta info
    with open(task_dir / "meta" / "info.json") as f:
        info = json.load(f)
    
    fps = info["fps"]
    total_episodes = info["total_episodes"]
    total_frames = info["total_frames"]
    action_dim = info["features"]["action"]["shape"][0]
    action_names = info["features"]["action"]["names"]
    
    # Load trajectory data
    df = load_all_parquets(task_dir)
    
    # Episode lengths
    episode_lengths = df.groupby("episode_index")["frame_index"].max() + 1  # +1 because 0-indexed
    
    # Episode times (in seconds)
    episode_times = df.groupby("episode_index")["timestamp"].max()
    
    # Convert action column to numpy array
    # Action is stored as a list in each row
    actions = np.stack(df["action"].values)  # Shape: (N, action_dim)
    episode_indices = df["episode_index"].values
    
    # Calculate action deltas per step (consecutive differences within each episode)
    action_deltas = []
    for ep_idx in range(total_episodes):
        ep_mask = episode_indices == ep_idx
        ep_actions = actions[ep_mask]
        if len(ep_actions) > 1:
            deltas = np.diff(ep_actions, axis=0)  # Shape: (T-1, action_dim)
            action_deltas.append(deltas)
    
    all_deltas = np.vstack(action_deltas) if action_deltas else np.array([])
    
    return {
        "task_name": task_name,
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "action_dim": action_dim,
        "action_names": action_names,
        "episode_lengths": {
            "min": int(episode_lengths.min()),
            "max": int(episode_lengths.max()),
            "mean": float(episode_lengths.mean()),
            "std": float(episode_lengths.std()),
            "median": float(episode_lengths.median()),
        },
        "episode_times_seconds": {
            "min": float(episode_times.min()),
            "max": float(episode_times.max()),
            "mean": float(episode_times.mean()),
            "std": float(episode_times.std()),
            "median": float(episode_times.median()),
        },
        "action_deltas_per_step": {
            "names": action_names,
            "abs_mean": all_deltas.mean(axis=0).tolist() if len(all_deltas) > 0 else [],
            "abs_std": all_deltas.std(axis=0).tolist() if len(all_deltas) > 0 else [],
            "abs_max": np.abs(all_deltas).max(axis=0).tolist() if len(all_deltas) > 0 else [],
            "abs_mean_overall": float(np.abs(all_deltas).mean()) if len(all_deltas) > 0 else 0,
            "abs_max_overall": float(np.abs(all_deltas).max()) if len(all_deltas) > 0 else 0,
            "percentile_95": np.percentile(np.abs(all_deltas), 95, axis=0).tolist() if len(all_deltas) > 0 else [],
            "percentile_99": np.percentile(np.abs(all_deltas), 99, axis=0).tolist() if len(all_deltas) > 0 else [],
        },
    }

def main():
    print("=" * 80)
    print("EAI Dataset Trajectory Analysis")
    print("=" * 80)
    
    tasks = ["lift", "sort", "stack"]
    results = {}
    
    for task in tasks:
        print(f"\nAnalyzing task: {task}")
        print("-" * 40)
        result = analyze_task(task)
        results[task] = result
        
        print(f"  Total episodes: {result['total_episodes']}")
        print(f"  Total frames: {result['total_frames']}")
        print(f"  FPS: {result['fps']}")
        print(f"  Action dimension: {result['action_dim']}")
        
        print(f"\n  Episode Lengths (steps):")
        el = result["episode_lengths"]
        print(f"    Min: {el['min']}, Max: {el['max']}")
        print(f"    Mean: {el['mean']:.1f} ± {el['std']:.1f}")
        print(f"    Median: {el['median']:.1f}")
        
        print(f"\n  Episode Times (seconds):")
        et = result["episode_times_seconds"]
        print(f"    Min: {et['min']:.2f}s, Max: {et['max']:.2f}s")
        print(f"    Mean: {et['mean']:.2f}s ± {et['std']:.2f}s")
        print(f"    Median: {et['median']:.2f}s")
        
        print(f"\n  Action Deltas per Step (degrees):")
        ad = result["action_deltas_per_step"]
        print(f"    Overall abs mean: {ad['abs_mean_overall']:.4f}")
        print(f"    Overall abs max: {ad['abs_max_overall']:.4f}")
        print(f"\n    Per-joint analysis:")
        for i, name in enumerate(ad['names']):
            if i < len(ad['abs_mean']):
                print(f"      {name}: mean={ad['abs_mean'][i]:.4f}, max={ad['abs_max'][i]:.4f}, "
                      f"p95={ad['percentile_95'][i]:.4f}, p99={ad['percentile_99'][i]:.4f}")
    
    # Summary recommendations
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    all_max_steps = [r["episode_lengths"]["max"] for r in results.values()]
    all_mean_steps = [r["episode_lengths"]["mean"] for r in results.values()]
    
    print(f"\n1. MAX STEP (Episode Length):")
    print(f"   - Across all tasks, max episode length: {max(all_max_steps)} steps")
    print(f"   - Mean episode lengths: {[f'{m:.0f}' for m in all_mean_steps]}")
    print(f"   - Recommended max_step: {int(max(all_max_steps) * 1.2)} (20% buffer)")
    
    print(f"\n2. ACTION SPACE (Per-step Movement):")
    for task, r in results.items():
        ad = r["action_deltas_per_step"]
        print(f"\n   {task.upper()} (FPS={r['fps']}):")
        print(f"     Max single-step delta: {ad['abs_max_overall']:.2f} degrees")
        print(f"     95th percentile delta: {max(ad['percentile_95']):.2f} degrees")
        print(f"     Recommended action bound: ~{max(ad['percentile_99']) * 1.1:.1f} degrees")
    
    # Save results to JSON
    output_path = DATASET_ROOT.parent / "analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
