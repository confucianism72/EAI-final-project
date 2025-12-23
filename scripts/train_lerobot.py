#!/usr/bin/env python3
"""
Train Diffusion Policy on eai-dataset using LeRobot.

Usage:
    source lerobot/.venv/bin/activate
    python scripts/train_lerobot.py --task lift
    python scripts/train_lerobot.py --task stack
    python scripts/train_lerobot.py --task sort
"""

import argparse
from pathlib import Path
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Policy on eai-dataset')
    parser.add_argument('--task', type=str, default='lift', choices=['lift', 'stack', 'sort'])
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=2000)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    # Dataset path (undistorted)
    dataset_path = Path(f'/home/admin/Desktop/eai-final-project/eai-dataset-undistorted/{args.task}')
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please run preprocessing first:")
        print(f"  python scripts/preprocess_undistort.py --task {args.task}")
        return
    
    # Output directory
    if args.output_dir is None:
        output_dir = Path(f'outputs/lerobot/{args.task}_diffusion')
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset metadata
    print(f"\nLoading dataset from {dataset_path}...")
    dataset = LeRobotDataset(str(dataset_path))
    
    print(f"  Episodes: {dataset.meta.total_episodes}")
    print(f"  Frames: {dataset.meta.total_frames}")
    print(f"  FPS: {dataset.meta.fps}")
    
    # Get features for policy
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print(f"\nInput features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")
    
    # Create policy
    print("\nCreating Diffusion Policy...")
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    
    # Create pre/post processors
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    
    # Setup delta timestamps for Diffusion Policy
    delta_timestamps = {
        "observation.images.front": [i / dataset.meta.fps for i in cfg.observation_delta_indices],
        "observation.images.wrist": [i / dataset.meta.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset.meta.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset.meta.fps for i in cfg.action_delta_indices],
    }
    
    # Reload dataset with delta timestamps
    dataset = LeRobotDataset(str(dataset_path), delta_timestamps=delta_timestamps)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nStarting training for {args.steps} steps...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output dir: {output_dir}")
    
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, output_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            
            if step % args.log_freq == 0:
                print(f"Step {step}/{args.steps} - Loss: {loss.item():.4f}")
            
            if step % args.save_freq == 0:
                checkpoint_path = output_dir / f"checkpoint_{step}"
                print(f"Saving checkpoint to {checkpoint_path}")
                policy.save_pretrained(checkpoint_path)
                preprocessor.save_pretrained(checkpoint_path)
                postprocessor.save_pretrained(checkpoint_path)
            
            if step >= args.steps:
                done = True
                break
    
    # Save final model
    final_path = output_dir / "final"
    print(f"\nSaving final model to {final_path}")
    policy.save_pretrained(final_path)
    preprocessor.save_pretrained(final_path)
    postprocessor.save_pretrained(final_path)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
