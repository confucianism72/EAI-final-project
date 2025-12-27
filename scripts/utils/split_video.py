"""
Split tiled parallel environment videos into individual videos.

ManiSkill's RecordEpisode wrapper tiles multiple parallel env renders 
into a single video using a sqrt(num_envs) x sqrt(num_envs) grid.

This script splits them back into individual videos using ffmpeg crop filter.

Usage:
    python scripts/utils/split_video.py path/to/video.mp4 --num_envs 8
    python scripts/utils/split_video.py path/to/videos/ --num_envs 8  # batch mode
"""

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


def get_video_info(video_path: str):
    """Get video resolution using cv2."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return width, height, fps, frames


def split_video(video_path: str, num_envs: int, output_dir: str = None, env_idx: int = None, 
                rgb_only: bool = False, cameras: int = 2):
    """
    Split a tiled video into individual environment videos using ffmpeg.
    
    Args:
        video_path: Path to input video
        num_envs: Number of parallel environments that were tiled
        output_dir: Output directory (default: same as input)
        env_idx: If specified, only extract this specific env (0-indexed)
        rgb_only: If True, only extract top half of each cell (RGB, ignore depth/segmentation)
        cameras: Number of cameras per environment (default=2 for single-arm: front+wrist)
    
    Returns:
        List of output video paths
    """
    video_path = Path(video_path)
    if output_dir is None:
        # Create subfolder for each source video
        output_dir = video_path.parent / "split" / video_path.stem
    else:
        output_dir = Path(output_dir) / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video info
    frame_width, frame_height, fps, total_frames = get_video_info(str(video_path))
    
    # Calculate grid dimensions (ManiSkill uses sqrt for nrows)
    nrows = int(np.sqrt(num_envs))
    ncols = math.ceil(num_envs / nrows)
    
    # Cell width = cameras × 640 (each camera is 640px wide)
    # Cell height = from frame_height / nrows
    cell_w = cameras * 640
    cell_h = frame_height // nrows
    
    # If rgb_only, only take top half of each cell (RGB is on top, position/seg on bottom)
    if rgb_only:
        cell_h = cell_h // 2
        print(f"Input video: {video_path.name} (RGB only mode - extracting top half)")
    else:
        print(f"Input video: {video_path.name}")
    
    print(f"  Resolution: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")
    print(f"  Grid: {nrows} rows x {ncols} cols, {cameras} cameras/env")
    print(f"  Cell size: {cell_w}x{cell_h}")
    
    # Determine which envs to extract
    if env_idx is not None:
        env_indices = [env_idx]
    else:
        env_indices = list(range(num_envs))
    
    output_paths = []
    
    for idx in env_indices:
        row = idx // ncols
        col = idx % ncols
        
        # Calculate crop position (top-left corner)
        x = col * cell_w
        # For rgb_only, we still need to calculate y based on full cell height
        if rgb_only:
            y = row * (cell_h * 2)  # Full cell height for row offset
        else:
            y = row * cell_h
        
        output_name = f"env{idx}.mp4"
        output_path = output_dir / output_name
        output_paths.append(output_path)
        
        # Use ffmpeg crop filter
        # crop=w:h:x:y
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"crop={cell_w}:{cell_h}:{x}:{y}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-an",  # No audio
            str(output_path)
        ]
        
        print(f"  Extracting env{idx}...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAILED")
            print(f"  Error: {result.stderr[:200]}")
        else:
            print("OK")
    
    print(f"Output videos saved to: {output_dir}")
    return output_paths


def split_videos_in_dir(video_dir: str, num_envs: int, rgb_only: bool = True, cameras: int = 2):
    """Split all tiled videos in a directory into individual env videos.
    
    This is the main function to import for batch processing.
    
    Args:
        video_dir: Directory containing mp4 files to split
        num_envs: Number of parallel environments
        rgb_only: Only extract RGB (top half), default True
        cameras: Number of cameras per env
    
    Returns:
        Number of videos processed
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        return 0
    
    # Find mp4 files that haven't been split yet
    video_files = list(video_dir.glob("*.mp4"))
    split_dir = video_dir / "split"
    
    videos_to_process = []
    for video_file in video_files:
        output_folder = split_dir / video_file.stem
        if not output_folder.exists():
            videos_to_process.append(video_file)
    
    if not videos_to_process:
        return 0
    
    print(f"Splitting {len(videos_to_process)} video(s)...")
    for video_file in videos_to_process:
        split_video(str(video_file), num_envs, None, None, rgb_only, cameras)
    
    return len(videos_to_process)


def main():
    parser = argparse.ArgumentParser(description="Split tiled parallel env videos")
    parser.add_argument("input", type=str, help="Input video file or directory")
    parser.add_argument("--num_envs", type=int, required=True, help="Number of parallel envs")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--env_idx", type=int, default=None, help="Only extract this env index (0-indexed)")
    parser.add_argument("--rgb_only", action="store_true", help="Only extract RGB (top half of each cell)")
    parser.add_argument("--cameras", type=int, default=2, help="Number of cameras per env (2 for single-arm, 3 for dual-arm)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Batch mode: process all mp4 files
        video_files = list(input_path.glob("*.mp4"))
        
        # Filter out videos that have already been split
        split_dir = input_path / "split"
        videos_to_process = []
        for video_file in video_files:
            output_folder = split_dir / video_file.stem
            if not output_folder.exists():
                videos_to_process.append(video_file)
        
        if not videos_to_process:
            # All videos already processed
            return
        
        print(f"Found {len(videos_to_process)} new video(s) to process (skipping {len(video_files) - len(videos_to_process)} already split)")
        for video_file in videos_to_process:
            split_video(str(video_file), args.num_envs, args.output_dir, args.env_idx, args.rgb_only, args.cameras)
    else:
        # Single file mode
        split_video(args.input, args.num_envs, args.output_dir, args.env_idx, args.rgb_only, args.cameras)


if __name__ == "__main__":
    main()
