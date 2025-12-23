#!/usr/bin/env python3
"""
Preprocess eai-dataset to undistort front camera images.
Creates a new dataset with undistorted images for LeRobot training.

Optimized version: Uses ffmpeg pipes to avoid disk I/O for intermediate frames.

Usage:
    python scripts/preprocess_undistort.py --task lift --alpha 0.25
    python scripts/preprocess_undistort.py --task stack --alpha 0.25
    python scripts/preprocess_undistort.py --task sort --alpha 0.25
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2

# Camera calibration parameters
MTX = np.array([
    [570.217, 0., 327.460],
    [0., 570.180, 260.836],
    [0., 0., 1.]
], dtype=np.float64)
DIST = np.array([-0.735, 0.949, 0.000189, -0.00200, -0.864], dtype=np.float64)


def undistort_video_pipe(input_path: Path, output_path: Path, alpha: float, verbose: bool = True):
    """Undistort a video using ffmpeg pipes (no disk I/O for intermediate frames)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get video info
    probe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of csv=p=0 "{input_path}"'
    result = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error probing {input_path}")
        return False
    
    parts = result.stdout.strip().split(',')
    w, h = int(parts[0]), int(parts[1])
    fps_str = parts[2]
    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        fps = num // den if den > 0 else 30
    else:
        fps = int(float(fps_str))
    
    # Compute undistortion maps once
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(MTX, DIST, None, new_mtx, (w, h), cv2.CV_32FC1)
    
    # FFmpeg decode pipe
    decode_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-i', str(input_path),
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'
    ]
    
    # FFmpeg encode pipe
    encode_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
        str(output_path)
    ]
    
    # Start pipes
    decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    frame_size = w * h * 3
    frame_count = 0
    
    try:
        while True:
            raw_frame = decode_proc.stdout.read(frame_size)
            if len(raw_frame) < frame_size:
                break
            
            # Convert to numpy, undistort, write back
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            encode_proc.stdin.write(undistorted.tobytes())
            frame_count += 1
    finally:
        decode_proc.stdout.close()
        encode_proc.stdin.close()
        decode_proc.wait()
        encode_proc.wait()
    
    if verbose:
        print(f"  Processed {frame_count} frames")
    
    return True


def process_video_wrapper(args):
    """Wrapper for parallel processing."""
    input_path, output_path, alpha, idx, total = args
    print(f"  [{idx}/{total}] {input_path.name}")
    return undistort_video_pipe(input_path, output_path, alpha, verbose=True)


def preprocess_dataset(task: str, alpha: float, src_root: Path, dst_root: Path, num_workers: int = 2):
    """Preprocess a task dataset to undistort front camera images."""
    src_dir = src_root / task
    dst_dir = dst_root / task
    
    if not src_dir.exists():
        print(f"Source directory {src_dir} does not exist!")
        return
    
    print(f"\n{'='*60}")
    print(f"Preprocessing {task} dataset")
    print(f"Alpha: {alpha}")
    print(f"Source: {src_dir}")
    print(f"Destination: {dst_dir}")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}")
    
    # Create destination directory
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy metadata
    print("\nCopying metadata...")
    shutil.copytree(src_dir / 'meta', dst_dir / 'meta', dirs_exist_ok=True)
    
    # Copy data (parquet files - no modification needed)
    print("Copying data files...")
    shutil.copytree(src_dir / 'data', dst_dir / 'data', dirs_exist_ok=True)
    
    # Process videos
    videos_dir = src_dir / 'videos'
    if not videos_dir.exists():
        print("No videos directory found!")
        return
    
    # Copy wrist camera videos as-is (no distortion)
    # Handles: wrist (single-arm), left_wrist/right_wrist (dual-arm)
    for video_subdir in videos_dir.iterdir():
        if video_subdir.is_dir() and 'front' not in video_subdir.name:
            dst_subdir = dst_dir / 'videos' / video_subdir.name
            print(f"\nCopying {video_subdir.name}...")
            shutil.copytree(video_subdir, dst_subdir, dirs_exist_ok=True)
    
    # Undistort front camera videos
    front_src = videos_dir / 'observation.images.front'
    front_dst = dst_dir / 'videos' / 'observation.images.front'
    if front_src.exists():
        print("\nUndistorting front camera videos...")
        video_files = list(front_src.rglob('*.mp4'))
        
        # Parallel processing
        tasks = [
            (video_file, front_dst / video_file.relative_to(front_src), alpha, i+1, len(video_files))
            for i, video_file in enumerate(video_files)
        ]
        
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(process_video_wrapper, tasks))
        else:
            for t in tasks:
                process_video_wrapper(t)
    
    # Update info.json to mark as preprocessed
    info_path = dst_dir / 'meta' / 'info.json'
    with open(info_path) as f:
        info = json.load(f)
    info['preprocessing'] = {
        'undistort_alpha': alpha,
        'front_camera_undistorted': True
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"\n✓ Done! Preprocessed dataset saved to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description='Undistort front camera in eai-dataset')
    parser.add_argument('--task', type=str, default='lift', choices=['lift', 'stack', 'sort'])
    parser.add_argument('--alpha', type=float, default=0.25, help='Undistortion alpha (default: 0.25)')
    parser.add_argument('--src', type=str, default='/home/admin/Desktop/eai-final-project/eai-dataset',
                        help='Source dataset root')
    parser.add_argument('--dst', type=str, default='/home/admin/Desktop/eai-final-project/eai-dataset-undistorted',
                        help='Destination dataset root')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    parser.add_argument('--all', action='store_true', help='Process all tasks')
    args = parser.parse_args()
    
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    
    if args.all:
        for task in ['lift', 'stack', 'sort']:
            preprocess_dataset(task, args.alpha, src_root, dst_root, args.workers)
    else:
        preprocess_dataset(args.task, args.alpha, src_root, dst_root, args.workers)


if __name__ == '__main__':
    main()
