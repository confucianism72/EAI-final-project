#!/usr/bin/env python3
"""
Preprocess eai-dataset to undistort front camera images.
Creates a new dataset with undistorted images for LeRobot training.

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
import numpy as np
import cv2

# Camera calibration parameters
MTX = np.array([
    [570.217, 0., 327.460],
    [0., 570.180, 260.836],
    [0., 0., 1.]
], dtype=np.float64)
DIST = np.array([-0.735, 0.949, 0.000189, -0.00200, -0.864], dtype=np.float64)

def undistort_video(input_path: Path, output_path: Path, alpha: float, verbose: bool = True):
    """Undistort a video file using ffmpeg.
    
    Since ffmpeg's lensfun filter requires a camera database entry, we use a two-step process:
    1. Decode AV1 to raw frames with ffmpeg
    2. Apply undistortion with opencv
    3. Encode back to mp4
    """
    import tempfile
    import os
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get video info
    probe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 "{input_path}"'
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
    
    # Compute undistortion maps
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(MTX, DIST, None, new_mtx, (w, h), cv2.CV_32FC1)
    
    # Create temp directory for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract frames
        extract_cmd = f'ffmpeg -y -i "{input_path}" -vsync 0 "{tmpdir}/frame_%06d.png" 2>/dev/null'
        subprocess.run(extract_cmd, shell=True, check=True)
        
        # Process each frame
        frame_files = sorted(Path(tmpdir).glob('frame_*.png'))
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
                cv2.imwrite(str(frame_file), undistorted)
        
        # Encode back to video
        encode_cmd = f'ffmpeg -y -framerate {fps} -i "{tmpdir}/frame_%06d.png" -c:v libx264 -pix_fmt yuv420p "{output_path}" 2>/dev/null'
        subprocess.run(encode_cmd, shell=True, check=True)
        
        if verbose:
            print(f"  Processed {len(frame_files)} frames")
    
    return True


def preprocess_dataset(task: str, alpha: float, src_root: Path, dst_root: Path):
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
    wrist_src = videos_dir / 'observation.images.wrist'
    wrist_dst = dst_dir / 'videos' / 'observation.images.wrist'
    if wrist_src.exists():
        print("\nCopying wrist camera videos...")
        shutil.copytree(wrist_src, wrist_dst, dirs_exist_ok=True)
    
    # Undistort front camera videos
    front_src = videos_dir / 'observation.images.front'
    front_dst = dst_dir / 'videos' / 'observation.images.front'
    if front_src.exists():
        print("\nUndistorting front camera videos...")
        video_files = list(front_src.rglob('*.mp4'))
        for i, video_file in enumerate(video_files):
            relative_path = video_file.relative_to(front_src)
            output_file = front_dst / relative_path
            print(f"  [{i+1}/{len(video_files)}] {relative_path}")
            undistort_video(video_file, output_file, alpha)
    
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
    parser.add_argument('--all', action='store_true', help='Process all tasks')
    args = parser.parse_args()
    
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    
    if args.all:
        for task in ['lift', 'stack', 'sort']:
            preprocess_dataset(task, args.alpha, src_root, dst_root)
    else:
        preprocess_dataset(args.task, args.alpha, src_root, dst_root)


if __name__ == '__main__':
    main()
