"""
Overlay script to compare simulation camera view with real camera view.
Now uses the camera_mode API for proper distortion handling.

Usage:
    python -m scripts.camera_overlay --camera-mode distorted
    python -m scripts.camera_overlay --camera-mode distort-twice
    python -m scripts.camera_overlay --camera-mode direct_pinhole
"""
import numpy as np
from PIL import Image
import gymnasium as gym
import argparse
import cv2
from scripts.track1_env import Track1Env


def create_overlay(sim_image: np.ndarray, real_image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create an overlay of sim and real images for comparison."""
    # Resize real image to match sim if needed
    if sim_image.shape[:2] != real_image.shape[:2]:
        real_image = cv2.resize(real_image, (sim_image.shape[1], sim_image.shape[0]))
    
    # Convert to same format
    if len(sim_image.shape) == 3 and sim_image.shape[2] == 4:
        sim_image = sim_image[:, :, :3]
    if len(real_image.shape) == 3 and real_image.shape[2] == 4:
        real_image = real_image[:, :, :3]
    
    # Create overlay
    overlay = cv2.addWeighted(sim_image, alpha, real_image, 1 - alpha, 0)
    
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Compare sim and real camera views")
    parser.add_argument("--real-image", type=str, 
                        default="eai-2025-fall-final-project-reference-scripts/front_camera.png",
                        help="Path to real camera image")
    parser.add_argument("--task", type=str, default="lift", choices=["lift", "stack", "sort"])
    parser.add_argument("--camera-mode", type=str, default="distorted",
                        choices=["distorted", "distort-twice", "direct_pinhole"],
                        help="Camera mode for simulation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend alpha (0=real, 1=sim)")
    parser.add_argument("--output", type=str, default="overlay_comparison.png", help="Output path")
    args = parser.parse_args()
    
    # Load real image
    real_img = np.array(Image.open(args.real_image))
    print(f"Loaded real image: {args.real_image}, shape: {real_img.shape}")
    
    # For comparison with distorted mode, use raw real image
    # For comparison with distort-twice or direct_pinhole, undistort the real image
    if args.camera_mode in ["distort-twice", "direct_pinhole"]:
        # Undistort real image
        W, H = 640, 480
        mtx = np.array([
            [570.21740069, 0., 327.45975405],
            [0., 570.1797441, 260.83642155],
            [0., 0., 1.]
        ], dtype=np.float64)
        dist = np.array([-0.735413911, 0.949258417, 0.000189059234, -0.00200351391, -0.864150312], dtype=np.float64)
        
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (W, H), 0.0, (W, H))
        real_img = cv2.undistort(real_img, mtx, dist, None, new_mtx)
        print("Undistorted real image for comparison")
    
    # Create sim environment and get camera view
    print(f"Creating environment with camera_mode={args.camera_mode}")
    env = gym.make(
        "Track1-v0",
        render_mode=None,
        obs_mode="rgbd",
        reward_mode="none",
        task=args.task,
        domain_randomization=False,
        camera_mode=args.camera_mode,
        num_envs=1,
    )
    
    obs, _ = env.reset()
    
    # Get front camera image from sim
    sim_rgb = None
    if "sensor_data" in obs and "front_camera" in obs["sensor_data"]:
        if "rgb" in obs["sensor_data"]["front_camera"]:
            sim_rgb = obs["sensor_data"]["front_camera"]["rgb"]
    elif "image" in obs and "front_camera" in obs["image"]:
        if "rgb" in obs["image"]["front_camera"]:
            sim_rgb = obs["image"]["front_camera"]["rgb"]
    
    if sim_rgb is not None:
        # Convert to numpy if tensor
        if hasattr(sim_rgb, 'cpu'):
            sim_rgb = sim_rgb.cpu().numpy()
        
        # Remove batch dimension if present
        if len(sim_rgb.shape) == 4:
            sim_rgb = sim_rgb[0]
        
        # Convert to uint8 if float
        if sim_rgb.dtype in [np.float32, np.float64]:
            sim_rgb = (sim_rgb * 255).clip(0, 255).astype(np.uint8)
        
        print(f"Sim image shape: {sim_rgb.shape}")
        
        # Save sim image
        sim_filename = f"sim_camera_{args.camera_mode}.png"
        Image.fromarray(sim_rgb).save(sim_filename)
        print(f"Saved {sim_filename}")
        
        # Create overlay
        overlay = create_overlay(sim_rgb, real_img, alpha=args.alpha)
        
        # Create side-by-side comparison: Sim | Overlay | Real
        h, w = sim_rgb.shape[:2]
        real_resized = cv2.resize(real_img[:, :, :3] if len(real_img.shape) == 3 and real_img.shape[2] >= 3 else real_img, 
                                   (w, h))
        
        # Ensure all images are 3-channel
        sim_3ch = sim_rgb[:, :, :3] if sim_rgb.shape[2] >= 3 else sim_rgb
        overlay_3ch = overlay[:, :, :3] if overlay.shape[2] >= 3 else overlay
        real_3ch = real_resized[:, :, :3] if real_resized.shape[2] >= 3 else real_resized
        
        # Stack: Sim | Overlay | Real
        comparison = np.hstack([sim_3ch, overlay_3ch, real_3ch])
        
        # Add labels at bottom
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_y = h - 15  # Bottom position
        cv2.putText(comparison, "Sim", (w//2 - 20, label_y), font, 0.7, (0, 255, 255), 2)
        cv2.putText(comparison, "Overlay", (w + w//2 - 40, label_y), font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "Real", (2*w + w//2 - 25, label_y), font, 0.7, (255, 255, 255), 2)
        
        # Add vertical separator lines
        cv2.line(comparison, (w, 0), (w, h), (128, 128, 128), 2)
        cv2.line(comparison, (2*w, 0), (2*w, h), (128, 128, 128), 2)
        
        # Save
        Image.fromarray(comparison).save(args.output)
        print(f"Saved comparison to {args.output} ({comparison.shape[1]}x{comparison.shape[0]})")
        
        # Also save just the overlay
        Image.fromarray(overlay).save("overlay_only.png")
        print("Saved overlay_only.png")
        
    else:
        print("Error: Could not get front_camera from observations")
        print(f"Observation keys: {obs.keys()}")
    
    env.close()


if __name__ == "__main__":
    main()
