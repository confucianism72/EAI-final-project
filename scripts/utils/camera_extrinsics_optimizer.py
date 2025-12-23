#!/usr/bin/env python3
"""
Camera Extrinsics Optimizer using Distance Transform.

Optimizes camera extrinsic parameters to align simulated 3D grid lines 
with real camera image using distance transform loss.

Coordinate System (from user):
- World +Y → Image down
- World +X → Image left  
- Optical axis = World -Z (camera looks down)
"""

import argparse
import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# Image dimensions
W, H = 640, 480

# Camera intrinsic (fixed, already calibrated)
MTX_INTRINSIC = np.array([
    [570.21740069, 0., 327.45975405],
    [0., 570.1797441, 260.83642155],
    [0., 0., 1.]
], dtype=np.float64)

DIST_COEFFS = np.array([
    -0.735413911, 0.949258417, 0.000189059234, -0.00200351391, -0.864150312
], dtype=np.float64)

# Sapien projection matrix (from sapien_proj_matrix.npy)
# This is obtained once from the environment and kept fixed
PROJ_MATRIX = np.array([
    [1.7819294, -0., -0.02331173, 0.],
    [0., -2.3757489, -0.08681846, 0.],
    [0., 0., -1.0001, -0.010001],
    [0., 0., -1., 0.]
], dtype=np.float64)


def get_grid_3d_points(num_samples_per_line: int = 30, num_samples_across_width: int = 3) -> np.ndarray:
    """
    Extract 3D grid line points for the small rectangle (x1-x2, y1-y2).
    Skip the y2 line (has reflective issues).
    
    Samples multiple points across the tape width for better gradient.
    """
    tape_half_width = 0.009
    w = tape_half_width
    x_1 = 0.204
    x_4 = 0.6
    y_1 = 0.15
    upper_height = 0.164

    x1 = x_1                           # 0.204
    x2 = x_4 - 0.204 - 2 * w           # ~0.378
    y1 = y_1                           # 0.15
    y2 = y_1 + upper_height + 2 * w    # ~0.332
    z = 0.001

    points = []
    
    # Width offsets: sample across tape width (perpendicular to tape direction)
    width_offsets = np.linspace(-w, w, num_samples_across_width)
    
    # 3 lines (skip y2 which has reflective issues):
    
    # Row at y1 (horizontal): from x1 to x2
    # Width is perpendicular = in Y direction
    for t in np.linspace(x1, x2 + 2*w, num_samples_per_line):
        for offset in width_offsets:
            points.append([t, y1 + w + offset, z])
    
    # Col at x1 (vertical): from y1 to y2
    # Width is perpendicular = in X direction
    for t in np.linspace(y1, y2 + 2*w, num_samples_per_line):
        for offset in width_offsets:
            points.append([x1 + w + offset, t, z])
    
    # Col at x2 (vertical): from y1 to y2
    # Width is perpendicular = in X direction
    for t in np.linspace(y1, y2 + 2*w, num_samples_per_line):
        for offset in width_offsets:
            points.append([x2 + w + offset, t, z])
    
    # Row at y2 (horizontal, partial): from x1 to midpoint (x1+x2)/2
    # The right half has reflective issues, so only use left half
    x_mid = (x1 + x2) / 2
    for t in np.linspace(x1, x_mid, num_samples_per_line // 2):
        for offset in width_offsets:
            points.append([t, y2 + w + offset, z])
    
    return np.array(points, dtype=np.float64)


def preprocess_real_image(image_path: str, visualize: bool = False) -> tuple:
    """Preprocess real image for distance transform optimization."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Undistort using alpha=0
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(MTX_INTRINSIC, DIST_COEFFS, (W, H), 0.0, (W, H))
    undistorted = cv2.undistort(img, MTX_INTRINSIC, DIST_COEFFS, None, new_mtx)
    
    # HSV filter for black tape (excludes red cubes)
    hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 100, 80])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Morphological closing to fill small gaps
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    closed_h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h)
    closed_v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v)
    binary_inv = cv2.bitwise_or(closed_h, closed_v)  # White = tape
    
    # Distance transform
    binary = cv2.bitwise_not(binary_inv)
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    if visualize:
        cv2.imwrite('debug_undistorted.png', undistorted)
        cv2.imwrite('debug_binary.png', binary_inv)
        cv2.imwrite('debug_distance_field.png', (dist_map * 5).astype(np.uint8))
        print("Saved debug images")
    
    return dist_map, undistorted, binary_inv, new_mtx


def params_to_extrinsic(params: np.ndarray) -> np.ndarray:
    """
    Convert 6-DOF parameters to 4x4 extrinsic matrix (cam2world).
    
    params: [tx, ty, tz, rx, ry, rz]
        - tx, ty, tz: translation (camera position in world)
        - rx, ry, rz: rotation (Euler angles in radians, XYZ order)
    
    Returns: 4x4 cam2world matrix
    """
    tx, ty, tz, rx, ry, rz = params
    
    # Build rotation matrix from Euler angles
    R = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
    
    # Build 4x4 matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = [tx, ty, tz]
    
    return extrinsic


def extrinsic_to_params(extrinsic: np.ndarray) -> np.ndarray:
    """Convert 4x4 extrinsic matrix to 6-DOF parameters."""
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    euler = Rotation.from_matrix(R).as_euler('xyz')
    
    return np.array([t[0], t[1], t[2], euler[0], euler[1], euler[2]])


def project_points(pts_world: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Project 3D world points to 2D pixel coordinates using cv2.projectPoints.
    
    Uses the coordinate system:
    - World +Y → Image down
    - World +X → Image left
    - Optical axis = World -Z
    
    params: [tx, ty, tz, rx, ry, rz] where rotation is small delta from initial
    """
    extrinsic = params_to_extrinsic(params)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    # Convert cam2world to world2cam for cv2.projectPoints
    R_w2c = R.T
    t_w2c = -R.T @ t
    
    rvec, _ = cv2.Rodrigues(R_w2c)
    
    projected, _ = cv2.projectPoints(pts_world, rvec, t_w2c, MTX_INTRINSIC, None)
    
    return projected.reshape(-1, 2)


def objective_function(params: np.ndarray, 
                       grid_points_3d: np.ndarray,
                       dist_map: np.ndarray) -> float:
    """
    Loss function: mean distance from projected points to nearest tape line.
    """
    projected = project_points(grid_points_3d, params)
    
    total_dist = 0.0
    valid_count = 0
    
    for pt in projected:
        px, py = int(round(pt[0])), int(round(pt[1]))
        if 0 <= px < W and 0 <= py < H:
            total_dist += dist_map[py, px]
            valid_count += 1
    
    if valid_count == 0:
        return 1e6
    
    outside_penalty = (len(projected) - valid_count) * 50.0
    return total_dist / valid_count + outside_penalty


def visualize_projection(image: np.ndarray, 
                         grid_points_3d: np.ndarray,
                         params: np.ndarray,
                         color: tuple = (0, 255, 0),
                         save_path: str = None) -> np.ndarray:
    """Visualize projected grid points on image."""
    vis_img = image.copy()
    projected = project_points(grid_points_3d, params)
    
    for pt in projected:
        px, py = int(round(pt[0])), int(round(pt[1]))
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(vis_img, (px, py), 3, color, -1)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"Saved: {save_path}")
    
    return vis_img


def optimize_camera_pose(real_image_path: str, 
                         visualize: bool = False,
                         max_iterations: int = 500) -> np.ndarray:
    """Main optimization routine."""
    
    # 1. Preprocess real image
    print("Preprocessing real image...")
    dist_map, undistorted, binary_inv, new_mtx = preprocess_real_image(real_image_path, visualize)
    
    # 2. Get 3D grid points
    print("Extracting 3D grid points...")
    grid_points_3d = get_grid_3d_points(num_samples_per_line=30)
    print(f"  Total grid points: {len(grid_points_3d)}")
    
    # 3. Initial guess based on user's coordinate system:
    # World +Y → Image down, World +X → Image left, Optical axis = -Z
    # R_world2cam = diag(-1, 1, -1), which equals R_cam2world (orthogonal)
    initial_extrinsic = np.array([
        [-1., 0., 0., 0.316],
        [0., 1., 0., 0.26],
        [0., 0., -1., 0.407],
        [0., 0., 0., 1.]
    ])
    initial_params = extrinsic_to_params(initial_extrinsic)
    
    print(f"Initial parameters (tx, ty, tz, rx, ry, rz):")
    print(f"  Translation: ({initial_params[0]:.4f}, {initial_params[1]:.4f}, {initial_params[2]:.4f})")
    print(f"  Rotation (rad): ({initial_params[3]:.4f}, {initial_params[4]:.4f}, {initial_params[5]:.4f})")
    
    # Visualize initial projection
    if visualize:
        visualize_projection(undistorted, grid_points_3d, initial_params,
                             color=(0, 0, 255), save_path="projection_initial.png")
    
    # 4. Compute initial loss
    initial_loss = objective_function(initial_params, grid_points_3d, dist_map)
    print(f"Initial loss: {initial_loss:.4f}")
    
    # 5. Optimize
    print("\nOptimizing camera pose...")
    
    iteration_count = [0]
    def callback(xk):
        iteration_count[0] += 1
        if iteration_count[0] % 50 == 0:
            loss = objective_function(xk, grid_points_3d, dist_map)
            print(f"  Iteration {iteration_count[0]}: loss = {loss:.4f}")
    
    result = minimize(
        objective_function,
        initial_params,
        args=(grid_points_3d, dist_map),
        method='Nelder-Mead',
        callback=callback,
        options={
            'maxiter': max_iterations,
            'xatol': 1e-6,
            'fatol': 1e-4,
            'disp': True
        }
    )
    
    optimized_params = result.x
    final_loss = objective_function(optimized_params, grid_points_3d, dist_map)
    
    print(f"\nOptimization complete!")
    print(f"  Converged: {result.success}")
    print(f"  Final loss: {final_loss:.4f} (improved by {initial_loss - final_loss:.4f})")
    
    print(f"\nOptimized parameters:")
    print(f"  Translation: ({optimized_params[0]:.6f}, {optimized_params[1]:.6f}, {optimized_params[2]:.6f})")
    print(f"  Rotation (rad): ({optimized_params[3]:.6f}, {optimized_params[4]:.6f}, {optimized_params[5]:.6f})")
    
    # Convert to extrinsic matrix for display
    opt_extrinsic = params_to_extrinsic(optimized_params)
    print(f"\n=== Optimized cam2world matrix ===")
    print(opt_extrinsic)
    
    # Visualize final projection
    if visualize:
        visualize_projection(undistorted, grid_points_3d, optimized_params,
                             color=(0, 255, 0), save_path="projection_optimized.png")
        
        vis_initial = visualize_projection(undistorted.copy(), grid_points_3d, initial_params,
                                           color=(0, 0, 255))
        vis_final = visualize_projection(vis_initial, grid_points_3d, optimized_params,
                                         color=(0, 255, 0))
        cv2.imwrite("projection_comparison.png", vis_final)
        print("Saved: projection_comparison.png (Red=initial, Green=optimized)")
    
    return optimized_params


def main():
    parser = argparse.ArgumentParser(description="Optimize camera extrinsics")
    parser.add_argument("--real-image", type=str, 
                        default="eai-2025-fall-final-project-reference-scripts/front_camera.png")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()
    
    optimize_camera_pose(args.real_image, args.visualize, args.max_iter)


if __name__ == "__main__":
    main()
