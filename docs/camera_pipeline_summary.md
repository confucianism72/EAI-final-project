# Camera Pipeline & Sim-to-Real Alignment Summary

**Date:** 2025-12-14
**Context:** Debugging and refining the camera distortion pipeline in `Track1Env` to achieve accurate alignment with real-world camera images.

## 1. Core Problem & Solution

**Problem:** 
Initial attempts to simulate the real camera's distortion and intrinsic properties resulted in misalignment (especially visible in tape lines) due to incorrect handling of:
1.  Non-square pixels (`fx != fy`) in the undistorted view.
2.  Principal point (`cx`, `cy`) offsets being ignored/overridden.
3.  Complex manual FOV calculations that didn't match OpenCV's `getOptimalNewCameraMatrix`.

**Solution: High-Resolution Scaled Rendering**
Instead of manually calculating FOV, we now use a **scaled rendering approach**:
1.  **Render Scale**: Set `render_scale = 3` (configurable).
2.  **Source Resolution**: Render the pure pinhole image at a high resolution: `(640 * scale) x (480 * scale)` = **1920 x 1440**.
3.  **Intrinsic Scaling**: Use `cv2.getOptimalNewCameraMatrix` to compute the correct pinhole intrinsic for the *unscaled* (640x480) view, then simply **multiply the intrinsic matrix by `scale`**.
    *   This ensures the render camera perfectly matches the theoretical "undistorted" real camera, just at a higher pixel density.

## 2. Key Implementation Details (`scripts/track1_env.py`)

### A. Intrinsic Calculation (`_default_sensor_configs` & `_setup_camera_processing_maps`)
We explicitly use `cv2.getOptimalNewCameraMatrix` with `newImgSize=(width, height)` to let OpenCV handle the aspect ratio and principal point logic automatically.

```python
# Real Intrinsic & Distortion Coeffs
mtx_intrinsic = ... # (from calibration)
dist_coeffs = ...

# 1. Get base undistorted intrinsic for 640x480 result
# newImgSize=(width, height) automatically scales parameters
intrinsic, _ = cv2.getOptimalNewCameraMatrix(
    mtx_intrinsic, dist_coeffs, (640, 480), 1.0, (width, height)
)

# 2. Use this 'intrinsic' directly for the SAPIEN camera
```

### B. Distortion Logic (`_setup_camera_processing_maps`)
We use `cv2.undistortPoints` to map distorted output pixels back to the rendered pinhole space.

1.  **Grid Generation**: Create a meshgrid of target pixels $(u_d, v_d)$ for the 640x480 output.
2.  **Undistort**:
    ```python
    undistorted_pts = cv2.undistortPoints(
        points,
        cameraMatrix=self.mtx_intrinsic,
        distCoeffs=self.dist_coeffs,
        P=self.render_intrinsic  # Project directly to 1920x1440 render space
    )
    ```
    *   **Crucial Step**: Passing `P=self.render_intrinsic` tells OpenCV to project the ray into our scaled render camera's pixel coordinates.
3.  **Normalization**: Normalize these coordinates to `[-1, 1]` for `grid_sample`.
    *   Normalization divides by `(SRC_W - 1, SRC_H - 1)`.

### C. Optimization
*   **Import**: Moved `import torch.nn.functional as tFunc` to top-level to avoid function-call overhead.
*   **Device**: Grids are moved to `device` once during checking setup.

## 3. Camera Modes

| Mode | Description | Resolution | Pipeline |
| :--- | :--- | :--- | :--- |
| **`distorted`** | Simulates real camera distortion. | 640 x 480 | Render (1920x1440) $\to$ Distortion Grid $\to$ Output (640x480) |
| **`distort-twice`** | Debug mode. Distorts then Undistorts. | 640 x 480 | Render $\to$ Distort $\to$ Undistort (alpha=0) |
| **`direct_pinhole`** | Pure pinhole (alpha=0 equivalent). | 640 x 480 | Render parameters *directly match* alpha=0 undistorted instrinsics. No post-processing. |

## 4. Helper Tools

### `scripts/camera_overlay.py`
Visual alignment tool.
*   **Usage**: `python -m scripts.camera_overlay --camera-mode distorted`
*   **Features**:
    *   Side-by-side "Sim | Overlay | Real" view.
    *   Supports `distorted` and `distort-twice` modes.
    *   Auto-undistorts real image when comparing in non-distorted modes.

### `scripts/check_wrist_camera.py`
Simple script to capture wrist camera images.
*   **Purpose**: Verify orientation/linkage of wrist cameras.
*   **Usage**: `python -m scripts.check_wrist_camera`
*   **Current State**: Wrist cameras are hardcoded to 640x480, 50° FOV, no distortion processing (identity logic).

## 5. Future Maintenance
*   **Changing Resolution**: Adjust `self.render_scale` in `track1_env.py`.
*   **Calibration Update**: Update `mtx_intrinsic` and `dist_coeffs` in `track1_env.py` if real camera calibration changes.
*   **Wrist Cameras**: Currently static. If wrist camera distortion is needed, strictly follow the `front_camera` pattern (high-res render -> distort).
