import cv2
import numpy as np

# Data from Track1Details.pdf / distort.py / undistort.py
W, H = 640, 480
image_size = (W, H)

mtx = np.array([
    [570.21740069, 0., 327.45975405],
    [0., 570.1797441, 260.83642155],
    [0., 0., 1.]
], dtype=np.float64)

dist = np.array([
    -0.735413911, 0.949258417, 0.000189059234, -0.00200351391, -0.864150312
], dtype=np.float64)

# Calculate the Optimal New Camera Matrix with alpha=1.0 (Keep all pixels)
# This represents the "Theoretical Pinhole Camera" that sees everything the distorted camera sees.
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, 1.0, image_size)

# Focal lengths
f_real_x = mtx[0, 0]
f_sim_x = new_mtx[0, 0]

# FOV Calculations
fov_x_real_linear = 2 * np.degrees(np.arctan(W / (2 * f_real_x))) # If it were linear
fov_x_sim = 2 * np.degrees(np.arctan(W / (2 * f_sim_x)))         # The required linear FOV

# Zoom/Scale Ratio needed at the center
# Real camera zooms in (f=570) compared to the wide-angle needed to cover the edges (f=sim)
scale_ratio = f_real_x / f_sim_x

print(f"Original Focal Length (Real): {f_real_x:.2f}")
print(f"New Focal Length (Sim/Alpha=1): {f_sim_x:.2f}")
print(f"Sim Required FOV (Horizontal): {fov_x_sim:.2f} degrees")
print(f"Scale Ratio (Real/Sim): {scale_ratio:.2f}")
print(f"Recommended Width: {W * scale_ratio:.2f}")
print(f"Recommended Height: {H * scale_ratio:.2f}")