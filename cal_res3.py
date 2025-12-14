import numpy as np
import cv2

# --- 1. 输入参数 ---
W_real = 640
H_real = 480

# 真实内参
K_real = np.array([
    [570.21740069, 0.0, 327.45975405],
    [0.0, 570.1797441, 260.83642155],
    [0.0, 0.0, 1.0]
])

# 真实畸变
D_real = np.array([-0.735413911, 0.949258417, 0.000189059, -0.002003513, -0.864150312])

# --- 2. 暴力扫描所有点获取归一化边界 ---
# 生成所有像素坐标网格
grid_y, grid_x = np.indices((H_real, W_real)) 
points_all = np.stack((grid_x, grid_y), axis=-1).astype(np.float32).reshape(-1, 1, 2)

# 计算归一化坐标 (Normalized Coordinates)
# 这一步得到的是物理世界中光线的"角度斜率" (x/z, y/z)，与焦距无关
points_undistorted = cv2.undistortPoints(points_all, K_real, D_real, P=None)
points_undistorted = points_undistorted.reshape(-1, 2)

# 获取边界
x_min, x_max = points_undistorted[:, 0].min(), points_undistorted[:, 0].max()
y_min, y_max = points_undistorted[:, 1].min(), points_undistorted[:, 1].max()

print(f"归一化范围: X[{x_min:.4f}, {x_max:.4f}], Y[{y_min:.4f}, {y_max:.4f}]")

# --- 3. 分别计算针孔参数 (FX, FY 分离) ---

# 直接继承真实相机的 fx, fy，保证中心区域 1:1 清晰度匹配
fx_pinhole = K_real[0, 0]
fy_pinhole = K_real[1, 1]

# 分别计算宽和高
W_pinhole = int(np.ceil(fx_pinhole * (x_max - x_min)))
H_pinhole = int(np.ceil(fy_pinhole * (y_max - y_min)))

# 分别计算光心 (Principal Point)
cx_pinhole = -x_min * fx_pinhole
cy_pinhole = -y_min * fy_pinhole

# 构建新的内参矩阵
K_pinhole = np.array([
    [fx_pinhole, 0, cx_pinhole],
    [0, fy_pinhole, cy_pinhole],
    [0, 0, 1]
])

print("-" * 30)
print("【最终针孔相机参数 (Fx, Fy 独立)】")
print(f"分辨率 (W x H): {W_pinhole} x {H_pinhole}")
print(f"内参 K_pinhole:\n{K_pinhole}")
print(f"水平FOV覆盖: {np.degrees(np.arctan(x_max) - np.arctan(x_min)):.1f} 度")
print(f"垂直FOV覆盖: {np.degrees(np.arctan(y_max) - np.arctan(y_min)):.1f} 度")
print("-" * 30)