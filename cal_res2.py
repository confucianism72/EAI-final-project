import numpy as np
import cv2

def calculate_optimal_pinhole_parameters(W_real, H_real, K_real, D_real):
    """
    自动计算能够完美覆盖任意畸变（桶形/枕形）且保证全图清晰的针孔参数。
    """
    
    # 1. 构造密集网格
    # 我们不仅看四个角，要看全图，因为最严重的拉伸可能出现在任何地方
    grid_y, grid_x = np.indices((H_real, W_real))
    points_distorted = np.stack((grid_x, grid_y), axis=-1).astype(np.float32).reshape(-1, 1, 2)

    # 2. 计算归一化坐标 (Normalized Coordinates)
    # points_norm 里的数值代表 x/z 和 y/z
    points_norm = cv2.undistortPoints(points_distorted, K_real, D_real, P=None)
    points_norm = points_norm.reshape(H_real, W_real, 2)
    
    # 3. 计算局部采样密度 (Local Sampling Density)
    # 我们需要知道：在真实图像上移动 1 个像素，在归一化平面上移动了多少？
    
    # 水平方向的步长 (梯度)
    # diff_x[i,j] = points_norm[i, j+1] - points_norm[i, j]
    grad_x = points_norm[:, 1:, 0] - points_norm[:, :-1, 0] # 只看 x 坐标的变化
    
    # 垂直方向的步长
    # diff_y[i,j] = points_norm[i+1, j] - points_norm[i, j]
    grad_y = points_norm[1:, :, 1] - points_norm[:-1, :, 1] # 只看 y 坐标的变化
    
    # 为了安全，取绝对值并忽略极小值(防止除零)
    grad_x = np.abs(grad_x)
    grad_y = np.abs(grad_y)
    
    # 找到全图中"步长最小"的地方
    # 步长越小，说明真实镜头把那里"放大/拉伸"得越厉害，针孔就需要越高的焦距去匹配
    min_step_x = np.min(grad_x[grad_x > 1e-6]) # 避免 0
    min_step_y = np.min(grad_y[grad_y > 1e-6])
    
    print(f"最小归一化步长: X={min_step_x:.6f}, Y={min_step_y:.6f}")

    # 4. 计算最佳焦距 (Optimal Focal Length)
    # 公式：f * min_step >= 1.0 (保证至少 1:1 的采样)
    # 我们还必须设定一个下限：不能小于真实焦距 (保证中心清晰)
    fx_optimal = max(K_real[0,0], 1.0 / min_step_x)
    fy_optimal = max(K_real[1,1], 1.0 / min_step_y)
    
    # 5. 计算画布尺寸 (Canvas Size)
    # 基于新的焦距和视野边界
    x_min, x_max = points_norm[:, :, 0].min(), points_norm[:, :, 0].max()
    y_min, y_max = points_norm[:, :, 1].min(), points_norm[:, :, 1].max()
    
    W_pinhole = int(np.ceil(fx_optimal * (x_max - x_min)))
    H_pinhole = int(np.ceil(fy_optimal * (y_max - y_min)))
    
    cx_pinhole = -x_min * fx_optimal
    cy_pinhole = -y_min * fy_optimal
    
    K_pinhole = np.array([
        [fx_optimal, 0, cx_pinhole],
        [0, fy_optimal, cy_pinhole],
        [0, 0, 1]
    ])
    
    return W_pinhole, H_pinhole, K_pinhole

# --- 测试 ---
W_real, H_real = 640, 480
K_real = np.array([[570.22, 0, 327.46], [0, 570.18, 260.84], [0, 0, 1]])

# 测试 1: 你的桶形畸变 (应该不需要增加焦距)
D_barrel = np.array([-0.735, 0.949, 0.000189, -0.002, -0.864])
print("\n--- 桶形畸变测试 ---")
W, H, K = calculate_optimal_pinhole_parameters(W_real, H_real, K_real, D_barrel)
print(f"推荐分辨率: {W}x{H}")
print(f"推荐焦距 fx: {K[0,0]:.2f} (原: {K_real[0,0]:.2f})")
print(f"推荐焦距 fy: {K[1,1]:.2f} (原: {K_real[1,1]:.2f})")

# # 测试 2: 模拟一个强枕形畸变 (应该会显著增加焦距)
# D_pincushion = np.array([0.5, 0.0, 0.0, 0.0, 0.0]) # 正数 = 枕形
# print("\n--- 枕形畸变测试 ---")
# W, H, K = calculate_optimal_pinhole_parameters(W_real, H_real, K_real, D_pincushion)
# print(f"推荐分辨率: {W}x{H}")
# print(f"推荐焦距 fx: {K[0,0]:.2f} (原: {K_real[0,0]:.2f})")