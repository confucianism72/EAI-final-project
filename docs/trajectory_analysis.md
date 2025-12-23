# EAI Dataset Trajectory Analysis

分析真实轨迹数据，用于确定模拟环境的 `max_step` 和 `action_space` 参数。

## 1. Episode 长度分析 (Max Step 参考)

| 任务 | 最小 | 最大 | 平均 | 标准差 | 中位数 |
|------|------|------|------|--------|--------|
| **Lift** | 125 | 296 | 179 | 32 | 174 |
| **Sort** | 23 | 619 | 335 | 76 | 326 |
| **Stack** | 88 | 384 | 249 | 42 | 247 |

**推荐 `max_step`** (含 20% buffer):
- **Lift**: 355 steps
- **Sort**: 743 steps
- **Stack**: 461 steps

## 2. Episode 时间

| 任务 | 最小 | 最大 | 平均 | FPS |
|------|------|------|------|-----|
| **Lift** | 4.1s | 9.8s | 5.9s | 30 |
| **Sort** | 0.7s | 20.6s | 11.1s | 30 |
| **Stack** | 2.9s | 12.8s | 8.3s | 30 |

## 3. 每关节 Action Space 分析

### Lift/Stack 任务 (单臂, 6 DoF)

| 关节 | 最大单步 delta | 95th 百分位 | 99th 百分位 | 建议 bound (度) |
|------|----------------|-------------|-------------|-----------------|
| `shoulder_pan` | 2.95° | 1.17° | 1.87° | ±2.5 |
| `shoulder_lift` | 7.25° | 2.83° | 3.92° | ±5.0 |
| `elbow_flex` | 5.55° | 2.09° | 3.00° | ±4.0 |
| `wrist_flex` | 3.79° | 1.16° | 1.70° | ±2.5 |
| `wrist_roll` | 2.28° | 0.67° | 1.14° | ±1.5 |
| `gripper` | 6.53° | 2.03° | 3.11° | ±4.0 |

### Sort 任务 (双臂, 12 DoF)

**左臂:**

| 关节 | 最大单步 delta | 95th 百分位 | 99th 百分位 | 建议 bound (度) |
|------|----------------|-------------|-------------|-----------------|
| `left_shoulder_pan` | 6.24° | 2.16° | 3.36° | ±4.0 |
| `left_shoulder_lift` | 14.42° | 4.03° | 6.04° | ±7.0 |
| `left_elbow_flex` | 11.40° | 3.74° | 6.11° | ±7.0 |
| `left_wrist_flex` | 6.10° | 1.89° | 3.18° | ±4.0 |
| `left_wrist_roll` | 5.47° | 0.89° | 1.77° | ±2.5 |
| `left_gripper` | 9.64° | 3.19° | 5.54° | ±6.5 |

**右臂:**

| 关节 | 最大单步 delta | 95th 百分位 | 99th 百分位 | 建议 bound (度) |
|------|----------------|-------------|-------------|-----------------|
| `right_shoulder_pan` | 6.22° | 2.25° | 3.42° | ±4.0 |
| `right_shoulder_lift` | 10.25° | 4.08° | 5.98° | ±7.0 |
| `right_elbow_flex` | 13.56° | 3.73° | 5.83° | ±7.0 |
| `right_wrist_flex` | 8.11° | 1.70° | 3.01° | ±4.0 |
| `right_wrist_roll` | 3.48° | 0.67° | 1.35° | ±2.0 |
| `right_gripper` | 11.54° | 2.54° | 4.50° | ±5.5 |

## 4. 关键发现

1. **大关节 vs 小关节**: `shoulder_lift` 和 `elbow_flex` 需要最大的 action range
2. **精细控制**: `wrist_roll` 动作幅度最小，需要精细调整
3. **任务差异**: Sort 双臂任务的 action range 比单臂任务大约 1.5-2x

## 5. 推荐配置

```yaml
# 单臂任务 (Lift, Stack)
action_bounds_single_arm:
  shoulder_pan: 0.044    # ~2.5 degrees
  shoulder_lift: 0.087   # ~5.0 degrees
  elbow_flex: 0.070      # ~4.0 degrees
  wrist_flex: 0.044      # ~2.5 degrees
  wrist_roll: 0.026      # ~1.5 degrees
  gripper: 0.070         # ~4.0 degrees

# 双臂任务 (Sort)
action_bounds_dual_arm:
  shoulder_pan: 0.070    # ~4.0 degrees
  shoulder_lift: 0.122   # ~7.0 degrees
  elbow_flex: 0.122      # ~7.0 degrees
  wrist_flex: 0.070      # ~4.0 degrees
  wrist_roll: 0.044      # ~2.5 degrees
  gripper: 0.113         # ~6.5 degrees
```

## 6. 使用方法

### 默认配置

```bash
# Lift 任务 (使用 single_arm 配置)
python scripts/train.py env.task=lift

# Stack 任务 (使用 single_arm 配置)
python scripts/train.py env.task=stack

# Sort 任务 (使用 dual_arm 配置)
python scripts/train.py env.task=sort control=dual_arm
```

### 自定义 Action Bounds

```bash
# 覆盖单个关节的 bounds
python scripts/train.py control.action_bounds.shoulder_lift=0.1

# 使用完全自定义的 bounds
python scripts/train.py 'control.action_bounds={shoulder_pan: 0.05, shoulder_lift: 0.1, elbow_flex: 0.08, wrist_flex: 0.05, wrist_roll: 0.03, gripper: 0.08}'
```

### 配置文件

| 配置文件 | 任务 | 说明 |
|----------|------|------|
| `configs/control/single_arm.yaml` | Lift, Stack | 单臂任务的 per-joint bounds |
| `configs/control/dual_arm.yaml` | Sort | 双臂任务的 per-joint bounds (更大范围) |

---

*分析脚本: `scripts/analyze_trajectory.py`*  
*详细 JSON 数据: `analysis_results.json`*
