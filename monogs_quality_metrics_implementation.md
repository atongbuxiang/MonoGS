# MonoGS 实物扫描三项评价指标实现方案

本文档只保留三项最方便在实物实验中实现的评价指标：

```text
1. 验证视角重建一致性：PSNR / SSIM
2. 终端覆盖率：Gaussian 中心的体素占用率
3. 末端路径长度：相邻末端位姿平移距离累加
```

这三项指标均建议在实验结束后离线统一计算。在线质量反馈 `q_t` 可以继续作为策略输入使用，但论文评价表中不再单独报告渲染残差、不确定性、重复观测率、`T_80`、jerk 或单位路径覆盖增益。

## 1. MonoGS 中可用的数据

MonoGS 渲染接口位于：

```text
gaussian_splatting/gaussian_renderer/__init__.py
```

调用：

```python
render_pkg = render(viewpoint, gaussians, pipeline_params, background)
```

可得到当前视角渲染图：

```python
render_pkg["render"]   # [3, H, W], RGB 渲染图，取值约为 [0, 1]
```

Gaussian 模型位于：

```text
gaussian_splatting/scene/gaussian_model.py
```

覆盖率计算需要：

```python
gaussians.get_xyz       # [N, 3], Gaussian 中心
gaussians.get_opacity   # [N, 1], Gaussian opacity
```

末端路径长度建议直接使用机器人控制端记录的末端相机位姿。如果只使用 MonoGS 估计位姿，也可以从每帧 `Camera` 对象中的 `R, T` 还原相机位姿，但实物实验中更推荐使用 Franka 正运动学得到的末端位姿日志。

## 2. 指标一：验证视角 PSNR / SSIM

### 2.1 定义

实验结束后，选取不参与重建优化的验证视角。对每个验证视角 `v`，真实图像为 `I_v`，Gaussian map 渲染图像为 `\hat{I}_v`。

均方误差：

```text
MSE_v = mean((I_v - \hat{I}_v)^2)
```

若图像归一化到 `[0, 1]`，则：

```text
PSNR_v = -10 log10(MSE_v)
```

最终报告所有验证视角的平均值：

```text
PSNR = (1 / |V|) sum_{v in V} PSNR_v
SSIM = (1 / |V|) sum_{v in V} SSIM(I_v, \hat{I}_v)
```

### 2.2 直接使用 MonoGS 现有实现

MonoGS 已经在 `utils/eval_utils.py` 中提供 `eval_rendering()`，运行：

```bash
python slam.py --config configs/mono/tum/fr3_office.yaml --eval
```

会在 headless 模式下运行并输出渲染指标。核心代码逻辑如下：

```python
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim


rendering = render(frame, gaussians, pipe, background)["render"]
image = torch.clamp(rendering, 0.0, 1.0)

mask = gt_image > 0
psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
ssim_score = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
```

### 2.3 建议新增的简化接口

如果实物实验有单独保存的验证帧，建议新增一个离线脚本，例如：

```text
scripts/eval_real_scan_metrics.py
```

其中 PSNR / SSIM 部分可以写成：

```python
import numpy as np
import torch

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim


@torch.no_grad()
def eval_validation_rendering(validation_frames, gaussians, pipe, background):
    psnr_values = []
    ssim_values = []

    for frame in validation_frames:
        gt = frame.original_image.cuda()
        pred = render(frame, gaussians, pipe, background)["render"]
        pred = torch.clamp(pred, 0.0, 1.0)

        # 若没有目标 mask，使用非黑像素作为有效区域；如果有分割 mask，应替换为目标区域 mask。
        mask = gt > 0
        psnr_score = psnr(pred[mask].unsqueeze(0), gt[mask].unsqueeze(0))
        ssim_score = ssim(pred.unsqueeze(0), gt.unsqueeze(0))

        psnr_values.append(psnr_score.item())
        ssim_values.append(ssim_score.item())

    return {
        "mean_psnr": float(np.mean(psnr_values)),
        "mean_ssim": float(np.mean(ssim_values)),
    }
```

论文中报告：

```text
PSNR ↑, SSIM ↑
```

其中箭头表示数值越大越好。

## 3. 指标二：终端覆盖率

### 3.1 定义

终端覆盖率用于近似扫描结束后目标区域被 Gaussian map 覆盖的程度。给定目标包围盒：

```text
bbox_min = [x_min, y_min, z_min]
bbox_max = [x_max, y_max, z_max]
```

将包围盒划分为 `G x G x G` 个体素。若体素内至少存在一个满足 opacity 阈值的 Gaussian 中心，则该体素记为已覆盖：

```text
O(i, j, k) = 1, if exists Gaussian g:
             xyz_g in voxel(i, j, k) and opacity_g > tau_opacity
```

终端覆盖率为：

```text
Coverage = sum_{i,j,k} O(i,j,k) / G^3
```

如果有目标有效体素 mask，也可以把分母从 `G^3` 改成有效体素数。但为了实现最简单，建议论文实验中先使用固定目标包围盒内的全部体素作为分母。

### 3.2 配置参数

建议在配置文件或评估脚本参数中写清楚：

```yaml
EvalMetrics:
  bbox_min: [-0.30, -0.30, 0.00]
  bbox_max: [ 0.30,  0.30, 0.50]
  coverage_grid_size: 32
  gaussian_opacity_th: 0.2
```

这些参数要和实物工作台坐标、MonoGS 重建尺度保持一致。如果使用单目 MonoGS，需要注意尺度漂移；如果机械臂末端位姿能提供尺度约束，则覆盖率更稳定。

### 3.3 代码实现

建议在新增脚本或工具文件中实现：

```python
import torch


@torch.no_grad()
def compute_terminal_coverage(
    gaussians,
    bbox_min,
    bbox_max,
    grid_size=32,
    opacity_th=0.2,
):
    """
    Args:
        gaussians: MonoGS GaussianModel
        bbox_min: list[float] or tuple[float], shape [3]
        bbox_max: list[float] or tuple[float], shape [3]
        grid_size: int, voxel grid resolution per axis
        opacity_th: float, opacity threshold for valid Gaussians

    Returns:
        coverage: float, occupied voxel ratio in [0, 1]
        occupied_count: int
        total_count: int
    """
    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach().squeeze(-1)

    device = xyz.device
    dtype = xyz.dtype
    bbox_min = torch.tensor(bbox_min, device=device, dtype=dtype)
    bbox_max = torch.tensor(bbox_max, device=device, dtype=dtype)

    in_box = ((xyz >= bbox_min) & (xyz <= bbox_max)).all(dim=1)
    valid = in_box & (opacity > opacity_th)

    total_count = int(grid_size ** 3)
    if valid.sum() == 0:
        return 0.0, 0, total_count

    xyz_norm = (xyz[valid] - bbox_min) / (bbox_max - bbox_min).clamp_min(1e-6)
    voxel_idx = torch.floor(xyz_norm * grid_size).long()
    voxel_idx = torch.clamp(voxel_idx, 0, grid_size - 1)

    linear_idx = (
        voxel_idx[:, 0] * grid_size * grid_size
        + voxel_idx[:, 1] * grid_size
        + voxel_idx[:, 2]
    )

    occupied_count = int(torch.unique(linear_idx).numel())
    coverage = occupied_count / total_count
    return float(coverage), occupied_count, total_count
```

调用示例：

```python
coverage, occupied, total = compute_terminal_coverage(
    gaussians=gaussians,
    bbox_min=[-0.30, -0.30, 0.00],
    bbox_max=[0.30, 0.30, 0.50],
    grid_size=32,
    opacity_th=0.2,
)

print(f"coverage={coverage:.4f}, occupied={occupied}, total={total}")
```

论文中报告：

```text
Coverage ↑
```

其中数值越大表示终端重建覆盖越完整。

## 4. 指标三：末端路径长度

### 4.1 定义

末端路径长度用于衡量机器人为了完成扫描移动了多远。设实验过程中记录了 `N` 个末端相机或末端执行器位置：

```text
p_1, p_2, ..., p_N,    p_i in R^3
```

则路径长度为：

```text
L_path = sum_{i=2}^{N} ||p_i - p_{i-1}||_2
```

单位通常为米。若所有方法使用相同时间预算，路径越短且重建指标越好，说明扫描效率越高。

### 4.2 推荐数据来源

优先使用 Franka 控制端记录的末端位姿，例如每个控制周期保存：

```text
timestamp, x, y, z, qx, qy, qz, qw
```

路径长度只需要 `x, y, z`。姿态四元数可以保留用于画轨迹或后续分析，但本文不再单独计算视角稳定性。

如果只能使用 MonoGS 位姿，可从每帧相机外参还原相机中心。但实物实验中，机器人正运动学的尺度通常比单目 SLAM 位姿更可靠。

### 4.3 CSV 日志格式

建议机器人端保存：

```csv
timestamp,x,y,z,qx,qy,qz,qw
0.000,0.421,0.032,0.315,0.0,0.0,0.0,1.0
0.033,0.422,0.034,0.315,0.0,0.0,0.0,1.0
```

### 4.4 代码实现

```python
import csv
import numpy as np


def load_ee_positions_from_csv(csv_path):
    positions = []
    timestamps = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            positions.append([
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            ])

    return np.asarray(timestamps), np.asarray(positions, dtype=np.float64)


def compute_path_length(positions):
    """
    Args:
        positions: np.ndarray, shape [N, 3]

    Returns:
        path_length: float, accumulated translation distance
    """
    if len(positions) < 2:
        return 0.0

    delta = positions[1:] - positions[:-1]
    segment_length = np.linalg.norm(delta, axis=1)
    return float(segment_length.sum())
```

调用示例：

```python
timestamps, positions = load_ee_positions_from_csv("logs/ours_ee_pose.csv")
path_length = compute_path_length(positions)
duration = timestamps[-1] - timestamps[0]

print(f"path_length={path_length:.4f} m")
print(f"duration={duration:.4f} s")
```

本文只报告末端路径长度。如果所有方法固定扫描时间，`duration` 只作为实验设置说明，不作为主指标。

论文中报告：

```text
Path Length ↓
```

其中数值越小表示完成同等扫描任务所需移动距离越短。

## 5. 统一离线评估脚本结构

建议新增：

```text
scripts/eval_real_scan_metrics.py
```

输出统一 JSON：

```json
{
  "mean_psnr": 24.81,
  "mean_ssim": 0.842,
  "coverage": 0.376,
  "path_length_m": 1.284
}
```

脚本主流程可以组织为：

```python
def evaluate_real_scan(
    validation_frames,
    gaussians,
    pipe,
    background,
    ee_pose_csv,
    bbox_min,
    bbox_max,
    grid_size=32,
    opacity_th=0.2,
):
    rendering_metrics = eval_validation_rendering(
        validation_frames=validation_frames,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
    )

    coverage, occupied, total = compute_terminal_coverage(
        gaussians=gaussians,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        grid_size=grid_size,
        opacity_th=opacity_th,
    )

    _, positions = load_ee_positions_from_csv(ee_pose_csv)
    path_length = compute_path_length(positions)

    return {
        "mean_psnr": rendering_metrics["mean_psnr"],
        "mean_ssim": rendering_metrics["mean_ssim"],
        "coverage": coverage,
        "coverage_occupied_voxels": occupied,
        "coverage_total_voxels": total,
        "path_length_m": path_length,
    }
```

## 6. 论文表格建议

主对比表只保留四列指标：

```text
Method | PSNR ↑ | SSIM ↑ | Coverage ↑ | Path Length ↓
```

示例：

```text
Fixed-circle       22.31  0.801  0.312  1.86
Single-frame       23.05  0.815  0.334  1.52
Memory-only        23.74  0.829  0.351  1.43
Quality-only       23.61  0.827  0.359  1.49
Ours               24.80  0.846  0.382  1.31
```

注意：这里的数字只是表格格式示例，不是实验结果。

## 7. 实现注意事项

1. 验证视角必须尽量不参与 MonoGS 优化，否则 `PSNR/SSIM` 会偏乐观。

2. 覆盖率依赖 `bbox_min / bbox_max / grid_size / opacity_th`，所有方法必须使用完全相同的参数。

3. 若单目 MonoGS 的尺度不稳定，应优先使用机器人坐标或固定标定关系确定目标包围盒。

4. 末端路径长度优先从 Franka 正运动学日志计算，不建议用不同方法各自估计的 SLAM 轨迹直接比较。

5. 不再报告重复观测率、`T_80`、动作平滑性、视角稳定性和单位路径覆盖增益；这些指标实现成本高，且容易引入额外阈值和同步误差。
