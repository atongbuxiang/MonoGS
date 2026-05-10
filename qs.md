# MonoGS 中 q_t 三类质量指标的实时实现方案

本文档说明如何在 MonoGS 仓库中实时计算质量向量：

```text
q_t = [q_t^{cov}, q_t^{unc}, q_t^{res}]
```

这套方案只保留三个实时评估指标：覆盖率、重建不确定性代理、渲染残差。最终送入 VLA 策略的质量反馈是 **3 维向量**。如果后续再拼接机械臂关节状态、末端位姿或语言特征，那些属于策略状态的其他部分，不计入这里的 `q_t` 维度。

需要注意，MonoGS 不会直接输出这三个论文级指标，但它已经提供了实时构造这些指标所需的底层量，包括当前帧渲染图、opacity、depth、Gaussian 可见性、Gaussian 尺度和观测次数。

## 1. MonoGS 中可用的实时信号

MonoGS 的渲染函数位于：

```text
gaussian_splatting/gaussian_renderer/__init__.py
```

调用：

```python
render_pkg = render(viewpoint, gaussians, pipeline_params, background)
```

会返回：

```python
{
    "render": rendered_image,          # [3, H, W] 当前视角 RGB 渲染图
    "viewspace_points": screenspace_points,
    "visibility_filter": radii > 0,    # [N] 当前视角可见 Gaussian
    "radii": radii,                    # [N] Gaussian 屏幕半径
    "depth": depth,                    # [1, H, W] 渲染深度
    "opacity": opacity,                # [1, H, W] 渲染不透明度
    "n_touched": n_touched,            # [N] 每个 Gaussian 触达像素数
}
```

Gaussian 模型位于：

```text
gaussian_splatting/scene/gaussian_model.py
```

可使用：

```python
gaussians.get_xyz        # [N, 3] Gaussian 中心
gaussians.get_scaling    # [N, 3] Gaussian 尺度
gaussians.get_opacity    # [N, 1] Gaussian opacity
gaussians.get_rotation   # [N, 4] Gaussian 旋转
gaussians.n_obs          # [N] Gaussian 在当前窗口内的观测次数
```

当前帧真实图像来自：

```python
viewpoint.original_image  # [3, H, W]
```

因此，三个指标都可以在 `utils/slam_frontend.py` 中 `tracking()` 返回当前帧 `render_pkg` 后实时计算。

## 2. 建议新增文件

建议在 MonoGS 中新增：

```text
utils/quality_metrics.py
```

主接口：

```python
@torch.no_grad()
def compute_quality_vector(
    viewpoint,
    gaussians,
    render_pkg,
    cfg,
):
    """
    Returns:
        q_norm: torch.Tensor, shape [3]
                [coverage, uncertainty, residual]
        q_debug: dict, 未归一化数值，便于日志记录
    """
```

## 3. q_t^{res}：渲染残差

### 3.1 定义

渲染残差衡量当前 Gaussian map 对当前真实图像的解释程度。若残差较大，说明当前视角下存在未充分建模区域、位姿误差、遮挡变化或外观不一致。

设真实图像为 `I_t`，渲染图像为 `\hat{I}_t`，opacity 为 `A_t`，有效像素 mask 为 `M_t`，则：

```text
q_t^{res} = mean(A_t * M_t * |I_t - \hat{I}_t|)
```

为了让指标对局部失败更敏感，实时实现中同时计算平均残差和 90 分位残差：

```text
q_t^{res} = 0.7 * res_mean + 0.3 * res_p90
```

该指标是单个标量。

### 3.2 代码

```python
import torch


def make_rgb_mask(viewpoint, cfg):
    gt = viewpoint.original_image.cuda()
    rgb_boundary_threshold = cfg["Training"].get("rgb_boundary_threshold", 0.01)
    mask = gt.sum(dim=0, keepdim=True) > rgb_boundary_threshold
    if getattr(viewpoint, "grad_mask", None) is not None:
        mask = mask & viewpoint.grad_mask.bool()
    return mask


@torch.no_grad()
def compute_render_residual(viewpoint, render_pkg, cfg):
    pred = torch.clamp(render_pkg["render"], 0.0, 1.0)
    gt = viewpoint.original_image.cuda()
    opacity = render_pkg["opacity"].detach()
    mask = make_rgb_mask(viewpoint, cfg)

    opacity_th = float(cfg["Quality"].get("opacity_th", 0.2))
    valid = mask & (opacity > opacity_th)
    if valid.sum() < 10:
        return torch.tensor(1.0, device=gt.device)

    residual = torch.abs(pred - gt).mean(dim=0, keepdim=True)
    residual = residual[valid]

    res_mean = residual.mean()
    res_p90 = torch.quantile(residual, 0.90)
    return 0.7 * res_mean + 0.3 * res_p90
```

### 3.3 实时开销

`q_t^{res}` 使用当前 tracking 已经得到的 `render_pkg`，不需要额外渲染，适合每个策略周期更新。

## 4. q_t^{cov}：3D 体素覆盖率

### 4.1 定义

覆盖率衡量当前 Gaussian map 在目标空间内覆盖了多少区域。这里不使用图像 opacity 覆盖率，只使用 3D 体素覆盖率，原因是本文更关心三维扫描进度，而不是当前单个视角是否能渲染出目标。

给定目标包围盒：

```text
bbox_min = [x_min, y_min, z_min]
bbox_max = [x_max, y_max, z_max]
```

将包围盒划分为 `G x G x G` 个体素。若某个体素内存在 opacity 大于阈值的 Gaussian，则认为该体素已覆盖：

```text
q_t^{cov} = occupied_voxels / total_voxels
```

该指标是单个标量，范围约为 `[0, 1]`。

### 4.2 代码

```python
@torch.no_grad()
def compute_voxel_coverage(gaussians, cfg):
    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach().squeeze(-1)

    qcfg = cfg["Quality"]
    bbox_min = torch.tensor(qcfg["bbox_min"], device=xyz.device, dtype=xyz.dtype)
    bbox_max = torch.tensor(qcfg["bbox_max"], device=xyz.device, dtype=xyz.dtype)
    grid_size = int(qcfg.get("coverage_grid_size", 32))
    opacity_th = float(qcfg.get("gaussian_opacity_th", 0.2))

    in_box = ((xyz >= bbox_min) & (xyz <= bbox_max)).all(dim=1)
    valid = in_box & (opacity > opacity_th)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=xyz.device)

    xyz_norm = (xyz[valid] - bbox_min) / (bbox_max - bbox_min).clamp_min(1e-6)
    idx = torch.floor(xyz_norm * grid_size).long()
    idx = torch.clamp(idx, 0, grid_size - 1)

    linear_idx = idx[:, 0] * grid_size * grid_size + idx[:, 1] * grid_size + idx[:, 2]
    occupied = torch.unique(linear_idx).numel()
    total = grid_size ** 3
    return torch.tensor(occupied / total, device=xyz.device, dtype=xyz.dtype)
```

### 4.3 实时开销

该指标需要遍历当前 Gaussian。若 Gaussian 数量较多，建议每 3 到 5 个策略周期更新一次，或仅在 keyframe / mapping 更新后计算，其余时刻复用最近一次结果。

示例：

```python
if frame_idx % 5 == 0:
    q_cov = compute_voxel_coverage(gaussians, cfg)
else:
    q_cov = last_q_cov
```

## 5. q_t^{unc}：重建不确定性代理

### 5.1 定义

MonoGS 不输出概率意义上的不确定性，因此这里的 `q_t^{unc}` 是代理指标。它描述当前可见 Gaussian 是否稳定、是否被充分观测、局部几何是否收敛。

本文只用几何和观测充分性构造该指标，不额外加入局部残差，避免与 `q_t^{res}` 重复。

使用三个子量：

```text
unc_scale：可见 Gaussian 尺度偏大，说明局部几何不够收敛
unc_aniso：可见 Gaussian 各向异性过强，说明约束方向不均衡
unc_obs：可见 Gaussian 观测次数少，说明该区域未被稳定看到
```

组合方式：

```text
q_t^{unc} = w_s * unc_scale + w_a * unc_aniso + w_o * unc_obs
```

默认权重：

```text
w_s = 0.4, w_a = 0.3, w_o = 0.3
```

该指标是单个标量。

### 5.2 代码

```python
@torch.no_grad()
def compute_uncertainty_proxy(gaussians, render_pkg, cfg):
    qcfg = cfg["Quality"]

    visible = render_pkg.get("n_touched", None)
    if visible is not None:
        visible = visible > 0
    else:
        visible = render_pkg["visibility_filter"]

    if visible.sum() < 10:
        return torch.tensor(1.0, device=gaussians.get_xyz.device)

    scales = gaussians.get_scaling.detach()[visible]

    scale_mean = scales.mean(dim=1)
    scale_ref = float(qcfg.get("scale_ref", 0.03))
    unc_scale = torch.clamp(scale_mean / scale_ref, 0.0, 2.0).mean() / 2.0

    s_max = scales.max(dim=1).values
    s_min = scales.min(dim=1).values.clamp_min(1e-6)
    aniso = s_max / s_min
    aniso_ref = float(qcfg.get("aniso_ref", 4.0))
    unc_aniso = torch.clamp((aniso - 1.0) / (aniso_ref - 1.0), 0.0, 1.0).mean()

    if hasattr(gaussians, "n_obs") and gaussians.n_obs.numel() == visible.numel():
        n_obs = gaussians.n_obs.to(scales.device).float()[visible]
        obs_ref = float(qcfg.get("obs_ref", 5.0))
        unc_obs = torch.clamp(1.0 - n_obs / obs_ref, 0.0, 1.0).mean()
    else:
        unc_obs = torch.tensor(0.5, device=scales.device)

    w_scale = float(qcfg.get("unc_w_scale", 0.4))
    w_aniso = float(qcfg.get("unc_w_aniso", 0.3))
    w_obs = float(qcfg.get("unc_w_obs", 0.3))

    return w_scale * unc_scale + w_aniso * unc_aniso + w_obs * unc_obs
```

### 5.3 实时开销

该指标只访问当前可见 Gaussian 的尺度和观测次数，开销较小，建议每个策略周期更新。

## 6. 统一计算 q_t

完整实现示例：

```python
@torch.no_grad()
def normalize_quality(q, cfg):
    qcfg = cfg["Quality"]
    q_min = torch.tensor(qcfg.get("q_min", [0.0, 0.0, 0.0]), device=q.device)
    q_max = torch.tensor(qcfg.get("q_max", [1.0, 1.0, 0.3]), device=q.device)
    return torch.clamp((q - q_min) / (q_max - q_min).clamp_min(1e-6), 0.0, 1.0)


@torch.no_grad()
def compute_quality_vector(viewpoint, gaussians, render_pkg, cfg):
    q_cov = compute_voxel_coverage(gaussians, cfg)
    q_unc = compute_uncertainty_proxy(gaussians, render_pkg, cfg)
    q_res = compute_render_residual(viewpoint, render_pkg, cfg)

    q = torch.stack([q_cov, q_unc, q_res]).float()
    q_norm = normalize_quality(q, cfg)

    q_debug = {
        "coverage": q_cov.item(),
        "uncertainty": q_unc.item(),
        "residual": q_res.item(),
    }
    return q_norm, q_debug
```

这里 `q_norm.shape == [3]`。三个维度依次为：

```text
0: coverage
1: uncertainty
2: residual
```

## 7. 配置文件建议

在 YAML 配置中增加：

```yaml
Quality:
  # Coverage
  bbox_min: [-0.30, -0.30, 0.00]
  bbox_max: [ 0.30,  0.30, 0.50]
  coverage_grid_size: 32
  gaussian_opacity_th: 0.2
  opacity_th: 0.2

  # Uncertainty
  scale_ref: 0.03
  aniso_ref: 4.0
  obs_ref: 5.0
  unc_w_scale: 0.4
  unc_w_aniso: 0.3
  unc_w_obs: 0.3

  # Normalization
  q_min: [0.0, 0.0, 0.0]
  q_max: [1.0, 1.0, 0.3]
```

这些参数需要根据你的工作台尺寸、相机尺度和 MonoGS 重建尺度调整。若单目 MonoGS 存在尺度漂移，`bbox_min / bbox_max / scale_ref` 需要使用同一坐标尺度下的值。

## 8. 接入 MonoGS 的位置

最自然的位置在 `utils/slam_frontend.py` 的 `tracking()` 后。该函数已经得到当前帧的 `render_pkg`：

```python
render_pkg = self.tracking(cur_frame_idx, viewpoint)
```

在这之后计算质量向量：

```python
from utils.quality_metrics import compute_quality_vector

render_pkg = self.tracking(cur_frame_idx, viewpoint)

q_t, q_debug = compute_quality_vector(
    viewpoint=viewpoint,
    gaussians=self.gaussians,
    render_pkg=render_pkg,
    cfg=self.config,
)
```

随后将 `q_t` 送给 VLA 策略：

```python
enhanced_state = torch.cat([robot_state, q_t], dim=-1)
action_chunk = vla_policy(obs_window, enhanced_state, language_instruction)
```

如果 MonoGS 与机器人控制进程分开运行，可以把 `q_t` 和 `q_debug` 放入队列消息中：

```python
self.q_main2policy.put({
    "frame_idx": cur_frame_idx,
    "q_t": q_t.detach().cpu().numpy(),   # shape: [3]
    "q_debug": q_debug,
})
```

## 9. 实时频率建议

建议按开销设置不同更新频率：

```text
q_res：每个策略周期更新
q_unc：每个策略周期更新
q_cov：每 3 到 5 个策略周期更新一次，或每次 keyframe / mapping 更新后计算
```

若策略采用滚动时域控制，每次生成 H 步动作但只执行前 h 步，则只需要在每次重新规划前计算一次 `q_t`。不需要在 Franka 底层控制频率下计算该向量。

## 10. 论文中建议如何表述

可以写成：

```text
本文从 MonoGS 在线维护的 Gaussian map 和当前帧渲染结果中提取三维质量反馈 q_t。
其中，覆盖率由目标包围盒内高 opacity Gaussian 的体素占用比例近似；
不确定性由当前可见 Gaussian 的尺度、各向异性和观测次数构造代理指标；
渲染残差由当前真实图像与 Gaussian map 渲染图像之间的 opacity 加权 L1 误差计算。
三个标量经归一化后拼接为 q_t，并作为增强状态输入 VLA 策略。
```

避免写成：

```text
MonoGS 直接输出覆盖率、不确定性和渲染残差。
```

更严谨的说法是：

```text
这些指标由 MonoGS 的渲染输出、Gaussian 属性和可见性统计实时构造。
```

## 11. 实现注意事项

1. `q_t^{res}` 使用当前帧已有渲染结果，不需要额外渲染。

2. `q_t^{unc}` 不是概率意义上的 uncertainty，而是几何和观测充分性的代理指标。论文中建议写“重建不确定性代理”或“局部不稳定性指标”。

3. `q_t^{cov}` 的在线版本服务于策略决策，论文离线评价时可以用相同定义统一重算，保证不同方法公平。

4. 若 Gaussian 数量变化，`n_obs` 和历史可见性 mask 的长度也会变化。实现时不要长期保存旧 mask 后直接与新 Gaussian 列表相与，除非同步处理 densify/prune 后的索引变化。

5. 第三视角相机可用于验证重建结果，但如果它的图像参与 MonoGS 更新，就不能再把同一帧作为独立 held-out 评价图像。

## 12. 数据维度结论

实时质量反馈 `q_t` 一共 **3 维**：

```text
q_t[0] = q_t^{cov}   # 3D 体素覆盖率
q_t[1] = q_t^{unc}   # 重建不确定性代理
q_t[2] = q_t^{res}   # 渲染残差
```

如果将其输入到状态编码器，可以直接作为长度为 3 的向量送入 MLP：

```python
e_t_q = quality_mlp(q_t)  # q_t.shape == [3]
```
