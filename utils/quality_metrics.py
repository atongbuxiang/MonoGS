import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


def _quality_cfg(cfg):
    return cfg.get("Quality", {})


def make_rgb_mask(viewpoint, cfg):
    gt = viewpoint.original_image.cuda()
    threshold = float(cfg["Training"].get("rgb_boundary_threshold", 0.01))
    mask = gt.sum(dim=0, keepdim=True) > threshold
    if getattr(viewpoint, "grad_mask", None) is not None:
        mask = mask & viewpoint.grad_mask.bool()
    return mask


@torch.no_grad()
def compute_render_residual(viewpoint, render_pkg, cfg):
    pred = torch.clamp(render_pkg["render"], 0.0, 1.0)
    gt = viewpoint.original_image.cuda()
    opacity = render_pkg["opacity"].detach()
    mask = make_rgb_mask(viewpoint, cfg)

    qcfg = _quality_cfg(cfg)
    opacity_th = float(qcfg.get("opacity_th", 0.2))
    valid = mask & (opacity > opacity_th)
    if valid.sum() < 10:
        return torch.tensor(1.0, device=gt.device)

    residual = torch.abs(pred - gt).mean(dim=0, keepdim=True)
    residual = residual[valid]
    return 0.7 * residual.mean() + 0.3 * torch.quantile(residual, 0.90)


@torch.no_grad()
def compute_voxel_coverage(gaussians, cfg):
    qcfg = _quality_cfg(cfg)
    grid_size = int(qcfg.get("coverage_grid_size", 32))
    total_count = int(grid_size**3)

    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach().squeeze(-1)
    device = xyz.device
    dtype = xyz.dtype

    bbox_min = torch.tensor(qcfg.get("bbox_min", [-1.0, -1.0, -1.0]), device=device, dtype=dtype)
    bbox_max = torch.tensor(qcfg.get("bbox_max", [1.0, 1.0, 1.0]), device=device, dtype=dtype)
    opacity_th = float(qcfg.get("gaussian_opacity_th", 0.2))

    in_box = ((xyz >= bbox_min) & (xyz <= bbox_max)).all(dim=1)
    valid = in_box & (opacity > opacity_th)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype), 0, total_count

    xyz_norm = (xyz[valid] - bbox_min) / (bbox_max - bbox_min).clamp_min(1e-6)
    voxel_idx = torch.floor(xyz_norm * grid_size).long()
    voxel_idx = torch.clamp(voxel_idx, 0, grid_size - 1)

    linear_idx = (
        voxel_idx[:, 0] * grid_size * grid_size
        + voxel_idx[:, 1] * grid_size
        + voxel_idx[:, 2]
    )
    occupied_count = int(torch.unique(linear_idx).numel())
    coverage = torch.tensor(occupied_count / total_count, device=device, dtype=dtype)
    return coverage, occupied_count, total_count


@torch.no_grad()
def compute_uncertainty_proxy(gaussians, render_pkg, cfg):
    qcfg = _quality_cfg(cfg)
    visible = render_pkg.get("n_touched", None)
    if visible is not None:
        visible = visible > 0
    else:
        visible = render_pkg["visibility_filter"]

    device = gaussians.get_xyz.device
    if visible.sum() < 10:
        return torch.tensor(1.0, device=device)

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


@torch.no_grad()
def normalize_quality(q, cfg):
    qcfg = _quality_cfg(cfg)
    q_min = torch.tensor(qcfg.get("q_min", [0.0, 0.0, 0.0]), device=q.device)
    q_max = torch.tensor(qcfg.get("q_max", [1.0, 1.0, 0.3]), device=q.device)
    return torch.clamp((q - q_min) / (q_max - q_min).clamp_min(1e-6), 0.0, 1.0)


@torch.no_grad()
def compute_quality_vector(viewpoint, gaussians, render_pkg, cfg):
    q_cov, occupied_count, total_count = compute_voxel_coverage(gaussians, cfg)
    q_unc = compute_uncertainty_proxy(gaussians, render_pkg, cfg)
    q_res = compute_render_residual(viewpoint, render_pkg, cfg)
    q = torch.stack([q_cov, q_unc, q_res]).float()
    q_norm = normalize_quality(q, cfg)
    q_debug = {
        "coverage": float(q_cov.item()),
        "uncertainty": float(q_unc.item()),
        "residual": float(q_res.item()),
        "coverage_occupied_voxels": occupied_count,
        "coverage_total_voxels": total_count,
    }
    return q_norm, q_debug


def compute_path_length(positions):
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(positions[1:] - positions[:-1], axis=1).sum())


def load_ee_positions_from_csv(csv_path):
    timestamps = []
    positions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            positions.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return np.asarray(timestamps), np.asarray(positions, dtype=np.float64)


def estimate_path_length_from_cameras(cameras):
    centers = []
    for idx in sorted(cameras):
        camera = cameras[idx]
        centers.append(camera.camera_center.detach().cpu().numpy())
    return compute_path_length(np.asarray(centers, dtype=np.float64))


@torch.no_grad()
def eval_validation_rendering(frames, gaussians, dataset, pipe, background, kf_indices=None, interval=5):
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.utils.image_utils import psnr
    from gaussian_splatting.utils.loss_utils import ssim

    kf_indices = set(kf_indices or [])
    psnr_values = []
    ssim_values = []
    for idx in range(0, len(frames), interval):
        if idx in kf_indices or idx not in frames:
            continue
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]
        pred = torch.clamp(render(frame, gaussians, pipe, background)["render"], 0.0, 1.0)
        mask = gt_image > 0
        if mask.sum() == 0:
            continue
        psnr_values.append(psnr(pred[mask].unsqueeze(0), gt_image[mask].unsqueeze(0)).item())
        ssim_values.append(ssim(pred.unsqueeze(0), gt_image.unsqueeze(0)).item())

    return {
        "mean_psnr": float(np.mean(psnr_values)) if psnr_values else math.nan,
        "mean_ssim": float(np.mean(ssim_values)) if ssim_values else math.nan,
        "num_validation_frames": len(psnr_values),
    }


def save_offline_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
