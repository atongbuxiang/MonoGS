import argparse
import json
from pathlib import Path
import sys
import math

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gaussian_splatting.utils.system_utils import searchForMaxIteration
from munch import munchify
from plyfile import PlyData
import torch

from utils.camera_utils import Camera
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.quality_metrics import (
    compute_path_length,
    compute_voxel_coverage,
    eval_validation_rendering,
    load_ee_positions_from_csv,
    save_offline_metrics,
)


def load_gaussians_from_ply(config, ply_path):
    model_params = munchify(config["model_params"])
    model_params.sh_degree = infer_sh_degree_from_ply(ply_path)
    gaussians = GaussianModel(model_params.sh_degree, config=config)
    gaussians.load_ply(str(ply_path))
    return gaussians


def infer_sh_degree_from_ply(ply_path):
    plydata = PlyData.read(str(ply_path))
    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    n_extra = len(extra_f_names)
    coeffs_per_channel = n_extra // 3 + 1
    sh_degree = int(round(math.sqrt(coeffs_per_channel) - 1))
    if 3 * ((sh_degree + 1) ** 2 - 1) != n_extra:
        raise ValueError(
            f"Cannot infer SH degree from {ply_path}: found {n_extra} f_rest_* properties."
        )
    return sh_degree


def default_final_ply(result_dir):
    result_dir = Path(result_dir)
    final_ply = result_dir / "point_cloud" / "final" / "point_cloud.ply"
    if final_ply.exists():
        return final_ply

    point_cloud_dir = result_dir / "point_cloud"
    iteration = searchForMaxIteration(str(point_cloud_dir))
    return point_cloud_dir / f"iteration_{iteration}" / "point_cloud.ply"


def build_validation_frames(dataset, interval=5):
    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=dataset.fx,
        fy=dataset.fy,
        cx=dataset.cx,
        cy=dataset.cy,
        W=dataset.width,
        H=dataset.height,
    ).transpose(0, 1)
    projection_matrix = projection_matrix.to(device=dataset.device)

    frames = {}
    for idx in range(0, len(dataset), interval):
        frame = Camera.init_from_dataset(dataset, idx, projection_matrix)
        frame.T = frame.T_gt.clone()
        frames[idx] = frame
    return frames


def evaluate_result_dir(result_dir, config_path=None, ee_pose_csv=None, output_path=None):
    result_dir = Path(result_dir)
    config_path = Path(config_path) if config_path else result_dir / "config.yml"
    config = load_config(str(config_path))

    model_params = munchify(config["model_params"])
    pipeline_params = munchify(config["pipeline_params"])
    dataset = load_dataset(model_params, model_params.source_path, config=config)
    gaussians = load_gaussians_from_ply(config, default_final_ply(result_dir))
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    eval_cfg = config.get("EvalMetrics", {})
    interval = int(eval_cfg.get("validation_interval", 5))
    frames = build_validation_frames(dataset, interval=interval)

    rendering = eval_validation_rendering(
        frames=frames,
        gaussians=gaussians,
        dataset=dataset,
        pipe=pipeline_params,
        background=background,
        kf_indices=[],
        interval=interval,
    )
    coverage, occupied, total = compute_voxel_coverage(gaussians, config)
    if ee_pose_csv:
        _, positions = load_ee_positions_from_csv(ee_pose_csv)
        path_length = compute_path_length(positions)
    else:
        path_length = float("nan")

    metrics = {
        "mean_psnr": rendering["mean_psnr"],
        "mean_ssim": rendering["mean_ssim"],
        "num_validation_frames": rendering["num_validation_frames"],
        "coverage": float(coverage.item()),
        "coverage_occupied_voxels": int(occupied),
        "coverage_total_voxels": int(total),
        "path_length_m": path_length,
    }
    output_path = output_path or result_dir / "real_scan_metrics.json"
    save_offline_metrics(metrics, output_path)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--ee-pose-csv")
    parser.add_argument("--output")
    args = parser.parse_args()

    metrics = evaluate_result_dir(
        result_dir=args.result_dir,
        config_path=args.config,
        ee_pose_csv=args.ee_pose_csv,
        output_path=args.output,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
