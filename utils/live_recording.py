import copy
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml

try:
    import torch
except Exception:
    torch = None


class LiveRGBDRecorder:
    def __init__(
        self,
        save_dir,
        base_config,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        frame_stride=1,
        depth_scale=1000.0,
        dataset_root="datasets",
        dataset_name=None,
    ):
        self.save_dir = Path(save_dir)
        self.base_config = copy.deepcopy(base_config)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)
        self.frame_stride = max(1, int(frame_stride))
        self.depth_scale = float(depth_scale)
        dataset_root = Path(dataset_root)
        if not dataset_root.is_absolute():
            dataset_root = Path.cwd() / dataset_root
        if dataset_name is None:
            dataset_name = self.save_dir.name
        self.dataset_dir = dataset_root / dataset_name
        self.rgb_dir = self.dataset_dir / "rgb"
        self.depth_dir = self.dataset_dir / "depth"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        self.frame_ids = []
        self.pose_entries = {}

    def should_record(self, frame_idx):
        return frame_idx % self.frame_stride == 0

    def _to_uint8_rgb(self, image_rgb):
        if torch is not None and isinstance(image_rgb, torch.Tensor):
            image_rgb = image_rgb.detach().cpu()
            if image_rgb.ndim == 3 and image_rgb.shape[0] in (1, 3):
                image_rgb = image_rgb.permute(1, 2, 0).numpy()
            else:
                image_rgb = image_rgb.numpy()

        image_rgb = np.asarray(image_rgb)
        if image_rgb.dtype != np.uint8:
            if np.issubdtype(image_rgb.dtype, np.floating):
                image_rgb = np.clip(np.rint(image_rgb * 255.0), 0, 255).astype(
                    np.uint8
                )
            else:
                image_rgb = image_rgb.astype(np.uint8)
        return image_rgb

    def record_frame(self, frame_idx, image_rgb, depth_m=None):
        if not self.should_record(frame_idx):
            return

        rgb_path = self.rgb_dir / f"{frame_idx:06d}.png"
        image_rgb = self._to_uint8_rgb(image_rgb)
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(rgb_path), bgr)

        if depth_m is not None:
            depth_path = self.depth_dir / f"{frame_idx:06d}.png"
            depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(
                np.rint(depth_mm * self.depth_scale), 0, np.iinfo(np.uint16).max
            ).astype(np.uint16)
            cv2.imwrite(str(depth_path), depth_mm)

        self.frame_ids.append(frame_idx)

    def record_pose(self, frame_idx, T_w2c):
        if not self.should_record(frame_idx):
            return

        T_c2w = np.linalg.inv(T_w2c.detach().cpu().numpy())
        quat_wxyz = trimesh.transformations.quaternion_from_matrix(T_c2w)
        qw, qx, qy, qz = [float(x) for x in quat_wxyz]
        tx, ty, tz = [float(x) for x in T_c2w[:3, 3]]
        self.pose_entries[frame_idx] = (tx, ty, tz, qx, qy, qz, qw)

    def finalize(self):
        frame_ids = [idx for idx in self.frame_ids if idx in self.pose_entries]
        frame_ids = sorted(set(frame_ids))
        if not frame_ids:
            return None

        rgb_txt = self.dataset_dir / "rgb.txt"
        depth_txt = self.dataset_dir / "depth.txt"
        pose_txt = self.dataset_dir / "pose.txt"

        with rgb_txt.open("w", encoding="utf-8") as f_rgb:
            f_rgb.write("# timestamp filename\n")
            for idx in frame_ids:
                ts = idx / 30.0
                f_rgb.write(f"{ts:.6f} rgb/{idx:06d}.png\n")

        with depth_txt.open("w", encoding="utf-8") as f_depth:
            f_depth.write("# timestamp filename\n")
            for idx in frame_ids:
                ts = idx / 30.0
                f_depth.write(f"{ts:.6f} depth/{idx:06d}.png\n")

        with pose_txt.open("w", encoding="utf-8") as f_pose:
            f_pose.write("# timestamp tx ty tz qx qy qz qw\n")
            for idx in frame_ids:
                ts = idx / 30.0
                tx, ty, tz, qx, qy, qz, qw = self.pose_entries[idx]
                f_pose.write(
                    f"{ts:.6f} {tx:.8f} {ty:.8f} {tz:.8f} "
                    f"{qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n"
                )

        offline_cfg = copy.deepcopy(self.base_config)
        offline_cfg.setdefault("Results", {})
        offline_cfg.setdefault("Dataset", {})
        offline_cfg.setdefault("EvalMetrics", {})

        offline_cfg["Dataset"]["type"] = "tum"
        offline_cfg["Dataset"]["sensor_type"] = "depth"
        offline_cfg["Dataset"]["dataset_path"] = str(self.dataset_dir.resolve())
        offline_cfg["Dataset"]["Calibration"] = {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0,
            "distorted": False,
            "width": self.width,
            "height": self.height,
            "depth_scale": self.depth_scale,
        }
        offline_cfg["EvalMetrics"]["validation_interval"] = 1
        offline_cfg["Results"]["use_gui"] = False
        offline_cfg["Results"]["eval_rendering"] = False
        offline_cfg["Results"]["eval_real_scan_metrics"] = False

        offline_cfg_path = self.save_dir / "offline_eval_config.yml"
        with offline_cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(offline_cfg, f, sort_keys=False)

        summary_path = self.save_dir / "replay_dataset_info.yml"
        with summary_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "dataset_dir": str(self.dataset_dir.resolve()),
                    "num_recorded_frames": len(frame_ids),
                    "frame_stride": self.frame_stride,
                    "offline_eval_config": str(offline_cfg_path.resolve()),
                },
                f,
                sort_keys=False,
            )

        return offline_cfg_path
