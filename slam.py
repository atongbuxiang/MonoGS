import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from math import nan
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.live_recording import LiveRGBDRecorder
from utils.multiprocessing_utils import FakeQueue
from utils.quality_metrics import (
    compute_voxel_coverage,
    estimate_path_length_from_cameras,
    save_offline_metrics,
)
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        self.eval_rendering = self.config["Results"]["eval_rendering"]
        self.eval_real_scan_metrics = bool(
            self.config["Results"].get(
                "eval_real_scan_metrics", self.config["Results"]["save_results"]
            )
        )

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        if (
            self.live_mode
            and self.config["Dataset"]["sensor_type"] == "depth"
            and save_dir is not None
        ):
            record_cfg = self.config.get("Recording", {})
            enabled = bool(record_cfg.get("enabled", True))
            if enabled and hasattr(self.dataset, "depth_scale"):
                self.dataset.recorder = LiveRGBDRecorder(
                    save_dir=save_dir,
                    base_config=self.config,
                    fx=self.dataset.fx,
                    fy=self.dataset.fy,
                    cx=self.dataset.cx,
                    cy=self.dataset.cy,
                    width=self.dataset.width,
                    height=self.dataset.height,
                    frame_stride=record_cfg.get("frame_stride", 1),
                    depth_scale=record_cfg.get("depth_scale", 1000.0),
                    dataset_root=record_cfg.get("dataset_root", "datasets"),
                    dataset_name=record_cfg.get("dataset_name"),
                )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        start.record()
        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        self.gaussians = self.frontend.gaussians
        kf_indices = self.frontend.kf_indices
        if self.save_dir is not None and self.config["Results"]["save_results"]:
            final_ply = Path(self.save_dir) / "point_cloud" / "final" / "point_cloud.ply"
            if not final_ply.exists() and self.gaussians is not None:
                save_gaussians(self.gaussians, self.save_dir, "final", final=True)

        final_rendering_result = None
        if self.eval_rendering and not self.live_mode:
            ATE = self.try_eval_ate()

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
                compute_lpips=False,
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
                compute_lpips=False,
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)
            final_rendering_result = rendering_result

        if self.eval_real_scan_metrics and self.save_dir is not None:
            if final_rendering_result is None:
                if self.live_mode:
                    final_rendering_result = {}
                else:
                    final_rendering_result = eval_rendering(
                        self.frontend.cameras,
                        self.gaussians,
                        self.dataset,
                        self.save_dir,
                        self.pipeline_params,
                        self.background,
                        kf_indices=kf_indices,
                        iteration="real_scan",
                        compute_lpips=False,
                    )
            self.save_real_scan_metrics(
                rendering_result=final_rendering_result,
                path_length_m=estimate_path_length_from_cameras(
                    self.frontend.cameras
                ),
                fps=FPS,
            )
        if self.live_mode and hasattr(self.dataset, "recorder") and self.dataset.recorder:
            offline_cfg_path = self.dataset.recorder.finalize()
            if offline_cfg_path is not None:
                Log(
                    f"Recorded replay dataset and offline eval config at {offline_cfg_path}",
                    tag="Eval",
                )

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass

    def try_eval_ate(self):
        if self.save_dir is None:
            return nan
        if len(self.frontend.kf_indices) < 3:
            Log("Skipping ATE: fewer than 3 keyframes.", tag="Eval")
            return nan
        try:
            return eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )
        except Exception as exc:
            Log(f"Skipping ATE: {exc}", tag="Eval")
            return nan

    def save_real_scan_metrics(self, rendering_result, path_length_m, fps=None):
        if self.save_dir is None:
            return
        coverage, occupied, total = compute_voxel_coverage(self.gaussians, self.config)
        quality_summary = {}
        if self.frontend.quality_logger is not None:
            quality_summary = self.frontend.quality_logger.summary()
        psnr_value = rendering_result.get("mean_psnr")
        ssim_value = rendering_result.get("mean_ssim")
        metrics = {
            "mean_psnr": psnr_value if psnr_value is not None else None,
            "mean_ssim": ssim_value if ssim_value is not None else None,
            "coverage": float(coverage.item()),
            "coverage_occupied_voxels": int(occupied),
            "coverage_total_voxels": int(total),
            "path_length_m": float(path_length_m),
            "num_frames": len(self.frontend.cameras),
            "num_keyframes": len(self.frontend.kf_indices),
        }
        if self.live_mode and (psnr_value is None or ssim_value is None):
            metrics["render_metrics_computed"] = False
            metrics["render_metrics_note"] = (
                "PSNR/SSIM are not computed in live Realsense mode during SLAM runtime."
            )
        else:
            metrics["render_metrics_computed"] = True
        if fps is not None and np.isfinite(fps):
            metrics["fps"] = float(fps)
        metrics.update(quality_summary)
        output_path = os.path.join(self.save_dir, "real_scan_metrics.json")
        save_offline_metrics(metrics, output_path)
        Log(f"Saved real scan metrics to {output_path}", tag="Eval")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].replace("\\", "/").split("/")
        dataset_tag = "_".join(path[-3:-1]) if len(path) >= 3 else path[-1]
        save_dir = os.path.join(
            config["Results"]["save_dir"], dataset_tag, current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
