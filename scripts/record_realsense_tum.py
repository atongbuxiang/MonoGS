import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    import pyrealsense2 as rs
except Exception as exc:
    raise SystemExit(
        "pyrealsense2 is required for this script. Install it in the active environment."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record a RealSense RGB-D dataset in a TUM-like layout."
    )
    parser.add_argument(
        "--output",
        default="datasets/realsense_d455_tum",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Target RGB-D recording frequency in Hz.",
    )
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=360)
    parser.add_argument("--color-fps", type=int, default=30)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=360)
    parser.add_argument("--depth-fps", type=int, default=30)
    parser.add_argument(
        "--imu",
        action="store_true",
        help="Also record accelerometer and gyroscope streams if available.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after saving this many RGB-D frames. 0 means unlimited.",
    )
    parser.add_argument(
        "--write-identity-pose",
        action="store_true",
        help=(
            "Write pose.txt with identity poses for every frame. This is only a "
            "placeholder and is not real camera trajectory."
        ),
    )
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def write_index_file(path, title, entries):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write("# timestamp filename\n")
        for timestamp, rel_path in entries:
            f.write(f"{timestamp:.6f} {rel_path}\n")


def write_motion_file(path, title, entries, labels):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write(f"# timestamp {' '.join(labels)}\n")
        for timestamp, values in entries:
            value_str = " ".join(f"{float(v):.9f}" for v in values)
            f.write(f"{timestamp:.6f} {value_str}\n")


def write_pose_file(path, entries):
    with path.open("w", encoding="utf-8") as f:
        f.write("# placeholder trajectory\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for timestamp in entries:
            f.write(f"{timestamp:.6f} 0 0 0 0 0 0 1\n")


def main():
    args = parse_args()
    output_dir = Path(args.output)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    ensure_dir(rgb_dir)
    ensure_dir(depth_dir)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        args.color_width,
        args.color_height,
        rs.format.bgr8,
        args.color_fps,
    )
    config.enable_stream(
        rs.stream.depth,
        args.depth_width,
        args.depth_height,
        rs.format.z16,
        args.depth_fps,
    )
    if args.imu:
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_unit_m = float(depth_sensor.get_depth_scale())
    depth_scale_for_loader = 1.0 / depth_unit_m

    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intr = color_profile.get_intrinsics()

    should_stop = False

    def handle_stop(signum, frame):
        del signum, frame
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    rgb_entries = []
    depth_entries = []
    accel_entries = []
    gyro_entries = []
    pose_timestamps = []

    last_saved_ts = None
    base_rs_ts = None
    base_wall_ts = None
    saved_frames = 0

    print(f"Recording to {output_dir}")
    print("Press Ctrl+C to stop.")
    if args.write_identity_pose:
        print("Warning: pose.txt will contain identity placeholders, not real poses.")

    try:
        while not should_stop:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_rs_ts = color_frame.get_timestamp() * 1e-3
            if base_rs_ts is None:
                base_rs_ts = color_rs_ts
                base_wall_ts = time.time() - base_rs_ts

            timestamp = base_wall_ts + color_rs_ts

            if args.imu:
                for frame in frames:
                    if not frame.is_motion_frame():
                        continue
                    motion = frame.as_motion_frame().get_motion_data()
                    motion_ts = base_wall_ts + frame.get_timestamp() * 1e-3
                    stream_type = frame.profile.stream_type()
                    values = (motion.x, motion.y, motion.z)
                    if stream_type == rs.stream.accel:
                        accel_entries.append((motion_ts, values))
                    elif stream_type == rs.stream.gyro:
                        gyro_entries.append((motion_ts, values))

            if last_saved_ts is not None and (timestamp - last_saved_ts) < (1.0 / args.hz):
                continue

            last_saved_ts = timestamp
            filename = f"{timestamp:.6f}.png"

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            cv2.imwrite(str(rgb_dir / filename), color_image)
            cv2.imwrite(str(depth_dir / filename), depth_image.astype(np.uint16))

            rgb_entries.append((timestamp, f"rgb/{filename}"))
            depth_entries.append((timestamp, f"depth/{filename}"))
            pose_timestamps.append(timestamp)
            saved_frames += 1

            if saved_frames % 10 == 0:
                print(f"Saved {saved_frames} RGB-D frames")

            if args.max_frames > 0 and saved_frames >= args.max_frames:
                break
    finally:
        pipeline.stop()

    write_index_file(output_dir / "rgb.txt", "color images", rgb_entries)
    write_index_file(output_dir / "depth.txt", "depth maps", depth_entries)

    if args.imu:
        write_motion_file(
            output_dir / "accelerometer.txt",
            "accelerometer data",
            accel_entries,
            ["ax", "ay", "az"],
        )
        write_motion_file(
            output_dir / "gyro.txt",
            "gyroscope data",
            gyro_entries,
            ["gx", "gy", "gz"],
        )

    if args.write_identity_pose:
        write_pose_file(output_dir / "pose.txt", pose_timestamps)

    calibration = {
        "camera": "Intel RealSense D455",
        "color": {
            "width": color_intr.width,
            "height": color_intr.height,
            "fx": float(color_intr.fx),
            "fy": float(color_intr.fy),
            "cx": float(color_intr.ppx),
            "cy": float(color_intr.ppy),
            "distortion_model": str(color_intr.model),
            "coeffs": [float(v) for v in color_intr.coeffs],
        },
        "depth_scale_m_per_unit": depth_unit_m,
        "depth_scale_for_monogs_loader": depth_scale_for_loader,
        "record_hz": args.hz,
        "saved_frames": saved_frames,
        "has_imu": bool(args.imu),
        "has_real_pose": False,
        "notes": (
            "D455 provides RGB, depth, and IMU streams, but not a native 6DoF pose "
            "stream like T265. Use an external tracker or SLAM output if you need "
            "real pose.txt / groundtruth.txt."
        ),
    }
    with (output_dir / "calibration.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(calibration, f, sort_keys=False)

    if saved_frames == 0:
        print("No frames were saved.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Done. Saved {saved_frames} RGB-D frames to {output_dir}")
    if not args.write_identity_pose:
        print(
            "No pose.txt was written because D455 does not provide native 6DoF pose. "
            "Add pose from an external source if you want a full TUM-style trajectory."
        )


if __name__ == "__main__":
    main()
