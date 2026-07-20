#!/usr/bin/env python3
"""Measure current grippertip offset in the target ArUco marker frame."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.duration import Duration
    from rclpy.time import Time
    import tf2_ros
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This tool needs ROS 2 Python packages. Source ROS first, for example: "
        "source /opt/ros/jazzy/setup.bash"
    ) from exc

TOOL_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = TOOL_DIR.parents[1]
VISION_SRC = WORKSPACE_ROOT / "src" / "vision_controller"
if VISION_SRC.exists():
    sys.path.insert(0, str(VISION_SRC))
NINIBOT_SRC = WORKSPACE_ROOT / "src" / "ninibot"
if NINIBOT_SRC.exists():
    sys.path.insert(0, str(NINIBOT_SRC))

from vision_controller.aruco_detection import ARUCO_DICTIONARIES
from vision_controller.aruco_detection import draw_text
from vision_controller.aruco_detection import estimate_marker_poses
from vision_controller.aruco_detection import load_camera_info
from vision_controller.aruco_detection import make_detector
from vision_controller.aruco_detection import open_camera
from vision_controller.aruco_detection import resolve_path
from vision_controller.aruco_detection import show_window
from ninibot.task_params import INNOMAKER_CAMERA_DEVICE
from ninibot.task_params import PRESS_BUTTON_PARAMS


DATA_DIR = WORKSPACE_ROOT / "data" / "calibration"
TASK_PARAMS_PATH = NINIBOT_SRC / "ninibot" / "task_params.py"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the target ArUco marker and print the current grippertip "
            "position expressed in that marker's coordinate frame."
        )
    )
    parser.add_argument(
        "--calibration",
        default=str(PRESS_BUTTON_PARAMS.get("calibration_path", DATA_DIR / "camera_calibration.yaml")),
    )
    parser.add_argument(
        "--camera",
        default=str(PRESS_BUTTON_PARAMS.get("camera_device", INNOMAKER_CAMERA_DEVICE)),
        help="OpenCV camera device path or index.",
    )
    parser.add_argument("--width", type=int, default=int(PRESS_BUTTON_PARAMS.get("frame_width", 640)))
    parser.add_argument("--height", type=int, default=int(PRESS_BUTTON_PARAMS.get("frame_height", 480)))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="")
    parser.add_argument(
        "--marker-length",
        type=float,
        default=float(PRESS_BUTTON_PARAMS.get("marker_length", 0.05)),
    )
    parser.add_argument(
        "--dictionary",
        choices=sorted(ARUCO_DICTIONARIES),
        default=str(PRESS_BUTTON_PARAMS.get("aruco_dictionary", "4X4_50")),
    )
    parser.add_argument(
        "--target-marker-id",
        type=int,
        default=int(PRESS_BUTTON_PARAMS.get("target_marker_id", 0)),
    )
    parser.add_argument(
        "--camera-frame",
        default=str(PRESS_BUTTON_PARAMS.get("camera_frame", "camera_optical_frame")),
    )
    parser.add_argument(
        "--gripper-frame",
        default=str(PRESS_BUTTON_PARAMS.get("gripper_frame", "grippertip")),
    )
    parser.add_argument(
        "--tf-timeout",
        type=float,
        default=float(PRESS_BUTTON_PARAMS.get("tf_timeout_sec", 0.5)),
    )
    parser.add_argument("--print-every", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=0, help="Stop after this many printed samples; 0 runs until Ctrl-C.")
    parser.add_argument(
        "--write-task-params",
        action="store_true",
        help=(
            "Average the collected samples and update PRESS_BUTTON_PARAMS "
            "target_x/y/z_offset in task_params.py."
        ),
    )
    parser.add_argument(
        "--task-params-file",
        type=Path,
        default=TASK_PARAMS_PATH,
        help="Python file containing PRESS_BUTTON_PARAMS.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Write without asking for interactive confirmation.",
    )
    parser.add_argument("--no-display", action="store_true")
    return parser


def format_cm(vector, digits: int = 1) -> str:
    value = np.asarray(vector, dtype=float) * 100.0
    return f"({value[0]:+.{digits}f}, {value[1]:+.{digits}f}, {value[2]:+.{digits}f})"


def lookup_grippertip_camera_xyz(node, tf_buffer, camera_frame, gripper_frame, timeout_sec):
    stamped = tf_buffer.lookup_transform(
        camera_frame,
        gripper_frame,
        Time(),
        timeout=Duration(seconds=timeout_sec),
    )
    translation = stamped.transform.translation
    return np.array([translation.x, translation.y, translation.z], dtype=float)


def tf_hint(exc) -> str:
    text = str(exc)
    if "does not exist" not in text:
        return ""
    return (
        " Hint: start robot_state_publisher/read_arm first, for example "
        "`ros2 launch arm_controller gripper_pose_debug.launch.py start_rviz:=False`."
    )


def marker_frame_offset(marker_camera_xyz, marker_rvec, grippertip_camera_xyz):
    # OpenCV's marker pose maps marker-frame points into camera frame:
    # p_camera = R_camera_marker * p_marker + t_camera_marker.
    r_camera_marker, _ = cv2.Rodrigues(marker_rvec.reshape(3, 1))
    return r_camera_marker.T @ (grippertip_camera_xyz - marker_camera_xyz)


def draw_overlay(frame, marker_id, marker_camera_xyz, grippertip_camera_xyz, offset_marker_xyz):
    lines = [
        f"id={marker_id}",
        f"marker_camera_cm={format_cm(marker_camera_xyz)}",
        f"grippertip_camera_cm={format_cm(grippertip_camera_xyz)}",
        f"grippertip_marker_cm={format_cm(offset_marker_xyz)}",
        "use xyz/100 as marker-frame target offset meters",
    ]
    for index, line in enumerate(lines):
        draw_text(frame, line, 30 + index * 28, (0, 255, 255))


def task_param_value(text, key):
    pattern = re.compile(
        rf'(?m)^\s*"{re.escape(key)}"\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,'
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Could not find {key!r} in task params file")
    return float(match.group(1))


def update_task_param_value(text, key, value):
    pattern = re.compile(
        rf'(?m)^(\s*"{re.escape(key)}"\s*:\s*)[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(\s*,.*)$'
    )
    new_text, count = pattern.subn(rf"\g<1>{value:.5f}\2", text, count=1)
    if count != 1:
        raise ValueError(f"Could not update {key!r} in task params file")
    return new_text


def maybe_write_task_params(path, offsets_xyz, std_xyz, sample_count, assume_yes=False):
    path = path.expanduser()
    text = path.read_text(encoding="utf-8")
    old_offsets = np.array(
        [
            task_param_value(text, "target_x_offset"),
            task_param_value(text, "target_y_offset"),
            task_param_value(text, "target_z_offset"),
        ],
        dtype=float,
    )

    print("\nCollected marker-frame offset samples:")
    print(f"  samples={sample_count}")
    print(f"  mean_m=({offsets_xyz[0]:+.5f}, {offsets_xyz[1]:+.5f}, {offsets_xyz[2]:+.5f})")
    print(f"  std_cm={format_cm(std_xyz, digits=2)}")
    print(f"  old_m=({old_offsets[0]:+.5f}, {old_offsets[1]:+.5f}, {old_offsets[2]:+.5f})")
    print(f"  file={path}")

    if not assume_yes:
        answer = input("Write mean to PRESS_BUTTON_PARAMS target offsets? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Did not write task params.")
            return False

    new_text = text
    for key, value in zip(
        ("target_x_offset", "target_y_offset", "target_z_offset"),
        offsets_xyz,
    ):
        new_text = update_task_param_value(new_text, key, float(value))

    path.write_text(new_text, encoding="utf-8")
    print("Updated task params target offsets.")
    return True


def main() -> int:
    args = build_arg_parser().parse_args()
    sample_goal = args.samples
    if args.write_task_params and sample_goal <= 0:
        sample_goal = 10

    calibration_path = resolve_path(args.calibration, base_dir=WORKSPACE_ROOT)
    calibration_width, calibration_height, camera_matrix, dist_coeffs = load_camera_info(calibration_path)
    width = args.width or calibration_width
    height = args.height or calibration_height

    rclpy.init()
    node = rclpy.create_node("measure_grippertip_marker_offset")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)

    detector = make_detector(args.dictionary)
    cap = open_camera(args.camera, width, height, args.fps, args.fourcc)
    if not cap.isOpened():
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(f"Could not open camera {args.camera!r}")

    print(
        "Measuring grippertip offset in marker frame. "
        f"camera={args.camera!r}, marker_id={args.target_marker_id}, "
        f"TF={args.camera_frame}<-{args.gripper_frame}"
    )
    print("Output units: cm. Divide by 100 for task_params target_*_offset meters.")
    if args.write_task_params:
        print(f"Will collect {sample_goal} samples, average them, then update task params.")

    last_print_time = 0.0
    printed_samples = 0
    offset_samples = []

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            ok, frame = cap.read()
            if not ok:
                print("Waiting for camera frame...")
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector(gray)
            if ids is None:
                if not args.no_display:
                    draw_text(frame, "waiting for ArUco", 30, (0, 0, 255))
                    show_window("grippertip marker offset", frame, 1)
                continue

            rvecs, tvecs = estimate_marker_poses(
                corners,
                args.marker_length,
                camera_matrix,
                dist_coeffs,
            )
            target_index = None
            for index, marker_id in enumerate(ids.flatten()):
                if args.target_marker_id < 0 or int(marker_id) == args.target_marker_id:
                    target_index = index
                    break

            if target_index is None or target_index >= len(tvecs) or target_index >= len(rvecs):
                if not args.no_display:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    draw_text(frame, f"target {args.target_marker_id} not visible", 30, (0, 0, 255))
                    show_window("grippertip marker offset", frame, 1)
                continue

            marker_camera_xyz = tvecs[target_index].astype(float)
            marker_rvec = rvecs[target_index].astype(float)
            try:
                grippertip_camera_xyz = lookup_grippertip_camera_xyz(
                    node,
                    tf_buffer,
                    args.camera_frame,
                    args.gripper_frame,
                    args.tf_timeout,
                )
            except Exception as exc:
                if time.monotonic() - last_print_time >= max(0.05, args.print_every):
                    print(
                        f"Waiting for TF {args.camera_frame}<-{args.gripper_frame}: "
                        f"{exc}{tf_hint(exc)}"
                    )
                    last_print_time = time.monotonic()
                continue

            offset_marker_xyz = marker_frame_offset(
                marker_camera_xyz,
                marker_rvec,
                grippertip_camera_xyz,
            )

            if not args.no_display:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                if hasattr(cv2, "drawFrameAxes"):
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        marker_rvec.reshape(3, 1),
                        marker_camera_xyz.reshape(3, 1),
                        args.marker_length * 0.5,
                    )
                draw_overlay(
                    frame,
                    int(ids.flatten()[target_index]),
                    marker_camera_xyz,
                    grippertip_camera_xyz,
                    offset_marker_xyz,
                )
                show_window("grippertip marker offset", frame, 1)

            now = time.monotonic()
            if now - last_print_time >= max(0.05, args.print_every):
                printed_samples += 1
                offset_samples.append(offset_marker_xyz.copy())
                print(
                    f"grippertip_marker_cm={format_cm(offset_marker_xyz)} "
                    f"marker_camera_cm={format_cm(marker_camera_xyz)} "
                    f"grippertip_camera_cm={format_cm(grippertip_camera_xyz)} "
                    f"suggested_offsets_m=({offset_marker_xyz[0]:+.4f}, "
                    f"{offset_marker_xyz[1]:+.4f}, {offset_marker_xyz[2]:+.4f})"
                )
                last_print_time = now
                if sample_goal > 0 and printed_samples >= sample_goal:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_display:
            try:
                cv2.destroyWindow("grippertip marker offset")
            except cv2.error:
                pass
        node.destroy_node()
        rclpy.shutdown()

    if args.write_task_params:
        if not offset_samples:
            print("No usable samples collected; did not write task params.")
            return 1
        samples = np.vstack(offset_samples)
        wrote = maybe_write_task_params(
            args.task_params_file,
            samples.mean(axis=0),
            samples.std(axis=0),
            len(offset_samples),
            assume_yes=args.yes,
        )
        return 0 if wrote else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
