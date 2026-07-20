#!/usr/bin/env python3
"""Measure the move-cup target frame offset in the target ArUco marker frame."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import select
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
from ninibot.task_params import MOVE_CUP_PARAMS


DATA_DIR = WORKSPACE_ROOT / "data" / "calibration"
TASK_PARAMS_PATH = NINIBOT_SRC / "ninibot" / "task_params.py"
TASK_PARAMS_DICT_NAME = "MOVE_CUP_PARAMS"
WINDOW_NAME = "move cup marker offset"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the move-cup target ArUco marker and print the current "
            "target frame position expressed in that marker's coordinate frame. "
            "Use this for wrist-frame move-cup calibration."
        )
    )
    parser.add_argument(
        "--calibration",
        default=str(MOVE_CUP_PARAMS.get("calibration_path", DATA_DIR / "camera_calibration.yaml")),
    )
    parser.add_argument(
        "--camera",
        default=str(MOVE_CUP_PARAMS.get("camera_device", INNOMAKER_CAMERA_DEVICE)),
        help="OpenCV camera device path or index.",
    )
    parser.add_argument("--width", type=int, default=int(MOVE_CUP_PARAMS.get("frame_width", 640)))
    parser.add_argument("--height", type=int, default=int(MOVE_CUP_PARAMS.get("frame_height", 480)))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="")
    parser.add_argument(
        "--marker-length",
        type=float,
        default=float(MOVE_CUP_PARAMS.get("marker_length", 0.05)),
    )
    parser.add_argument(
        "--dictionary",
        choices=sorted(ARUCO_DICTIONARIES),
        default=str(MOVE_CUP_PARAMS.get("aruco_dictionary", "4X4_50")),
    )
    parser.add_argument(
        "--target-marker-id",
        type=int,
        default=int(MOVE_CUP_PARAMS.get("target_marker_id", 1)),
    )
    parser.add_argument(
        "--camera-frame",
        default=str(MOVE_CUP_PARAMS.get("camera_frame", "camera_optical_frame")),
    )
    parser.add_argument(
        "--target-frame",
        default="wrist",
        help="TF frame to calibrate as the move-cup target point.",
    )
    parser.add_argument(
        "--tf-timeout",
        type=float,
        default=float(MOVE_CUP_PARAMS.get("tf_timeout_sec", 0.5)),
    )
    parser.add_argument("--print-every", type=float, default=0.5)
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help=(
            "Stop after this many captured samples. With --capture-on-enter, "
            "this is the number of Enter presses; without it, this is the "
            "number of printed samples. 0 runs until Ctrl-C."
        ),
    )
    parser.add_argument(
        "--capture-on-enter",
        action="store_true",
        help="Only record a sample when Enter is pressed.",
    )
    parser.add_argument(
        "--write-task-params",
        action="store_true",
        help=(
            "Average the collected samples and update MOVE_CUP_PARAMS "
            "target_x/y/z_offset in task_params.py."
        ),
    )
    parser.add_argument(
        "--task-params-file",
        type=Path,
        default=TASK_PARAMS_PATH,
        help="Python file containing MOVE_CUP_PARAMS.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Write without asking for interactive confirmation.",
    )
    parser.add_argument("--no-display", action="store_true")
    return parser


def format_cm(vector, digits: int = 1) -> str:
    value = np.asarray(vector, dtype=float).reshape(3) * 100.0
    return f"({value[0]:+.{digits}f}, {value[1]:+.{digits}f}, {value[2]:+.{digits}f})"


def format_m(vector, digits: int = 4) -> str:
    value = np.asarray(vector, dtype=float).reshape(3)
    return f"({value[0]:+.{digits}f}, {value[1]:+.{digits}f}, {value[2]:+.{digits}f})"


def lookup_frame_camera_xyz(tf_buffer, camera_frame, target_frame, timeout_sec):
    stamped = tf_buffer.lookup_transform(
        camera_frame,
        target_frame,
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
        "`ros2 launch arm_controller gripper_pose_debug.launch.py`."
    )


def marker_frame_offset(marker_camera_xyz, marker_rvec, target_camera_xyz):
    # OpenCV's marker pose maps marker-frame points into camera frame:
    # p_camera = R_camera_marker * p_marker + t_camera_marker.
    marker_camera_xyz = np.asarray(marker_camera_xyz, dtype=float).reshape(3)
    marker_rvec = np.asarray(marker_rvec, dtype=float).reshape(3, 1)
    target_camera_xyz = np.asarray(target_camera_xyz, dtype=float).reshape(3)
    r_camera_marker, _ = cv2.Rodrigues(marker_rvec)
    return r_camera_marker.T @ (target_camera_xyz - marker_camera_xyz)


def draw_overlay(
    frame,
    marker_id,
    target_frame,
    marker_camera_xyz,
    target_camera_xyz,
    offset_marker_xyz,
):
    lines = [
        f"id={marker_id}",
        f"target_frame={target_frame}",
        f"marker_camera_cm={format_cm(marker_camera_xyz)}",
        f"{target_frame}_camera_cm={format_cm(target_camera_xyz)}",
        f"{target_frame}_marker_cm={format_cm(offset_marker_xyz)}",
        "use xyz/100 as MOVE_CUP target offsets",
    ]
    for index, line in enumerate(lines):
        draw_text(frame, line, 30 + index * 28, (0, 255, 255))


def find_dict_block(text, dict_name):
    match = re.search(rf"(?m)^{re.escape(dict_name)}\s*=\s*{{", text)
    if match is None:
        raise ValueError(f"Could not find {dict_name} in task params file")

    depth = 0
    for index in range(match.start(), len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return match.start(), index + 1

    raise ValueError(f"Could not find the end of {dict_name}")


def dict_param_value(text, dict_name, key):
    start, end = find_dict_block(text, dict_name)
    block = text[start:end]
    pattern = re.compile(
        rf'(?m)^\s*"{re.escape(key)}"\s*:\s*'
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,"
    )
    match = pattern.search(block)
    if match is None:
        raise ValueError(f"Could not find {dict_name}[{key!r}] in task params file")
    return float(match.group(1))


def update_dict_param_value(text, dict_name, key, value):
    start, end = find_dict_block(text, dict_name)
    block = text[start:end]
    pattern = re.compile(
        rf'(?m)^(\s*"{re.escape(key)}"\s*:\s*)'
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        r"(\s*,.*)$"
    )
    new_block, count = pattern.subn(rf"\g<1>{value:.5f}\2", block, count=1)
    if count != 1:
        raise ValueError(f"Could not update {dict_name}[{key!r}] in task params file")
    return text[:start] + new_block + text[end:]


def maybe_write_task_params(path, offsets_xyz, std_xyz, sample_count, assume_yes=False):
    path = path.expanduser()
    text = path.read_text(encoding="utf-8")
    old_offsets = np.array(
        [
            dict_param_value(text, TASK_PARAMS_DICT_NAME, "target_x_offset"),
            dict_param_value(text, TASK_PARAMS_DICT_NAME, "target_y_offset"),
            dict_param_value(text, TASK_PARAMS_DICT_NAME, "target_z_offset"),
        ],
        dtype=float,
    )

    print("\nCollected marker-frame offset samples:")
    print(f"  dict={TASK_PARAMS_DICT_NAME}")
    print(f"  samples={sample_count}")
    print(f"  mean_m={format_m(offsets_xyz, digits=5)}")
    print(f"  std_cm={format_cm(std_xyz, digits=2)}")
    print(f"  old_m={format_m(old_offsets, digits=5)}")
    print(f"  file={path}")

    if not assume_yes:
        answer = input("Write mean to MOVE_CUP_PARAMS target offsets? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Did not write task params.")
            return False

    new_text = text
    for key, value in zip(
        ("target_x_offset", "target_y_offset", "target_z_offset"),
        offsets_xyz,
    ):
        new_text = update_dict_param_value(
            new_text,
            TASK_PARAMS_DICT_NAME,
            key,
            float(value),
        )

    path.write_text(new_text, encoding="utf-8")
    print("Updated MOVE_CUP_PARAMS target offsets.")
    return True


def stdin_enter_pressed() -> bool:
    if not sys.stdin.isatty():
        return False
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return False
    sys.stdin.readline()
    return True


def target_marker_index(ids, target_marker_id):
    if ids is None:
        return None

    for index, marker_id in enumerate(ids.flatten()):
        if target_marker_id < 0 or int(marker_id) == target_marker_id:
            return index
    return None


def main() -> int:
    args = build_arg_parser().parse_args()
    sample_goal = args.samples
    if args.write_task_params and sample_goal <= 0:
        sample_goal = 1 if args.capture_on_enter else 10

    calibration_path = resolve_path(args.calibration, base_dir=WORKSPACE_ROOT)
    calibration_width, calibration_height, camera_matrix, dist_coeffs = load_camera_info(calibration_path)
    width = args.width or calibration_width
    height = args.height or calibration_height

    rclpy.init()
    node = rclpy.create_node("measure_move_cup_marker_offset")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)

    detector = make_detector(args.dictionary)
    cap = open_camera(args.camera, width, height, args.fps, args.fourcc)
    if not cap.isOpened():
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(f"Could not open camera {args.camera!r}")

    print(
        "Measuring move-cup target frame offset in marker frame. "
        f"camera={args.camera!r}, marker_id={args.target_marker_id}, "
        f"TF={args.camera_frame}<-{args.target_frame}"
    )
    print("Output units: cm. Divide by 100 for MOVE_CUP_PARAMS target_*_offset meters.")
    print(
        "For wrist-frame move_cup, set MOVE_CUP_PARAMS horizontal_tool_offset "
        "and z_tool_offset to 0.0 before using these offsets."
    )
    if args.capture_on_enter:
        print("Press Enter to capture the current visible offset; Ctrl-C to cancel.")
    if args.write_task_params:
        print(f"Will collect {sample_goal} sample(s), average them, then update task params.")

    last_print_time = 0.0
    printed_samples = 0
    offset_samples = []
    latest_sample = None

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
                latest_sample = None
                if not args.no_display:
                    draw_text(frame, "waiting for ArUco", 30, (0, 0, 255))
                    show_window(WINDOW_NAME, frame, 1)
                continue

            rvecs, tvecs = estimate_marker_poses(
                corners,
                args.marker_length,
                camera_matrix,
                dist_coeffs,
            )
            index = target_marker_index(ids, args.target_marker_id)

            if index is None or index >= len(tvecs) or index >= len(rvecs):
                latest_sample = None
                if not args.no_display:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    draw_text(frame, f"target {args.target_marker_id} not visible", 30, (0, 0, 255))
                    show_window(WINDOW_NAME, frame, 1)
                continue

            marker_camera_xyz = np.asarray(tvecs[index], dtype=float).reshape(3)
            marker_rvec = np.asarray(rvecs[index], dtype=float).reshape(3)
            try:
                target_camera_xyz = lookup_frame_camera_xyz(
                    tf_buffer,
                    args.camera_frame,
                    args.target_frame,
                    args.tf_timeout,
                )
            except Exception as exc:
                latest_sample = None
                if time.monotonic() - last_print_time >= max(0.05, args.print_every):
                    print(
                        f"Waiting for TF {args.camera_frame}<-{args.target_frame}: "
                        f"{exc}{tf_hint(exc)}"
                    )
                    last_print_time = time.monotonic()
                continue

            offset_marker_xyz = marker_frame_offset(
                marker_camera_xyz,
                marker_rvec,
                target_camera_xyz,
            )
            latest_sample = (
                int(ids.flatten()[index]),
                marker_camera_xyz,
                target_camera_xyz,
                offset_marker_xyz,
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
                    latest_sample[0],
                    args.target_frame,
                    marker_camera_xyz,
                    target_camera_xyz,
                    offset_marker_xyz,
                )
                show_window(WINDOW_NAME, frame, 1)

            now = time.monotonic()
            if now - last_print_time >= max(0.05, args.print_every):
                printed_samples += 1
                if not args.capture_on_enter:
                    offset_samples.append(offset_marker_xyz.copy())
                print(
                    f"{args.target_frame}_marker_cm={format_cm(offset_marker_xyz)} "
                    f"marker_camera_cm={format_cm(marker_camera_xyz)} "
                    f"{args.target_frame}_camera_cm={format_cm(target_camera_xyz)} "
                    f"suggested_offsets_m={format_m(offset_marker_xyz)}"
                )
                last_print_time = now
                if (
                    not args.capture_on_enter
                    and sample_goal > 0
                    and printed_samples >= sample_goal
                ):
                    break

            if args.capture_on_enter and stdin_enter_pressed():
                if latest_sample is None:
                    print("No valid visible marker/TF sample to capture yet.")
                    continue
                captured = latest_sample[3].copy()
                offset_samples.append(captured)
                print(
                    f"captured {len(offset_samples)}: "
                    f"{args.target_frame}_marker_cm={format_cm(captured)} "
                    f"suggested_offsets_m={format_m(captured)}"
                )
                if sample_goal > 0 and len(offset_samples) >= sample_goal:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_display:
            try:
                cv2.destroyWindow(WINDOW_NAME)
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
