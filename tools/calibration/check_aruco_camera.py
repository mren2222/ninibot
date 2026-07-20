#!/usr/bin/env python3
"""Print ArUco marker xyz in the calibrated camera frame."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = TOOL_DIR.parents[1]
VISION_SRC = WORKSPACE_ROOT / "src" / "vision_controller"
if VISION_SRC.exists():
    sys.path.insert(0, str(VISION_SRC))

from vision_controller.aruco_detection import ARUCO_DICTIONARIES
from vision_controller.aruco_detection import destroy_windows
from vision_controller.aruco_detection import draw_text
from vision_controller.aruco_detection import estimate_marker_poses
from vision_controller.aruco_detection import load_camera_info
from vision_controller.aruco_detection import make_detector
from vision_controller.aruco_detection import open_camera
from vision_controller.aruco_detection import resolve_path
from vision_controller.aruco_detection import show_window

DATA_DIR = WORKSPACE_ROOT / "data" / "calibration"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check ArUco pose estimation in the camera optical frame."
    )
    parser.add_argument("--calibration", default=str(DATA_DIR / "camera_calibration.yaml"))
    parser.add_argument("--camera-id", type=int, default=4)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="YUYV")
    parser.add_argument("--marker-length", type=float, default=0.05)
    parser.add_argument("--dictionary", choices=sorted(ARUCO_DICTIONARIES), default="4X4_50")
    parser.add_argument("--target-marker-id", type=int, default=0)
    parser.add_argument("--all-markers", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--output")
    parser.add_argument("--save-last")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--print-every", type=float, default=0.5)
    return parser


def format_xyz_cm(vector) -> str:
    xyz = np.asarray(vector, dtype=float) * 100.0
    return f"({xyz[0]:+.1f}, {xyz[1]:+.1f}, {xyz[2]:+.1f})cm"


def check_frame(frame, detector, marker_length, target_marker_id, all_markers, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector(gray)
    if ids is None:
        draw_text(frame, "No ArUco marker detected", 30, (0, 0, 255))
        return []

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    rvecs, tvecs = estimate_marker_poses(corners, marker_length, camera_matrix, dist_coeffs)
    results = []

    for index, marker_id in enumerate(ids.flatten()):
        if index >= len(tvecs):
            continue
        if not all_markers and int(marker_id) != target_marker_id:
            continue

        marker_camera_xyz = tvecs[index].astype(float)
        distance = float(np.linalg.norm(marker_camera_xyz))

        if hasattr(cv2, "drawFrameAxes") and index < len(rvecs):
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvecs[index].reshape(3, 1),
                tvecs[index].reshape(3, 1),
                marker_length * 0.5,
            )

        text = (
            f"id={int(marker_id)} marker_camera_xyz={format_xyz_cm(marker_camera_xyz)} "
            f"dist={distance * 100.0:.1f}cm"
        )
        draw_text(frame, text, 30 + len(results) * 28, (0, 255, 255))
        results.append((marker_camera_xyz, distance, text))

    if not results:
        draw_text(frame, f"Target marker {target_marker_id} not visible", 30, (0, 0, 255))
    return results


def summarize(results):
    if not results:
        print("No usable marker observations.")
        return 1

    arr = np.array([[*item[0], item[1]] for item in results], dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    print("\nSummary:")
    print(f"  observations={len(results)}")
    print(
        "  marker_camera_xyz_cm "
        f"mean=({mean[0] * 100.0:.1f}, {mean[1] * 100.0:.1f}, "
        f"{mean[2] * 100.0:.1f})"
    )
    print(
        "  marker_camera_xyz_cm "
        f"std =({std[0] * 100.0:.2f}, {std[1] * 100.0:.2f}, "
        f"{std[2] * 100.0:.2f})"
    )
    print(f"  distance_cm mean={mean[3] * 100.0:.1f}, std={std[3] * 100.0:.2f}")
    return 0


def run_image(args, detector, camera_matrix, dist_coeffs):
    frame = cv2.imread(str(resolve_path(args.image)))
    if frame is None:
        print(f"Failed to read image: {args.image}")
        return 1

    results = check_frame(
        frame,
        detector,
        args.marker_length,
        args.target_marker_id,
        args.all_markers,
        camera_matrix,
        dist_coeffs,
    )
    for result in results:
        print(result[2])

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        print(f"Saved annotated image to {output_path}")

    if not args.no_display:
        show_window("aruco camera check", frame, 0)
        destroy_windows()
    return summarize(results)


def run_camera(args, image_width, image_height, detector, camera_matrix, dist_coeffs):
    cap = open_camera(
        args.camera_id,
        args.width or image_width,
        args.height or image_height,
        args.fps,
        args.fourcc,
    )
    if not cap.isOpened():
        print(f"Failed to open camera index {args.camera_id}")
        return 1

    print(f"Camera opened: index={args.camera_id}")
    display_enabled = not args.no_display
    results = []
    last_print = 0.0
    last_frame = None
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                return 1

            frame_results = check_frame(
                frame,
                detector,
                args.marker_length,
                args.target_marker_id,
                args.all_markers,
                camera_matrix,
                dist_coeffs,
            )
            last_frame = frame
            if frame_results:
                results.extend(frame_results)
                now = time.monotonic()
                if now - last_print >= args.print_every:
                    for result in frame_results:
                        print(result[2])
                    last_print = now

            if display_enabled:
                key, display_enabled = show_window("aruco camera check", frame, 1)
                if key in (ord("q"), 27):
                    break
            else:
                time.sleep(0.01)

            frame_count += 1
            if args.frames is not None and frame_count >= args.frames:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        if args.save_last and last_frame is not None:
            save_last_path = resolve_path(args.save_last)
            save_last_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_last_path), last_frame)
        destroy_windows()

    return summarize(results)


def main() -> int:
    args = build_arg_parser().parse_args()
    calibration_path = resolve_path(args.calibration)
    image_width, image_height, camera_matrix, dist_coeffs = load_camera_info(calibration_path)
    detector = make_detector(args.dictionary)

    print(f"Loaded intrinsics: {calibration_path}")
    print(
        f"Dictionary: {args.dictionary}, marker_id={args.target_marker_id}, "
        f"marker_length={args.marker_length:.3f} m"
    )

    if args.image:
        return run_image(args, detector, camera_matrix, dist_coeffs)
    return run_camera(args, image_width, image_height, detector, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    raise SystemExit(main())
