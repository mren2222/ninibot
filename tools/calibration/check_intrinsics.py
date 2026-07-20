#!/usr/bin/env python3
"""Check camera intrinsics with ArUco corner reprojection error."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
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
from vision_controller.aruco_detection import marker_object_points
from vision_controller.aruco_detection import open_camera
from vision_controller.aruco_detection import reprojection_errors
from vision_controller.aruco_detection import resolve_path
from vision_controller.aruco_detection import show_window

DATA_DIR = WORKSPACE_ROOT / "data" / "calibration"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check intrinsics using ArUco reprojection error.")
    parser.add_argument("--calibration", default=str(DATA_DIR / "camera_calibration.yaml"))
    parser.add_argument("--camera-id", type=int, default=4)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="YUYV")
    parser.add_argument("--marker-length", type=float, default=0.05)
    parser.add_argument("--dictionary", choices=sorted(ARUCO_DICTIONARIES), default="4X4_50")
    parser.add_argument("--target-marker-id", type=int, default=0)
    parser.add_argument("--image")
    parser.add_argument("--output")
    parser.add_argument("--save-last")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--print-every", type=float, default=0.5)
    return parser


def check_frame(frame, detector, object_points, marker_length, target_marker_id, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector(gray)
    if ids is None:
        draw_text(frame, "No ArUco marker detected", 30, (0, 0, 255))
        return None

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    rvecs, tvecs = estimate_marker_poses(corners, marker_length, camera_matrix, dist_coeffs)

    for index, marker_id in enumerate(ids.flatten()):
        if int(marker_id) != target_marker_id or index >= len(rvecs):
            continue

        mean_px, rms_px, max_px = reprojection_errors(
            object_points,
            corners[index],
            rvecs[index],
            tvecs[index],
            camera_matrix,
            dist_coeffs,
        )
        if hasattr(cv2, "drawFrameAxes"):
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvecs[index].reshape(3, 1),
                tvecs[index].reshape(3, 1),
                marker_length * 0.5,
            )

        text = f"id={int(marker_id)} mean={mean_px:.2f}px rms={rms_px:.2f}px max={max_px:.2f}px"
        draw_text(frame, text, 30, (0, 255, 255))
        return mean_px, rms_px, max_px, text

    draw_text(frame, f"Target marker {target_marker_id} not visible", 30, (0, 0, 255))
    return None


def summarize(results):
    if not results:
        print("No usable target marker observations.")
        return 1

    means = [item[0] for item in results]
    rms = [item[1] for item in results]
    maxes = [item[2] for item in results]
    print("\nSummary:")
    print(f"  observations={len(results)}")
    print(f"  reproj_mean_px mean={statistics.fmean(means):.2f}, max={max(means):.2f}")
    print(f"  reproj_rms_px  mean={statistics.fmean(rms):.2f}, max={max(rms):.2f}")
    print(f"  reproj_max_px  mean={statistics.fmean(maxes):.2f}, max={max(maxes):.2f}")
    return 0


def run_image(args, detector, object_points, camera_matrix, dist_coeffs):
    frame = cv2.imread(str(resolve_path(args.image)))
    if frame is None:
        print(f"Failed to read image: {args.image}")
        return 1

    result = check_frame(
        frame,
        detector,
        object_points,
        args.marker_length,
        args.target_marker_id,
        camera_matrix,
        dist_coeffs,
    )
    if result:
        print(result[3])

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        print(f"Saved annotated image to {output_path}")

    if not args.no_display:
        show_window("intrinsics check", frame, 0)
        destroy_windows()
    return summarize([] if result is None else [result[:3]])


def run_camera(args, image_width, image_height, detector, object_points, camera_matrix, dist_coeffs):
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

            result = check_frame(
                frame,
                detector,
                object_points,
                args.marker_length,
                args.target_marker_id,
                camera_matrix,
                dist_coeffs,
            )
            last_frame = frame
            if result:
                results.append(result[:3])
                now = time.monotonic()
                if now - last_print >= args.print_every:
                    print(result[3])
                    last_print = now

            if display_enabled:
                key, display_enabled = show_window("intrinsics check", frame, 1)
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
    object_points = marker_object_points(args.marker_length)

    print(f"Loaded intrinsics: {calibration_path}")
    print(f"Dictionary: {args.dictionary}, marker_id={args.target_marker_id}, marker_length={args.marker_length:.3f} m")

    if args.image:
        return run_image(args, detector, object_points, camera_matrix, dist_coeffs)
    return run_camera(args, image_width, image_height, detector, object_points, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    raise SystemExit(main())
