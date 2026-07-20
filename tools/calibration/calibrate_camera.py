#!/usr/bin/env python3
"""Calibrate camera intrinsics from chessboard images with OpenCV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


TOOL_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = TOOL_DIR.parents[1]
DATA_DIR = WORKSPACE_ROOT / "data" / "calibration"
DEFAULT_PATTERN = "8x6"
DEFAULT_SQUARE_SIZE = 0.028
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def parse_pattern(value: str) -> tuple[int, int]:
    try:
        cols_text, rows_text = value.lower().split("x", maxsplit=1)
        cols = int(cols_text)
        rows = int(rows_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pattern must look like 8x6") from exc

    if cols <= 0 or rows <= 0:
        raise argparse.ArgumentTypeError("pattern dimensions must be positive")

    return cols, rows


def resolve_path(path_text: str | Path, *, for_output: bool = False) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    script_candidate = (TOOL_DIR / path).resolve()
    if cwd_candidate.exists() or (path.parts and path.parts[0] == TOOL_DIR.name):
        return cwd_candidate
    if for_output and path.parent != Path("."):
        return cwd_candidate
    return script_candidate


def find_images(image_dir: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(pattern))
        images.extend(image_dir.glob(pattern.upper()))
    return sorted(set(images))


def make_object_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def detect_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ok:
            return True, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def format_float_list(values: np.ndarray | list[float]) -> str:
    flattened = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{value:.12g}" for value in flattened) + "]"


def write_ros_camera_yaml(
    output_path: Path,
    camera_name: str,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    width, height = image_size
    distortion = np.zeros(5, dtype=float)
    source_distortion = dist_coeffs.reshape(-1)
    distortion[: min(5, source_distortion.size)] = source_distortion[:5]

    projection_matrix = np.array(
        [
            camera_matrix[0, 0],
            0.0,
            camera_matrix[0, 2],
            0.0,
            0.0,
            camera_matrix[1, 1],
            camera_matrix[1, 2],
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        dtype=float,
    )

    text = (
        f"image_width: {width}\n"
        f"image_height: {height}\n"
        f"camera_name: {camera_name}\n"
        "camera_matrix:\n"
        "  rows: 3\n"
        "  cols: 3\n"
        f"  data: {format_float_list(camera_matrix)}\n"
        "distortion_model: plumb_bob\n"
        "distortion_coefficients:\n"
        "  rows: 1\n"
        "  cols: 5\n"
        f"  data: {format_float_list(distortion)}\n"
        "rectification_matrix:\n"
        "  rows: 3\n"
        "  cols: 3\n"
        f"  data: {format_float_list(np.eye(3))}\n"
        "projection_matrix:\n"
        "  rows: 3\n"
        "  cols: 4\n"
        f"  data: {format_float_list(projection_matrix)}\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def reprojection_errors(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[float, list[float]]:
    total_error = 0.0
    total_points = 0
    per_image: list[float] = []

    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2)
        point_count = len(projected)
        per_image.append(error / point_count)
        total_error += error * error
        total_points += point_count

    return float(np.sqrt(total_error / total_points)), per_image


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from captured chessboard images.",
    )
    parser.add_argument(
        "--image-dir",
        default=str(DATA_DIR / "images"),
        help="Directory containing calibration images.",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "camera_calibration.yaml"),
        help="Output ROS camera_info YAML path.",
    )
    parser.add_argument(
        "--pattern",
        type=parse_pattern,
        default=parse_pattern(DEFAULT_PATTERN),
        help="Inner-corner chessboard pattern as COLSxROWS, for example 8x6.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=DEFAULT_SQUARE_SIZE,
        help="Real square size in meters, for example 0.028 for 28 mm.",
    )
    parser.add_argument(
        "--camera-name",
        default="innomaker_u20cam",
        help="camera_name written to the YAML file.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=10,
        help="Minimum number of usable chessboard images required.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show corner detections in an OpenCV window while processing.",
    )
    parser.add_argument(
        "--debug-dir",
        help="Optional directory for images with detected corners drawn.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.square_size <= 0:
        print("--square-size must be positive", file=sys.stderr)
        return 2

    image_dir = resolve_path(args.image_dir)
    output_path = resolve_path(args.output, for_output=True)
    debug_dir = resolve_path(args.debug_dir) if args.debug_dir else None

    if not image_dir.exists():
        print(f"No image directory found: {image_dir}", file=sys.stderr)
        return 1

    images = find_images(image_dir)
    if not images:
        print(f"No calibration images found in: {image_dir}", file=sys.stderr)
        return 1

    object_template = make_object_points(args.pattern, args.square_size)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    print(f"Found {len(images)} image(s) in {image_dir}")
    print(
        "Using chessboard inner corners "
        f"{args.pattern[0]}x{args.pattern[1]}, square size {args.square_size} m"
    )

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(
                f"Skipped {image_path.name}: image size {current_size} "
                f"does not match first image size {image_size}"
            )
            continue

        ok, corners = detect_corners(gray, args.pattern)
        if not ok or corners is None:
            print(f"Skipped, chessboard not found: {image_path.name}")
            continue

        objpoints.append(object_template.copy())
        imgpoints.append(corners)
        print(f"Used: {image_path.name}")

        if args.show or debug_dir:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, args.pattern, corners, ok)
            if debug_dir:
                cv2.imwrite(str(debug_dir / image_path.name), vis)
            if args.show:
                cv2.imshow("calibration corners", vis)
                cv2.waitKey(150)

    if args.show:
        cv2.destroyAllWindows()

    if len(objpoints) < args.min_images:
        print(
            f"Only {len(objpoints)} valid image(s). "
            f"Capture at least {args.min_images} varied views.",
            file=sys.stderr,
        )
        return 1

    if image_size is None:
        print("No readable calibration images found.", file=sys.stderr)
        return 1

    print(f"Calibrating with {len(objpoints)} valid image(s)...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    mean_error, per_image_errors = reprojection_errors(
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )

    print("\nRMS reprojection error:")
    print(rms)
    print("\nMean reprojection error:")
    print(mean_error)
    print("\nCamera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)

    worst_error = max(per_image_errors)
    print(f"\nWorst per-image reprojection error: {worst_error:.4f} px")

    write_ros_camera_yaml(
        output_path,
        args.camera_name,
        image_size,
        camera_matrix,
        dist_coeffs,
    )
    print(f"\nSaved calibration to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
