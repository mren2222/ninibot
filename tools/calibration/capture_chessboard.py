#!/usr/bin/env python3
"""Capture chessboard images for camera calibration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2


TOOL_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = TOOL_DIR.parents[1]
DATA_DIR = WORKSPACE_ROOT / "data" / "calibration"
DEFAULT_PATTERN = "8x6"


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


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    script_candidate = (TOOL_DIR / path).resolve()
    if cwd_candidate.exists() or (path.parts and path.parts[0] == TOOL_DIR.name):
        return cwd_candidate
    return script_candidate


def next_capture_index(save_dir: Path) -> int:
    highest = -1
    pattern = re.compile(r"calib_(\d+)\.png$")
    for image_path in save_dir.glob("calib_*.png"):
        match = pattern.match(image_path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture chessboard images from a camera.")
    parser.add_argument("--camera-id", type=int, default=4, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=640, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=480, help="Requested frame height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument(
        "--fourcc",
        default="YUYV",
        help="Requested pixel format/FourCC, for example YUYV or MJPG.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(DATA_DIR / "images"),
        help="Directory where captured images are saved.",
    )
    parser.add_argument(
        "--pattern",
        type=parse_pattern,
        default=parse_pattern(DEFAULT_PATTERN),
        help="Inner-corner chessboard pattern as COLSxROWS, for example 8x6.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if args.fourcc:
        fourcc = args.fourcc.upper()
        if len(fourcc) != 4:
            print("--fourcc must be exactly 4 characters, for example YUYV or MJPG")
            return 2
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    if not cap.isOpened():
        print(f"Failed to open camera index {args.camera_id}")
        return 1

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: index={args.camera_id}, size={frame_width}x{frame_height}")
    print(f"Saving captures to: {save_dir}")
    print("SPACE: save image | q/ESC: quit")

    count = next_capture_index(save_dir)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            args.pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            cv2.drawChessboardCorners(display, args.pattern, corners, found)

        status = "FOUND" if found else "searching"
        cv2.putText(
            display,
            f"{status} | next: calib_{count:03d}.png | SPACE save | q quit",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 0) if found else (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("capture chessboard", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            filename = save_dir / f"calib_{count:03d}.png"
            cv2.imwrite(str(filename), frame)
            print(f"Saved {filename}")
            count += 1
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
