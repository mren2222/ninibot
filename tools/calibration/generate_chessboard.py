#!/usr/bin/env python3
"""Generate a printable chessboard image for OpenCV calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a chessboard PNG.")
    parser.add_argument(
        "--pattern",
        type=parse_pattern,
        default=parse_pattern(DEFAULT_PATTERN),
        help="Inner-corner chessboard pattern as COLSxROWS, for example 8x6.",
    )
    parser.add_argument(
        "--square-px",
        type=int,
        default=120,
        help="Square size in pixels in the generated image.",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "chessboard_8x6_inner.png"),
        help="Output PNG path.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.square_px <= 0:
        print("--square-px must be positive")
        return 2

    inner_cols, inner_rows = args.pattern
    squares_x = inner_cols + 1
    squares_y = inner_rows + 1
    img_w = squares_x * args.square_px
    img_h = squares_y * args.square_px

    img = np.ones((img_h, img_w), dtype=np.uint8) * 255
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                cv2.rectangle(
                    img,
                    (x * args.square_px, y * args.square_px),
                    ((x + 1) * args.square_px, (y + 1) * args.square_px),
                    0,
                    -1,
                )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Saved {output_path}")
    print(
        f"Inner corners: {inner_cols}x{inner_rows}; "
        f"squares: {squares_x}x{squares_y}; image: {img_w}x{img_h} px"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
