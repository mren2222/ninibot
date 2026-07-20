#!/usr/bin/env python3
"""Generate ArUco marker PNG files into this asset directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from cv2 import aruco


DICTIONARIES = {
    '4X4_50': aruco.DICT_4X4_50,
    '6X6_50': aruco.DICT_6X6_50,
}


def parse_marker_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(',') if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate ArUco marker PNG assets.')
    parser.add_argument('--dictionary', choices=sorted(DICTIONARIES), default='4X4_50')
    parser.add_argument('--ids', default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--size-px', type=int, default=600)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aruco_dict = aruco.getPredefinedDictionary(DICTIONARIES[args.dictionary])
    prefix = args.dictionary.lower()

    for marker_id in parse_marker_ids(args.ids):
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, args.size_px)
        output_path = args.output_dir / f'aruco_{prefix}_id_{marker_id}.png'
        cv2.imwrite(str(output_path), marker_img)
        print(f'Saved {output_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
