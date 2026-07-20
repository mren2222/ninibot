#!/usr/bin/env python3
"""Create a letter-size PDF from generated ArUco marker PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


MARKER_SIZE_CM = 5.0
GAP_CM = 2
COLS = 3
ROWS = 5


def build_arg_parser() -> argparse.ArgumentParser:
    asset_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Build an ArUco marker print PDF.')
    parser.add_argument('--input-dir', type=Path, default=asset_dir)
    parser.add_argument('--pattern', default='aruco_4x4_50_id_*.png')
    parser.add_argument(
        '--output',
        type=Path,
        default=asset_dir / 'aruco_4x4_50_ids_0_to_9_letter_5cm_with_gap.pdf',
    )
    parser.add_argument('--marker-size-cm', type=float, default=MARKER_SIZE_CM)
    parser.add_argument('--gap-cm', type=float, default=GAP_CM)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    page_width, page_height = letter

    marker_size = args.marker_size_cm * cm
    gap = args.gap_cm * cm

    grid_width = COLS * marker_size + (COLS - 1) * gap
    margin_x = (page_width - grid_width) / 2
    margin_y = 1 * cm

    png_files = sorted(args.input_dir.glob(args.pattern))
    if len(png_files) == 0:
        raise ValueError(f'No marker PNG files found: {args.input_dir / args.pattern}')

    png_files = png_files[:10]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(args.output), pagesize=letter)

    for index, png_path in enumerate(png_files):
        row = index // COLS
        col = index % COLS

        x = margin_x + col * (marker_size + gap)
        y = page_height - margin_y - marker_size - row * (marker_size + gap)

        pdf.drawImage(
            str(png_path),
            x,
            y,
            width=marker_size,
            height=marker_size,
            preserveAspectRatio=True,
            mask='auto',
        )

    pdf.save()

    print(f'Saved PDF: {args.output}')
    print(f'Each marker size: {args.marker_size_cm} cm x {args.marker_size_cm} cm')
    print(f'Gap between markers: {args.gap_cm} cm')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
