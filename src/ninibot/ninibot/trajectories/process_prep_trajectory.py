# flake8: noqa
import argparse
import json
from pathlib import Path

from ninibot.trajectories.prep_trajectory import make_processed_template
from ninibot.trajectories.prep_trajectory import plot_joint_trajectory
from ninibot.trajectories.prep_trajectory import plot_trajectory_comparison
from ninibot.trajectories.prep_trajectory import read_raw_csv
from ninibot.trajectories.prep_trajectory import retarget_raw_shape
from ninibot.trajectories.prep_trajectory import retarget_trajectory
from ninibot.trajectories.prep_trajectory import trim_end
from ninibot.trajectories.prep_trajectory import trim_start
from ninibot.trajectories.prep_trajectory import write_latest_copy
from ninibot.trajectories.prep_trajectory import write_processed_csv
from ninibot.trajectories.prep_trajectory import write_processed_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a kinesthetic fine-align-prep recording into a retargetable trajectory template."
    )
    parser.add_argument("raw_csv", help="Raw CSV saved by record_prep_trajectory")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to the raw CSV parent directory.",
    )
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Scale processed duration without changing the taught spatial path.",
    )
    parser.add_argument(
        "--trim-start-sec",
        type=float,
        default=0.0,
        help="Drop this many seconds from the beginning before smoothing.",
    )
    parser.add_argument(
        "--trim-end-sec",
        type=float,
        default=0.0,
        help="Drop this many seconds from the end after start trimming.",
    )
    parser.add_argument(
        "--path-smoothing",
        default="spline",
        choices=["spline", "savgol", "none"],
        help="Path smoothing method applied to delta_shape.",
    )
    parser.add_argument(
        "--joint4-mode",
        default="horizontal_offset",
        choices=["horizontal_offset", "recorded"],
        help="Use recorded joint4 shape or recompute joint4 as horizontal wrist plus a decaying offset.",
    )
    parser.add_argument(
        "--endpoint-radius-offset-m",
        type=float,
        default=0.0,
        help="Offset the processed endpoint radius before saving the retargetable template.",
    )
    parser.add_argument(
        "--endpoint-z-offset-m",
        type=float,
        default=0.0,
        help="Offset the processed endpoint z before saving the retargetable template.",
    )
    parser.add_argument(
        "--time-profile",
        default="linear",
        choices=["linear", "minimum_jerk"],
        help="Runtime progress profile applied after path smoothing.",
    )
    parser.add_argument(
        "--jerk-smoothing",
        default="none",
        choices=["none", "savgol"],
        help="Optional second-pass position smoothing before derivative metrics are computed.",
    )
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--smooth-polyorder", type=int, default=3)
    parser.add_argument("--jerk-smooth-window", type=int, default=41)
    parser.add_argument("--jerk-smooth-polyorder", type=int, default=5)
    parser.add_argument(
        "--spline-smoothing",
        type=float,
        default=0.0003,
        help="Dimensionless smoothing strength for spline path smoothing.",
    )
    parser.add_argument(
        "--flat-velocity-threshold-deg-s",
        type=float,
        default=2.0,
        help="Treat longer low-speed regions below this joint velocity as flat.",
    )
    parser.add_argument(
        "--flat-min-duration-sec",
        type=float,
        default=0.35,
        help="Minimum low-speed run length that receives extra spline weight.",
    )
    parser.add_argument(
        "--flat-weight",
        type=float,
        default=20.0,
        help="Spline fitting weight multiplier for detected flat regions.",
    )
    parser.add_argument(
        "--shape-scale",
        type=float,
        default=1.0,
        help="Scale the demonstrated shape deviation before saving.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the diagnostic plot PNG.",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not update latest_processed.json in the output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_path = Path(args.raw_csv)
    output_dir = Path(args.output_dir) if args.output_dir else raw_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_times, raw_positions = read_raw_csv(raw_path)
    raw_duration_sec = float(raw_times[-1] - raw_times[0])
    raw_times, raw_positions = trim_start(
        raw_times,
        raw_positions,
        trim_start_sec=args.trim_start_sec,
    )
    raw_times, raw_positions = trim_end(
        raw_times,
        raw_positions,
        trim_end_sec=args.trim_end_sec,
    )

    template = make_processed_template(
        raw_times,
        raw_positions,
        sample_rate_hz=args.sample_rate,
        duration_scale=args.duration_scale,
        path_smoothing=args.path_smoothing,
        joint4_mode=args.joint4_mode,
        endpoint_radius_offset_m=args.endpoint_radius_offset_m,
        endpoint_z_offset_m=args.endpoint_z_offset_m,
        time_profile=args.time_profile,
        jerk_smoothing=args.jerk_smoothing,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
        jerk_smooth_window=args.jerk_smooth_window,
        jerk_smooth_polyorder=args.jerk_smooth_polyorder,
        spline_smoothing=args.spline_smoothing,
        flat_velocity_threshold_deg_s=args.flat_velocity_threshold_deg_s,
        flat_min_duration_sec=args.flat_min_duration_sec,
        flat_weight=args.flat_weight,
        shape_scale=args.shape_scale,
    )
    template["source_raw_csv"] = str(raw_path)
    template["source_raw_duration_sec"] = raw_duration_sec
    template["trim_start_sec"] = float(args.trim_start_sec)
    template["trim_end_sec"] = float(args.trim_end_sec)
    template["processor_options"] = {
        "sample_rate": args.sample_rate,
        "duration_scale": args.duration_scale,
        "trim_start_sec": args.trim_start_sec,
        "trim_end_sec": args.trim_end_sec,
        "path_smoothing": args.path_smoothing,
        "joint4_mode": args.joint4_mode,
        "endpoint_radius_offset_m": args.endpoint_radius_offset_m,
        "endpoint_z_offset_m": args.endpoint_z_offset_m,
        "time_profile": args.time_profile,
        "jerk_smoothing": args.jerk_smoothing,
        "smooth_window": args.smooth_window,
        "smooth_polyorder": args.smooth_polyorder,
        "jerk_smooth_window": args.jerk_smooth_window,
        "jerk_smooth_polyorder": args.jerk_smooth_polyorder,
        "spline_smoothing": args.spline_smoothing,
        "flat_velocity_threshold_deg_s": args.flat_velocity_threshold_deg_s,
        "flat_min_duration_sec": args.flat_min_duration_sec,
        "flat_weight": args.flat_weight,
        "shape_scale": args.shape_scale,
    }

    processed_json = output_dir / "processed_template.json"
    processed_csv = output_dir / "processed_samples.csv"
    metrics_json = output_dir / "processing_metrics.json"
    plot_path = None
    comparison_plot_path = None

    write_processed_json(processed_json, template)
    write_processed_csv(processed_csv, template)

    metrics = {
        "duration_sec": template["duration_sec"],
        "sample_rate_hz": template["sample_rate_hz"],
        "duration_scale": float(args.duration_scale),
        "joint_names": template["joint_names"],
        "max_abs_velocity_rad_s": template["max_abs_velocity_rad_s"],
        "max_abs_acceleration_rad_s2": template["max_abs_acceleration_rad_s2"],
        "max_abs_jerk_rad_s3": template["max_abs_jerk_rad_s3"],
        "source_raw_duration_sec": raw_duration_sec,
        "trim_start_sec": float(args.trim_start_sec),
        "trim_end_sec": float(args.trim_end_sec),
        "path_smoothing": args.path_smoothing,
        "joint4_mode": args.joint4_mode,
        "endpoint_radius_offset_m": float(args.endpoint_radius_offset_m),
        "endpoint_z_offset_m": float(args.endpoint_z_offset_m),
        "endpoint_adjustment": template.get("endpoint_adjustment"),
        "joint4_offset_profile": template.get("joint4_offset_profile"),
        "time_profile": args.time_profile,
        "jerk_smoothing": args.jerk_smoothing,
        "jerk_smooth_window": int(args.jerk_smooth_window),
        "jerk_smooth_polyorder": int(args.jerk_smooth_polyorder),
        "spline_smoothing": float(args.spline_smoothing),
        "flat_velocity_threshold_deg_s": float(args.flat_velocity_threshold_deg_s),
        "flat_min_duration_sec": float(args.flat_min_duration_sec),
        "flat_weight": float(args.flat_weight),
        "processed_json": str(processed_json),
        "processed_csv": str(processed_csv),
    }

    if not args.no_plot:
        plot_path = output_dir / "processed_joint_dynamics.png"
        comparison_plot_path = output_dir / "raw_vs_processed_comparison.png"
        try:
            plot_start = template["processed_start_rad"]
            plot_end = template["processed_end_rad"]
            raw_plot_times, raw_plot_positions = retarget_raw_shape(
                raw_times,
                raw_positions,
                plot_start,
                plot_end,
                sample_rate_hz=template["sample_rate_hz"],
                shape_scale=1.0,
            )
            plot_times, plot_positions = retarget_trajectory(
                template,
                plot_start,
                plot_end,
                duration_sec=template["duration_sec"],
                sample_rate_hz=template["sample_rate_hz"],
                shape_scale=1.0,
            )
            plot_joint_trajectory(
                plot_path,
                plot_times,
                plot_positions,
                "Fine-align-prep processed trajectory",
            )
            plot_trajectory_comparison(
                comparison_plot_path,
                raw_plot_times,
                raw_plot_positions,
                plot_times,
                plot_positions,
                "Fine-align-prep collected vs processed comparison",
            )
            metrics["plot"] = str(plot_path)
            metrics["comparison_plot"] = str(comparison_plot_path)
            metrics["plot_trajectory"] = "taught_start_to_taught_end_processed"
            metrics["plot_start_rad"] = list(plot_start)
            metrics["plot_end_rad"] = list(plot_end)
        except RuntimeError as exc:
            plot_path = None
            comparison_plot_path = None
            print(f"Warning: {exc}")

    if not args.no_latest:
        latest = output_dir / "latest_processed.json"
        write_latest_copy(processed_json, latest)
        metrics["latest_processed_json"] = str(latest)

        if plot_path is not None:
            plot_alias = output_dir / "plot.png"
            write_latest_copy(plot_path, plot_alias)
            metrics["plot_png"] = str(plot_alias)

        if comparison_plot_path is not None:
            comparison_plot_alias = output_dir / "comparison_plot.png"
            write_latest_copy(comparison_plot_path, comparison_plot_alias)
            metrics["comparison_plot_png"] = str(comparison_plot_alias)

    with metrics_json.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")

    print(f"Wrote {processed_json}")
    print(f"Wrote {processed_csv}")
    print(f"Wrote {metrics_json}")


if __name__ == "__main__":
    main()
