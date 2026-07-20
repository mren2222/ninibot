# flake8: noqa
import csv
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


JOINT_NAMES = ("2", "3", "4")
RAD_FIELDS = ("joint2_rad", "joint3_rad", "joint4_rad")
DEG_FIELDS = ("joint2_deg", "joint3_deg", "joint4_deg")
DELTA_FIELDS = ("delta_shape_joint2_rad", "delta_shape_joint3_rad", "delta_shape_joint4_rad")


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def odd_window_length(requested, sample_count):
    if sample_count < 3:
        return sample_count

    window = int(requested)
    window = max(3, min(window, sample_count))

    if window % 2 == 0:
        window -= 1

    return max(3, window)


def moving_average(values, window):
    if window <= 1 or len(values) < 3:
        return np.asarray(values, dtype=float)

    window = odd_window_length(window, len(values))
    if window <= 1:
        return np.asarray(values, dtype=float)

    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    columns = []

    for index in range(values.shape[1]):
        columns.append(np.convolve(padded[:, index], kernel, mode="valid"))

    return np.stack(columns, axis=1)


def smooth_positions(positions, window_length=15, polyorder=3):
    positions = np.asarray(positions, dtype=float)

    if len(positions) < 5:
        return positions.copy()

    window = odd_window_length(window_length, len(positions))

    try:
        from scipy.signal import savgol_filter

        order = min(int(polyorder), window - 1)
        return savgol_filter(
            positions,
            window_length=window,
            polyorder=max(1, order),
            axis=0,
            mode="interp",
        )
    except Exception:
        return moving_average(positions, window)


def read_raw_csv(path):
    path = Path(path)
    rows = []

    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)

        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No trajectory rows found in {path}")

    times = np.asarray([float(row["t_sec"]) for row in rows], dtype=float)
    positions = np.asarray(
        [
            [float(row[field]) for field in RAD_FIELDS]
            for row in rows
        ],
        dtype=float,
    )

    times = times - times[0]

    keep = [0]
    for index in range(1, len(times)):
        if times[index] > times[keep[-1]]:
            keep.append(index)

    times = times[keep]
    positions = positions[keep]

    if len(times) < 2:
        raise ValueError(f"Need at least two unique samples in {path}")

    return times, positions


def resample_uniform(times, positions, sample_rate_hz=100.0):
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    duration = float(times[-1] - times[0])

    if duration <= 0.0:
        raise ValueError("Trajectory duration must be positive")

    dt = 1.0 / float(sample_rate_hz)
    count = max(2, int(math.ceil(duration / dt)) + 1)
    uniform_times = np.linspace(0.0, duration, count)
    columns = []

    for index in range(positions.shape[1]):
        columns.append(np.interp(uniform_times, times, positions[:, index]))

    return uniform_times, np.stack(columns, axis=1)


def trim_start(times, positions, trim_start_sec=0.0):
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    trim = float(trim_start_sec)

    if trim <= 0.0:
        return times - times[0], positions.copy()

    duration = float(times[-1] - times[0])

    if trim >= duration:
        raise ValueError(
            f"trim_start_sec={trim:.3f} must be smaller than trajectory duration={duration:.3f}"
        )

    trim_time = times[0] + trim
    start_position = np.asarray(
        [
            np.interp(trim_time, times, positions[:, index])
            for index in range(positions.shape[1])
        ],
        dtype=float,
    )
    keep = times > trim_time
    trimmed_times = np.concatenate(([0.0], times[keep] - trim_time))
    trimmed_positions = np.vstack((start_position, positions[keep]))

    if len(trimmed_times) < 2:
        raise ValueError("Need at least two samples after trimming")

    return trimmed_times, trimmed_positions


def trim_end(times, positions, trim_end_sec=0.0):
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    trim = float(trim_end_sec)

    if trim <= 0.0:
        return times - times[0], positions.copy()

    duration = float(times[-1] - times[0])

    if trim >= duration:
        raise ValueError(
            f"trim_end_sec={trim:.3f} must be smaller than trajectory duration={duration:.3f}"
        )

    end_time = times[-1] - trim
    end_position = np.asarray(
        [
            np.interp(end_time, times, positions[:, index])
            for index in range(positions.shape[1])
        ],
        dtype=float,
    )
    keep = times < end_time
    trimmed_times = np.concatenate((times[keep], [end_time]))
    trimmed_positions = np.vstack((positions[keep], end_position))
    trimmed_times = trimmed_times - trimmed_times[0]

    if len(trimmed_times) < 2:
        raise ValueError("Need at least two samples after trimming")

    return trimmed_times, trimmed_positions


def finite_difference(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)

    if len(times) < 2:
        return np.zeros_like(values)

    return np.gradient(values, times, axis=0, edge_order=1)


def linear_baseline(start, end, s_values):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    s_values = np.asarray(s_values, dtype=float).reshape(-1, 1)
    return start + s_values * (end - start)


def smootherstep(s):
    s = max(0.0, min(1.0, float(s)))
    return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)


def smootherstep_values(values):
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return values * values * values * (values * (values * 6.0 - 15.0) + 10.0)


def time_profile_values(s_values, profile="linear"):
    profile = str(profile).lower()
    s_values = np.asarray(s_values, dtype=float)

    if profile in ("linear", "none", "raw"):
        return s_values

    if profile in ("minimum_jerk", "min_jerk", "smootherstep"):
        return smootherstep_values(s_values)

    raise ValueError(f"Unsupported time_profile={profile!r}")


def resample_positions_by_progress(positions, source_s_values, query_s_values):
    positions = np.asarray(positions, dtype=float)
    source_s_values = np.asarray(source_s_values, dtype=float)
    query_s_values = np.asarray(query_s_values, dtype=float)
    columns = []

    for index in range(positions.shape[1]):
        columns.append(np.interp(query_s_values, source_s_values, positions[:, index]))

    return np.stack(columns, axis=1)


def endpoint_correct_positions(positions, start, end, s_values):
    positions = np.asarray(positions, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    correction = linear_baseline(positions[0] - start, positions[-1] - end, s_values)
    corrected = positions - correction
    corrected[0] = start
    corrected[-1] = end
    return corrected


def smooth_positions_for_jerk(
    positions,
    start,
    end,
    s_values,
    method="none",
    window_length=41,
    polyorder=5,
):
    method = str(method).lower()

    if method in ("none", "off", "false"):
        return np.asarray(positions, dtype=float).copy()

    if method not in ("savgol", "savgol_position", "position"):
        raise ValueError(f"Unsupported jerk_smoothing={method!r}")

    smoothed = smooth_positions(
        positions,
        window_length=window_length,
        polyorder=polyorder,
    )
    return endpoint_correct_positions(smoothed, start, end, s_values)


def joint4_horizontal_rad(joint2_rad, joint3_rad, horizontal_wrist_pitch_offset_rad):
    return -float(joint2_rad) - float(joint3_rad) + float(horizontal_wrist_pitch_offset_rad)


def horizontal_offset_from_level_end(desired_end_rad):
    desired_end_rad = np.asarray(desired_end_rad, dtype=float)
    return float(desired_end_rad[2] + desired_end_rad[0] + desired_end_rad[1])


def default_horizontal_wrist_pitch_offset_rad():
    from ninibot.task_params import CYLINDRICAL_VISION_PARAMS

    return math.radians(float(CYLINDRICAL_VISION_PARAMS["horizontal_wrist_pitch_offset_deg"]))


def default_cylindrical_kinematics():
    from arm_controller.kinematics.cylindrical import CylindricalArmKinematics
    from ninibot.task_params import CYLINDRICAL_VISION_PARAMS

    params = dict(CYLINDRICAL_VISION_PARAMS)
    return CylindricalArmKinematics(
        l1=params["l1"],
        l2=params["l2"],
        horizontal_tool_offset=params["horizontal_tool_offset"],
        z_tool_offset=params["z_tool_offset"],
        theta_joint_sign=params["theta_joint_sign"],
        theta_joint_offset_deg=params["theta_joint_offset_deg"],
        base_min_deg=params["base_min_deg"],
        base_max_deg=params["base_max_deg"],
        shoulder_min_deg=params["shoulder_min_deg"],
        shoulder_max_deg=params["shoulder_max_deg"],
        elbow_min_deg=params["elbow_min_deg"],
        elbow_max_deg=params["elbow_max_deg"],
        wrist_flex_min_deg=params["wrist_flex_min_deg"],
        wrist_flex_max_deg=params["wrist_flex_max_deg"],
        horizontal_wrist_pitch_offset_deg=params["horizontal_wrist_pitch_offset_deg"],
    )


def offset_endpoint_radius_z(endpoint_rad, radius_offset_m=0.0, z_offset_m=0.0):
    radius_offset = float(radius_offset_m)
    z_offset = float(z_offset_m)
    endpoint = np.asarray(endpoint_rad, dtype=float).copy()

    if math.isclose(radius_offset, 0.0) and math.isclose(z_offset, 0.0):
        return endpoint, {
            "enabled": False,
            "radius_offset_m": radius_offset,
            "z_offset_m": z_offset,
        }

    kinematics = default_cylindrical_kinematics()
    joint2_deg = math.degrees(float(endpoint[0]))
    joint3_deg = math.degrees(float(endpoint[1]))
    original_pose = kinematics.cylindrical_from_joint_degs(
        0.0,
        joint2_deg,
        joint3_deg,
    )
    target_radius = original_pose.radius + radius_offset
    target_z = original_pose.z + z_offset
    joint23 = kinematics.joint23_ik(target_radius, target_z)

    if joint23 is None:
        raise ValueError(
            f"Adjusted endpoint is unreachable: radius={target_radius:.3f}, "
            f"z={target_z:.3f}, error={kinematics.last_error}"
        )

    adjusted_joint2_deg, adjusted_joint3_deg = joint23
    adjusted_joint4_deg = kinematics.wrist_flex_deg(
        adjusted_joint2_deg,
        adjusted_joint3_deg,
    )

    if not kinematics.in_range(
        adjusted_joint4_deg,
        kinematics.wrist_flex_min_deg,
        kinematics.wrist_flex_max_deg,
    ):
        raise ValueError(
            f"Adjusted endpoint wrist flex out of range: {adjusted_joint4_deg:.1f} deg"
        )

    endpoint[0] = math.radians(adjusted_joint2_deg)
    endpoint[1] = math.radians(adjusted_joint3_deg)

    return endpoint, {
        "enabled": True,
        "radius_offset_m": radius_offset,
        "z_offset_m": z_offset,
        "original_radius_m": float(original_pose.radius),
        "original_z_m": float(original_pose.z),
        "adjusted_radius_m": float(target_radius),
        "adjusted_z_m": float(target_z),
        "adjusted_joint2_deg": float(adjusted_joint2_deg),
        "adjusted_joint3_deg": float(adjusted_joint3_deg),
        "adjusted_horizontal_joint4_deg": float(adjusted_joint4_deg),
    }


def apply_horizontal_offset_joint4(
    positions,
    start_rad,
    desired_end_rad,
    s_values,
    horizontal_wrist_pitch_offset_rad=None,
):
    positions = np.asarray(positions, dtype=float).copy()
    start = np.asarray(start_rad, dtype=float)
    end = np.asarray(desired_end_rad, dtype=float)
    s_values = np.asarray(s_values, dtype=float)

    if horizontal_wrist_pitch_offset_rad is None:
        horizontal_wrist_pitch_offset_rad = horizontal_offset_from_level_end(end)

    start_horizontal = joint4_horizontal_rad(
        start[0],
        start[1],
        horizontal_wrist_pitch_offset_rad,
    )
    start_offset = float(start[2] - start_horizontal)

    for index in range(len(positions)):
        horizontal = joint4_horizontal_rad(
            positions[index, 0],
            positions[index, 1],
            horizontal_wrist_pitch_offset_rad,
        )
        positions[index, 2] = horizontal + start_offset * (1.0 - smootherstep(s_values[index]))

    return positions


def zero_endpoint_delta(delta, s_values):
    delta = np.asarray(delta, dtype=float)
    return delta - linear_baseline(delta[0], delta[-1], s_values)


def flat_lock_weights(
    reference_positions,
    times,
    velocity_threshold_deg_s=2.0,
    min_duration_sec=0.35,
    flat_weight=20.0,
):
    if reference_positions is None or times is None:
        return None

    reference_positions = np.asarray(reference_positions, dtype=float)
    times = np.asarray(times, dtype=float)
    flat_weight = float(flat_weight)

    if flat_weight <= 1.0 or len(times) != len(reference_positions):
        return None

    if len(times) < 3:
        return None

    dt = float(np.median(np.diff(times)))
    if dt <= 0.0:
        return None

    min_count = max(2, int(round(float(min_duration_sec) / dt)))
    velocity_threshold = math.radians(max(0.0, float(velocity_threshold_deg_s)))
    velocity = finite_difference(reference_positions, times)
    weights = np.ones_like(reference_positions, dtype=float)

    for joint_index in range(reference_positions.shape[1]):
        flat = np.abs(velocity[:, joint_index]) <= velocity_threshold
        index = 0

        while index < len(flat):
            if not flat[index]:
                index += 1
                continue

            end = index
            while end < len(flat) and flat[end]:
                end += 1

            if end - index >= min_count:
                weights[index:end, joint_index] = flat_weight

            index = end

    return weights


def smooth_delta_shape(
    delta,
    s_values,
    method="spline",
    smooth_window=15,
    smooth_polyorder=3,
    spline_smoothing=0.0003,
    flat_reference_positions=None,
    times=None,
    flat_velocity_threshold_deg_s=2.0,
    flat_min_duration_sec=0.35,
    flat_weight=20.0,
):
    method = str(method).lower()
    delta = zero_endpoint_delta(delta, s_values)

    if method in ("none", "raw"):
        return delta

    if method in ("savgol", "savgol_position", "savgol_delta"):
        return zero_endpoint_delta(
            smooth_positions(
                delta,
                window_length=smooth_window,
                polyorder=smooth_polyorder,
            ),
            s_values,
        )

    if method not in ("spline", "bspline", "b_spline"):
        raise ValueError(f"Unsupported path_smoothing={method!r}")

    try:
        from scipy.interpolate import UnivariateSpline
    except Exception:
        return zero_endpoint_delta(
            smooth_positions(
                delta,
                window_length=smooth_window,
                polyorder=smooth_polyorder,
            ),
            s_values,
        )

    s_values = np.asarray(s_values, dtype=float)
    weight_matrix = flat_lock_weights(
        flat_reference_positions,
        times,
        velocity_threshold_deg_s=flat_velocity_threshold_deg_s,
        min_duration_sec=flat_min_duration_sec,
        flat_weight=flat_weight,
    )
    columns = []

    for joint_index in range(delta.shape[1]):
        values = delta[:, joint_index]
        variance = max(float(np.var(values)), 1e-12)
        smoothing = max(0.0, float(spline_smoothing)) * len(s_values) * variance
        weights = None

        if weight_matrix is not None:
            weights = weight_matrix[:, joint_index]
            smoothing *= float(np.mean(weights * weights))

        spline = UnivariateSpline(
            s_values,
            values,
            w=weights,
            k=min(3, len(s_values) - 1),
            s=smoothing,
        )
        columns.append(spline(s_values))

    return zero_endpoint_delta(np.stack(columns, axis=1), s_values)


def make_processed_template(
    raw_times,
    raw_positions,
    sample_rate_hz=100.0,
    duration_scale=1.0,
    path_smoothing="spline",
    joint4_mode="horizontal_offset",
    endpoint_radius_offset_m=0.0,
    endpoint_z_offset_m=0.0,
    time_profile="linear",
    jerk_smoothing="none",
    smooth_window=15,
    smooth_polyorder=3,
    jerk_smooth_window=41,
    jerk_smooth_polyorder=5,
    spline_smoothing=0.0003,
    flat_velocity_threshold_deg_s=2.0,
    flat_min_duration_sec=0.35,
    flat_weight=20.0,
    shape_scale=1.0,
):
    joint4_mode = str(joint4_mode).lower()
    time_profile = str(time_profile).lower()
    jerk_smoothing = str(jerk_smoothing).lower()
    duration_scale = float(duration_scale)

    if joint4_mode not in ("horizontal_offset", "recorded"):
        raise ValueError(f"Unsupported joint4_mode={joint4_mode!r}")

    if duration_scale <= 0.0:
        raise ValueError("duration_scale must be positive")

    uniform_times, uniform_positions = resample_uniform(
        raw_times,
        raw_positions,
        sample_rate_hz=sample_rate_hz,
    )

    duration = float(uniform_times[-1])
    s_values = uniform_times / max(duration, 1e-9)
    raw_start = uniform_positions[0]
    raw_end = uniform_positions[-1]
    processed_start = raw_start.copy()
    processed_end = raw_end.copy()
    horizontal_wrist_pitch_offset_rad = default_horizontal_wrist_pitch_offset_rad()
    endpoint_adjustment = None
    processed_end, endpoint_adjustment = offset_endpoint_radius_z(
        processed_end,
        radius_offset_m=endpoint_radius_offset_m,
        z_offset_m=endpoint_z_offset_m,
    )

    if joint4_mode == "horizontal_offset":
        processed_end[2] = joint4_horizontal_rad(
            processed_end[0],
            processed_end[1],
            horizontal_wrist_pitch_offset_rad,
        )

    baseline = linear_baseline(processed_start, processed_end, s_values)
    raw_delta = uniform_positions - baseline

    if joint4_mode == "horizontal_offset":
        raw_delta[:, 2] = 0.0

    delta = smooth_delta_shape(
        raw_delta,
        s_values,
        method=path_smoothing,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        spline_smoothing=spline_smoothing,
        flat_reference_positions=uniform_positions,
        times=uniform_times,
        flat_velocity_threshold_deg_s=flat_velocity_threshold_deg_s,
        flat_min_duration_sec=flat_min_duration_sec,
        flat_weight=flat_weight,
    ) * float(shape_scale)
    retargeted_template = baseline + delta

    progress_values = time_profile_values(s_values, time_profile)

    if not np.allclose(progress_values, s_values):
        retargeted_template = resample_positions_by_progress(
            retargeted_template,
            s_values,
            progress_values,
        )
        retargeted_template = endpoint_correct_positions(
            retargeted_template,
            processed_start,
            processed_end,
            s_values,
        )

    if not math.isclose(duration_scale, 1.0):
        output_duration = duration * duration_scale
        output_count = max(2, int(math.ceil(output_duration * float(sample_rate_hz))) + 1)
        output_times = np.linspace(0.0, output_duration, output_count)
        output_s_values = output_times / max(output_duration, 1e-9)
        retargeted_template = resample_positions_by_progress(
            retargeted_template,
            s_values,
            output_s_values,
        )
        uniform_times = output_times
        duration = output_duration
        s_values = output_s_values
        baseline = linear_baseline(processed_start, processed_end, s_values)

    retargeted_template = smooth_positions_for_jerk(
        retargeted_template,
        processed_start,
        processed_end,
        s_values,
        method=jerk_smoothing,
        window_length=jerk_smooth_window,
        polyorder=jerk_smooth_polyorder,
    )

    if joint4_mode == "horizontal_offset":
        retargeted_template = apply_horizontal_offset_joint4(
            retargeted_template,
            processed_start,
            processed_end,
            s_values,
            horizontal_wrist_pitch_offset_rad=horizontal_wrist_pitch_offset_rad,
        )

    delta = zero_endpoint_delta(retargeted_template - baseline, s_values)
    retargeted_template = baseline + delta

    if joint4_mode == "horizontal_offset":
        delta[:, 2] = 0.0
        retargeted_template = baseline + delta
        retargeted_template = apply_horizontal_offset_joint4(
            retargeted_template,
            processed_start,
            processed_end,
            s_values,
            horizontal_wrist_pitch_offset_rad=horizontal_wrist_pitch_offset_rad,
        )

    velocity = finite_difference(retargeted_template, uniform_times)
    acceleration = finite_difference(velocity, uniform_times)
    jerk = finite_difference(acceleration, uniform_times)

    samples = []
    for index, t_sec in enumerate(uniform_times):
        row = {
            "t_sec": float(t_sec),
            "s": float(s_values[index]),
        }

        for joint_index, field in enumerate(RAD_FIELDS):
            row[field] = float(retargeted_template[index, joint_index])

        for joint_index, field in enumerate(DEG_FIELDS):
            row[field] = math.degrees(float(retargeted_template[index, joint_index]))

        for joint_index, field in enumerate(DELTA_FIELDS):
            row[field] = float(delta[index, joint_index])

        for joint_index, joint_name in enumerate(JOINT_NAMES):
            row[f"joint{joint_name}_vel_rad_s"] = float(velocity[index, joint_index])
            row[f"joint{joint_name}_acc_rad_s2"] = float(acceleration[index, joint_index])
            row[f"joint{joint_name}_jerk_rad_s3"] = float(jerk[index, joint_index])

        samples.append(row)

    max_abs_velocity = np.max(np.abs(velocity), axis=0).tolist()
    max_abs_acceleration = np.max(np.abs(acceleration), axis=0).tolist()
    max_abs_jerk = np.max(np.abs(jerk), axis=0).tolist()

    return {
        "schema": "ninibot.fine_align_prep_trajectory.v1",
        "joint_names": list(JOINT_NAMES),
        "duration_sec": duration,
        "sample_rate_hz": float(sample_rate_hz),
        "duration_scale": duration_scale,
        "raw_start_rad": raw_start.tolist(),
        "raw_end_rad": raw_end.tolist(),
        "processed_start_rad": processed_start.tolist(),
        "processed_end_rad": processed_end.tolist(),
        "endpoint_adjustment": endpoint_adjustment,
        "endpoint_radius_offset_m": float(endpoint_radius_offset_m),
        "endpoint_z_offset_m": float(endpoint_z_offset_m),
        "path_smoothing": str(path_smoothing),
        "joint4_mode": joint4_mode,
        "time_profile": time_profile,
        "jerk_smoothing": jerk_smoothing,
        "joint4_offset_profile": "horizontal_plus_decaying_offset",
        "horizontal_wrist_pitch_offset_rad": float(horizontal_wrist_pitch_offset_rad),
        "jerk_smooth_window": int(jerk_smooth_window),
        "jerk_smooth_polyorder": int(jerk_smooth_polyorder),
        "spline_smoothing": float(spline_smoothing),
        "flat_velocity_threshold_deg_s": float(flat_velocity_threshold_deg_s),
        "flat_min_duration_sec": float(flat_min_duration_sec),
        "flat_weight": float(flat_weight),
        "shape_scale": float(shape_scale),
        "max_abs_velocity_rad_s": max_abs_velocity,
        "max_abs_acceleration_rad_s2": max_abs_acceleration,
        "max_abs_jerk_rad_s3": max_abs_jerk,
        "samples": samples,
    }


def load_processed_template(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    if data.get("schema") != "ninibot.fine_align_prep_trajectory.v1":
        raise ValueError(f"Unsupported fine-align-prep trajectory schema in {path}")

    if data.get("joint_names") != list(JOINT_NAMES):
        raise ValueError(f"Expected joint_names={JOINT_NAMES}, got {data.get('joint_names')}")

    return data


def template_arrays(template):
    samples = template["samples"]
    s_values = np.asarray([float(sample["s"]) for sample in samples], dtype=float)
    delta = np.asarray(
        [
            [float(sample[field]) for field in DELTA_FIELDS]
            for sample in samples
        ],
        dtype=float,
    )
    return s_values, delta


def interpolate_delta_shape(template, s):
    s_values, delta = template_arrays(template)
    s = max(0.0, min(1.0, float(s)))
    columns = [
        float(np.interp(s, s_values, delta[:, index]))
        for index in range(delta.shape[1])
    ]
    return np.asarray(columns, dtype=float)


def retarget_sample(template, start_rad, desired_end_rad, s, shape_scale=1.0):
    start = np.asarray(start_rad, dtype=float)
    end = np.asarray(desired_end_rad, dtype=float)
    s = max(0.0, min(1.0, float(s)))
    baseline = start + s * (end - start)
    position = baseline + interpolate_delta_shape(template, s) * float(shape_scale)

    if template.get("joint4_mode", "recorded") == "horizontal_offset":
        position = apply_horizontal_offset_joint4(
            np.asarray([position], dtype=float),
            start,
            end,
            np.asarray([s], dtype=float),
        )[0]

    return position


def retarget_trajectory(
    template,
    start_rad,
    desired_end_rad,
    duration_sec=None,
    sample_rate_hz=None,
    shape_scale=1.0,
):
    if duration_sec is None or duration_sec <= 0.0:
        duration_sec = float(template["duration_sec"])

    if sample_rate_hz is None or sample_rate_hz <= 0.0:
        sample_rate_hz = float(template["sample_rate_hz"])

    count = max(2, int(math.ceil(float(duration_sec) * float(sample_rate_hz))) + 1)
    times = np.linspace(0.0, float(duration_sec), count)
    s_values = times / max(float(duration_sec), 1e-9)
    positions = np.asarray(
        [
            retarget_sample(
                template,
                start_rad,
                desired_end_rad,
                s,
                shape_scale=shape_scale,
            )
            for s in s_values
        ],
        dtype=float,
    )
    return times, positions


def retarget_raw_shape(
    raw_times,
    raw_positions,
    start_rad,
    desired_end_rad,
    sample_rate_hz=100.0,
    shape_scale=1.0,
):
    times, positions = resample_uniform(
        raw_times,
        raw_positions,
        sample_rate_hz=sample_rate_hz,
    )
    duration = float(times[-1])
    s_values = times / max(duration, 1e-9)
    raw_baseline = linear_baseline(positions[0], positions[-1], s_values)
    delta = (positions - raw_baseline) * float(shape_scale)
    delta = zero_endpoint_delta(delta, s_values)
    target_baseline = linear_baseline(start_rad, desired_end_rad, s_values)
    return times, target_baseline + delta


def write_processed_json(path, template):
    path = Path(path)
    ensure_parent(path)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(template, stream, indent=2)
        stream.write("\n")


def write_processed_csv(path, template):
    path = Path(path)
    ensure_parent(path)
    samples = template["samples"]

    if not samples:
        raise ValueError("Cannot write an empty processed trajectory")

    fieldnames = list(samples[0].keys())

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def write_latest_copy(source_path, latest_path):
    source_path = Path(source_path)
    latest_path = Path(latest_path)
    ensure_parent(latest_path)
    shutil.copyfile(source_path, latest_path)


def write_raw_csv(path, rows):
    path = Path(path)
    ensure_parent(path)

    fieldnames = [
        "t_sec",
        "stamp_sec",
        "joint2_rad",
        "joint3_rad",
        "joint4_rad",
        "joint2_deg",
        "joint3_deg",
        "joint4_deg",
    ]

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata_json(path, metadata):
    path = Path(path)
    ensure_parent(path)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


def plot_joint_trajectory(path, times, positions, figure_title):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ninibot_matplotlib")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    times = np.asarray(times, dtype=float)
    pos = np.asarray(positions, dtype=float)
    vel = finite_difference(pos, times)
    acc = finite_difference(vel, times)
    jerk = finite_difference(acc, times)

    series = [
        ("position_deg", np.degrees(pos)),
        ("velocity_deg_s", np.degrees(vel)),
        ("acceleration_deg_s2", np.degrees(acc)),
        ("jerk_deg_s3", np.degrees(jerk)),
    ]

    fig, axes = plt.subplots(len(series), 1, figsize=(11, 12), sharex=True)

    for axis, (ylabel, values) in zip(axes, series):
        for index, joint in enumerate(JOINT_NAMES):
            axis.plot(times, values[:, index], label=f"joint{joint}")

        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(figure_title)
    fig.tight_layout()
    ensure_parent(path)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_trajectory_comparison(
    path,
    raw_times,
    raw_positions,
    smooth_times,
    smooth_positions,
    figure_title,
):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ninibot_matplotlib")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    raw_times = np.asarray(raw_times, dtype=float)
    raw_pos = np.asarray(raw_positions, dtype=float)
    smooth_times = np.asarray(smooth_times, dtype=float)
    smooth_pos = np.asarray(smooth_positions, dtype=float)

    raw_vel = finite_difference(raw_pos, raw_times)
    raw_acc = finite_difference(raw_vel, raw_times)
    raw_jerk = finite_difference(raw_acc, raw_times)
    smooth_vel = finite_difference(smooth_pos, smooth_times)
    smooth_acc = finite_difference(smooth_vel, smooth_times)
    smooth_jerk = finite_difference(smooth_acc, smooth_times)

    rows = [
        ("position_deg", np.degrees(raw_pos), np.degrees(smooth_pos)),
        ("velocity_deg_s", np.degrees(raw_vel), np.degrees(smooth_vel)),
        ("acceleration_deg_s2", np.degrees(raw_acc), np.degrees(smooth_acc)),
        ("jerk_deg_s3", np.degrees(raw_jerk), np.degrees(smooth_jerk)),
    ]

    fig, axes = plt.subplots(
        len(rows),
        len(JOINT_NAMES),
        figsize=(15, 12),
        sharex=True,
    )

    for row_index, (ylabel, raw_values, smooth_values) in enumerate(rows):
        for joint_index, joint in enumerate(JOINT_NAMES):
            axis = axes[row_index, joint_index]
            axis.plot(
                raw_times,
                raw_values[:, joint_index],
                color="0.45",
                linestyle="--",
                linewidth=1.1,
                label="collected",
            )
            axis.plot(
                smooth_times,
                smooth_values[:, joint_index],
                color="tab:blue",
                linewidth=1.4,
                label="processed",
            )
            axis.grid(True, alpha=0.3)

            if row_index == 0:
                axis.set_title(f"joint{joint}")

            if joint_index == 0:
                axis.set_ylabel(ylabel)

            if row_index == len(rows) - 1:
                axis.set_xlabel("time (s)")

            if row_index == 0 and joint_index == len(JOINT_NAMES) - 1:
                axis.legend(loc="best")

    fig.suptitle(figure_title)
    fig.tight_layout()
    ensure_parent(path)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_processed(path, template):
    samples = template["samples"]
    times = np.asarray([sample["t_sec"] for sample in samples], dtype=float)
    positions = np.asarray(
        [
            [sample[field] for field in RAD_FIELDS]
            for sample in samples
        ],
        dtype=float,
    )
    plot_joint_trajectory(
        path,
        times,
        positions,
        "Fine-align-prep processed template trajectory",
    )
