import math
from dataclasses import dataclass
from dataclasses import fields

import numpy as np

from ninibot.skills.press_button.math_utils import clamp
from ninibot.skills.press_button.math_utils import gripper_percent_to_rad


@dataclass(frozen=True)
class PressButtonConfig:
    target_topic: str
    real_joint_states_topic: str
    command_topic: str
    status_topic: str
    control_rate: float
    settle_cycles: int
    joint_goal_tolerance: float
    camera_device: str
    camera_index: int
    camera_source: object
    frame_width: int
    frame_height: int
    calibration_path: str
    aruco_dictionary: str
    target_marker_id: int
    marker_length: float
    base_frame: str
    camera_frame: str
    gripper_frame: str
    urdf_path: str
    tf_timeout_sec: float
    target_x_offset: float
    target_y_offset: float
    target_z_offset: float
    target_marker_offset_xyz: np.ndarray
    prep_joint1_deg: float
    prep_joint2_deg: float
    prep_joint3_deg: float
    prep_joint4_deg: float
    prep_trajectory_duration_sec: float
    prep_j2_transition_sec: float
    search_pan_min_deg: float
    search_pan_max_deg: float
    search_pan_speed_deg_s: float
    move_to_marker_speed_scale: float
    align_wrist_pitch_deg: float
    ik_joint4_min_deg: float
    ik_joint4_max_deg: float
    track_marker_during_move: bool
    move_goal_update_period_sec: float
    fine_align_enable: bool
    fine_align_start_distance: float
    fine_align_tolerance: float
    fine_align_max_step: float
    fine_align_speed_scale: float
    fine_align_max_joint_step_rad: float
    fine_align_lock_joint4: bool
    fine_align_use_ik_service: bool
    fine_align_ik_service_name: str
    fine_align_ik_service_timeout_sec: float
    fine_align_ik_service_fallback: bool
    fine_align_max_cycles: int
    fine_align_integral_enable: bool
    fine_align_max_command_ahead: float
    fine_align_marker_samples: int
    fine_align_sample_wait_sec: float
    fine_align_settle_cycles: int
    fine_align_move_timeout_sec: float
    fine_align_joint_goal_tolerance: float
    press_wait_sec: float
    press_joint2_delta_deg: float
    press_joint3_delta_deg: float
    press_joint4_delta_deg: float
    press_motion_speed_scale: float
    press_motion_settle_cycles: int
    press_repeat_wait_sec: float
    press_cycles: int
    fixed_joint5_rad: float
    ik_position_tolerance: float
    joint6_percent: float
    joint6_rad: float
    debug_window: bool
    debug_window_name: str
    publish_debug_image_enabled: bool
    debug_image_topic: str
    lock_wait_sec: float
    auto_start: bool

    @classmethod
    def from_values(cls, values, normalize_camera_source, resolve_urdf_path):
        camera_device = str(values["camera_device"]).strip()
        camera_index = int(values["camera_index"])
        search_pan_min_deg = float(values["search_pan_min_deg"])
        search_pan_max_deg = float(values["search_pan_max_deg"])
        if search_pan_min_deg > search_pan_max_deg:
            search_pan_min_deg, search_pan_max_deg = (
                search_pan_max_deg,
                search_pan_min_deg,
            )

        ik_joint4_min_deg = float(values["ik_joint4_min_deg"])
        ik_joint4_max_deg = float(values["ik_joint4_max_deg"])
        if ik_joint4_min_deg > ik_joint4_max_deg:
            ik_joint4_min_deg, ik_joint4_max_deg = (
                ik_joint4_max_deg,
                ik_joint4_min_deg,
            )

        joint6_percent = clamp(float(values["joint6_percent"]), 0.0, 100.0)
        joint6_rad = gripper_percent_to_rad(joint6_percent)
        target_marker_offset_xyz = np.asarray(
            [
                float(values["target_x_offset"]),
                float(values["target_y_offset"]),
                float(values["target_z_offset"]),
            ],
            dtype=float,
        )
        return cls(
            target_topic=values["target_topic"],
            real_joint_states_topic=values["real_joint_states_topic"],
            command_topic=values["command_topic"],
            status_topic=values["status_topic"],
            control_rate=float(values["control_rate"]),
            settle_cycles=int(values["settle_cycles"]),
            joint_goal_tolerance=math.radians(float(values["joint_goal_tolerance_deg"])),
            camera_device=camera_device,
            camera_index=camera_index,
            camera_source=normalize_camera_source(
                camera_device if camera_device else camera_index
            ),
            frame_width=int(values["frame_width"]),
            frame_height=int(values["frame_height"]),
            calibration_path=values["calibration_path"],
            aruco_dictionary=values["aruco_dictionary"],
            target_marker_id=int(values["target_marker_id"]),
            marker_length=float(values["marker_length"]),
            base_frame=values["base_frame"],
            camera_frame=values["camera_frame"],
            gripper_frame=values["gripper_frame"],
            urdf_path=resolve_urdf_path(values["urdf_path"]),
            tf_timeout_sec=float(values["tf_timeout_sec"]),
            target_x_offset=float(values["target_x_offset"]),
            target_y_offset=float(values["target_y_offset"]),
            target_z_offset=float(values["target_z_offset"]),
            target_marker_offset_xyz=target_marker_offset_xyz,
            prep_joint1_deg=float(values["prep_joint1_deg"]),
            prep_joint2_deg=float(values["prep_joint2_deg"]),
            prep_joint3_deg=float(values["prep_joint3_deg"]),
            prep_joint4_deg=float(values["prep_joint4_deg"]),
            prep_trajectory_duration_sec=max(
                0.0,
                float(values["prep_trajectory_duration_sec"]),
            ),
            prep_j2_transition_sec=max(0.0, float(values["prep_j2_transition_sec"])),
            search_pan_min_deg=search_pan_min_deg,
            search_pan_max_deg=search_pan_max_deg,
            search_pan_speed_deg_s=abs(float(values["search_pan_speed_deg_s"])),
            move_to_marker_speed_scale=max(
                0.0,
                float(values["move_to_marker_speed_scale"]),
            ),
            align_wrist_pitch_deg=float(values["align_wrist_pitch_deg"]),
            ik_joint4_min_deg=ik_joint4_min_deg,
            ik_joint4_max_deg=ik_joint4_max_deg,
            track_marker_during_move=bool(values["track_marker_during_move"]),
            move_goal_update_period_sec=max(
                0.0,
                float(values["move_goal_update_period_sec"]),
            ),
            fine_align_enable=bool(values["fine_align_enable"]),
            fine_align_start_distance=max(
                0.0,
                float(values["fine_align_start_distance"]),
            ),
            fine_align_tolerance=max(0.0, float(values["fine_align_tolerance"])),
            fine_align_max_step=max(0.0, float(values["fine_align_max_step"])),
            fine_align_speed_scale=max(0.0, float(values["fine_align_speed_scale"])),
            fine_align_max_joint_step_rad=math.radians(
                max(0.0, float(values["fine_align_max_joint_step_deg"]))
            ),
            fine_align_lock_joint4=bool(values["fine_align_lock_joint4"]),
            fine_align_use_ik_service=bool(values["fine_align_use_ik_service"]),
            fine_align_ik_service_name=str(values["fine_align_ik_service_name"]),
            fine_align_ik_service_timeout_sec=max(
                0.0,
                float(values["fine_align_ik_service_timeout_sec"]),
            ),
            fine_align_ik_service_fallback=bool(
                values["fine_align_ik_service_fallback"]
            ),
            fine_align_max_cycles=max(1, int(values["fine_align_max_cycles"])),
            fine_align_integral_enable=bool(values["fine_align_integral_enable"]),
            fine_align_max_command_ahead=max(
                0.0,
                float(values["fine_align_max_command_ahead"]),
            ),
            fine_align_marker_samples=max(1, int(values["fine_align_marker_samples"])),
            fine_align_sample_wait_sec=max(
                0.0,
                float(values["fine_align_sample_wait_sec"]),
            ),
            fine_align_settle_cycles=max(1, int(values["fine_align_settle_cycles"])),
            fine_align_move_timeout_sec=max(
                0.0,
                float(values["fine_align_move_timeout_sec"]),
            ),
            fine_align_joint_goal_tolerance=math.radians(
                max(0.0, float(values["fine_align_joint_goal_tolerance_deg"]))
            ),
            press_wait_sec=max(0.0, float(values["press_wait_sec"])),
            press_joint2_delta_deg=float(values["press_joint2_delta_deg"]),
            press_joint3_delta_deg=float(values["press_joint3_delta_deg"]),
            press_joint4_delta_deg=float(values["press_joint4_delta_deg"]),
            press_motion_speed_scale=max(
                0.0,
                float(values["press_motion_speed_scale"]),
            ),
            press_motion_settle_cycles=max(
                1,
                int(values["press_motion_settle_cycles"]),
            ),
            press_repeat_wait_sec=max(0.0, float(values["press_repeat_wait_sec"])),
            press_cycles=max(1, int(values["press_cycles"])),
            fixed_joint5_rad=math.radians(float(values["fixed_joint5_deg"])),
            ik_position_tolerance=float(values["ik_position_tolerance"]),
            joint6_percent=joint6_percent,
            joint6_rad=joint6_rad,
            debug_window=bool(values["debug_window"]),
            debug_window_name=values["debug_window_name"],
            publish_debug_image_enabled=bool(values["publish_debug_image"]),
            debug_image_topic=values["debug_image_topic"],
            lock_wait_sec=max(0.0, float(values["lock_wait_sec"])),
            auto_start=bool(values["auto_start"]),
        )

    def apply_to(self, target):
        for field in fields(self):
            setattr(target, field.name, getattr(self, field.name))

