from dataclasses import dataclass
from dataclasses import fields
import math

import numpy as np


@dataclass
class MoveCupConfig:
    command_topic: str
    status_topic: str
    target_topic: str
    real_joint_states_topic: str
    control_rate: float
    joint_goal_tolerance: float
    settle_cycles: int
    grip_settle_sec: float
    release_settle_sec: float
    release_pan_offset_deg: float
    post_release_clearance_retract_m: float
    post_release_clearance_duration_sec: float
    target_check_joint_count: int
    allow_timeout_advance: bool
    phase_timeout_sec: float
    timeout_advance_tolerance_deg: float
    lift_z_offset: float
    lift_duration_sec: float
    pre_pan_radius: float
    pre_pan_retract_duration_sec: float
    post_pan_extend_duration_sec: float
    horizontal_move_duration_sec: float
    lower_duration_sec: float
    use_dynamic_carry_z: bool
    carry_z: float
    target_theta_deg: float
    target_radius: float
    target_z: float
    l1: float
    l2: float
    horizontal_tool_offset: float
    z_tool_offset: float
    theta_joint_sign: float
    theta_joint_offset_deg: float
    base_min_deg: float
    base_max_deg: float
    shoulder_min_deg: float
    shoulder_max_deg: float
    elbow_min_deg: float
    elbow_max_deg: float
    wrist_flex_min_deg: float
    wrist_flex_max_deg: float
    horizontal_wrist_pitch_offset_deg: float
    wrist_roll_deg: float
    hold_gripper_percent: float
    release_gripper_percent: float
    return_joint2_deg: float
    return_joint3_deg: float
    camera_device: str
    camera_index: int
    frame_width: int
    frame_height: int
    calibration_path: str
    aruco_dictionary: str
    target_marker_id: int
    marker_length: float
    base_frame: str
    camera_frame: str
    tf_timeout_sec: float
    target_marker_offset_xyz: object
    target_base_topic: str
    target_lock_samples: int
    target_lock_max_std_m: float
    target_radius_margin_m: float
    target_search_timeout_sec: float
    target_search_theta_min_deg: float
    target_search_theta_max_deg: float
    target_search_step_deg: float
    search_joint2_deg: float
    search_joint3_deg: float
    search_joint4_deg: float
    search_prep_min_duration_sec: float
    search_prep_head_down_duration_sec: float
    search_prep_duration_scale: float
    search_prep_trajectory_samples: int
    search_prep_max_vel_deg: list
    search_prep_max_acc_deg: list
    return_search_prep_final_offset_deg: float
    aruco_debug_window: bool
    aruco_debug_window_name: str

    @classmethod
    def from_node(cls, node):
        def value(name):
            return node.get_parameter(name).value

        return cls(
            command_topic=value('command_topic'),
            status_topic=value('status_topic'),
            target_topic=value('target_topic'),
            real_joint_states_topic=value('real_joint_states_topic'),
            control_rate=float(value('control_rate')),
            joint_goal_tolerance=math.radians(
                float(value('joint_goal_tolerance_deg'))
            ),
            settle_cycles=int(value('settle_cycles')),
            grip_settle_sec=float(value('grip_settle_sec')),
            release_settle_sec=float(value('release_settle_sec')),
            release_pan_offset_deg=float(value('release_pan_offset_deg')),
            post_release_clearance_retract_m=max(
                0.0,
                float(value('post_release_clearance_retract_m')),
            ),
            post_release_clearance_duration_sec=max(
                0.0,
                float(value('post_release_clearance_duration_sec')),
            ),
            target_check_joint_count=max(
                1,
                min(4, int(value('target_check_joint_count'))),
            ),
            allow_timeout_advance=bool(value('allow_timeout_advance')),
            phase_timeout_sec=float(value('phase_timeout_sec')),
            timeout_advance_tolerance_deg=float(
                value('timeout_advance_tolerance_deg')
            ),
            lift_z_offset=float(value('lift_z_offset')),
            lift_duration_sec=float(value('lift_duration_sec')),
            pre_pan_radius=float(value('pre_pan_radius')),
            pre_pan_retract_duration_sec=float(
                value('pre_pan_retract_duration_sec')
            ),
            post_pan_extend_duration_sec=float(
                value('post_pan_extend_duration_sec')
            ),
            horizontal_move_duration_sec=float(
                value('horizontal_move_duration_sec')
            ),
            lower_duration_sec=float(value('lower_duration_sec')),
            use_dynamic_carry_z=bool(value('use_dynamic_carry_z')),
            carry_z=float(value('carry_z')),
            target_theta_deg=float(value('target_theta_deg')),
            target_radius=float(value('target_radius')),
            target_z=float(value('target_z')),
            l1=float(value('l1')),
            l2=float(value('l2')),
            horizontal_tool_offset=float(value('horizontal_tool_offset')),
            z_tool_offset=float(value('z_tool_offset')),
            theta_joint_sign=float(value('theta_joint_sign')),
            theta_joint_offset_deg=float(value('theta_joint_offset_deg')),
            base_min_deg=float(value('base_min_deg')),
            base_max_deg=float(value('base_max_deg')),
            shoulder_min_deg=float(value('shoulder_min_deg')),
            shoulder_max_deg=float(value('shoulder_max_deg')),
            elbow_min_deg=float(value('elbow_min_deg')),
            elbow_max_deg=float(value('elbow_max_deg')),
            wrist_flex_min_deg=float(value('wrist_flex_min_deg')),
            wrist_flex_max_deg=float(value('wrist_flex_max_deg')),
            horizontal_wrist_pitch_offset_deg=float(
                value('horizontal_wrist_pitch_offset_deg')
            ),
            wrist_roll_deg=float(value('wrist_roll_deg')),
            hold_gripper_percent=float(value('hold_gripper_percent')),
            release_gripper_percent=float(value('release_gripper_percent')),
            return_joint2_deg=float(value('return_joint2_deg')),
            return_joint3_deg=float(value('return_joint3_deg')),
            camera_device=str(value('camera_device')),
            camera_index=int(value('camera_index')),
            frame_width=int(value('frame_width')),
            frame_height=int(value('frame_height')),
            calibration_path=value('calibration_path'),
            aruco_dictionary=value('aruco_dictionary'),
            target_marker_id=int(value('target_marker_id')),
            marker_length=float(value('marker_length')),
            base_frame=value('base_frame'),
            camera_frame=value('camera_frame'),
            tf_timeout_sec=float(value('tf_timeout_sec')),
            target_marker_offset_xyz=np.asarray(
                [
                    float(value('target_x_offset')),
                    float(value('target_y_offset')),
                    float(value('target_z_offset')),
                ],
                dtype=float,
            ),
            target_base_topic=value('target_base_topic'),
            target_lock_samples=max(1, int(value('target_lock_samples'))),
            target_lock_max_std_m=max(
                0.0,
                float(value('target_lock_max_std_m')),
            ),
            target_radius_margin_m=max(
                0.0,
                float(value('target_radius_margin_m')),
            ),
            target_search_timeout_sec=float(value('target_search_timeout_sec')),
            target_search_theta_min_deg=float(
                value('target_search_theta_min_deg')
            ),
            target_search_theta_max_deg=float(
                value('target_search_theta_max_deg')
            ),
            target_search_step_deg=float(value('target_search_step_deg')),
            search_joint2_deg=float(value('search_joint2_deg')),
            search_joint3_deg=float(value('search_joint3_deg')),
            search_joint4_deg=float(value('search_joint4_deg')),
            search_prep_min_duration_sec=max(
                0.1,
                float(value('search_prep_min_duration_sec')),
            ),
            search_prep_head_down_duration_sec=max(
                0.1,
                float(value('search_prep_head_down_duration_sec')),
            ),
            search_prep_duration_scale=max(
                1.0,
                float(value('search_prep_duration_scale')),
            ),
            search_prep_trajectory_samples=max(
                16,
                int(value('search_prep_trajectory_samples')),
            ),
            search_prep_max_vel_deg=[
                float(value('search_prep_j1_max_vel_deg')),
                float(value('search_prep_j2_max_vel_deg')),
                float(value('search_prep_j3_max_vel_deg')),
                float(value('search_prep_j4_max_vel_deg')),
            ],
            search_prep_max_acc_deg=[
                float(value('search_prep_j1_max_acc_deg')),
                float(value('search_prep_j2_max_acc_deg')),
                float(value('search_prep_j3_max_acc_deg')),
                float(value('search_prep_j4_max_acc_deg')),
            ],
            return_search_prep_final_offset_deg=float(
                value('return_search_prep_final_offset_deg')
            ),
            aruco_debug_window=bool(value('aruco_debug_window')),
            aruco_debug_window_name=value('aruco_debug_window_name'),
        )

    def apply_to_node(self, node):
        for field in fields(self):
            setattr(node, field.name, getattr(self, field.name))
