from ninibot.skills.press_button.local_ik import LocalIkConfig
from ninibot.skills.press_button.local_ik import PressButtonLocalIk
from ninibot.skills.press_button.params import PrepTrajectoryConfig
from ninibot.skills.press_button.params import PressMotionConfig
from ninibot.skills.press_button.params import SearchScanConfig
from ninibot.skills.press_button.prep_trajectory import PrepTrajectoryPlanner
from ninibot.skills.press_button.press_motion import PressMotionPlanner
from ninibot.skills.press_button.search_scan import SearchScanner


def create_local_ik(config, log_progress, warn):
    return PressButtonLocalIk(
        LocalIkConfig(
            urdf_path=config.urdf_path,
            base_frame=config.base_frame,
            gripper_frame=config.gripper_frame,
            prep_joint1_deg=config.prep_joint1_deg,
            prep_joint2_deg=config.prep_joint2_deg,
            prep_joint3_deg=config.prep_joint3_deg,
            prep_joint4_deg=config.prep_joint4_deg,
            search_pan_min_deg=config.search_pan_min_deg,
            search_pan_max_deg=config.search_pan_max_deg,
            align_wrist_pitch_deg=config.align_wrist_pitch_deg,
            ik_joint4_min_deg=config.ik_joint4_min_deg,
            ik_joint4_max_deg=config.ik_joint4_max_deg,
            fixed_joint5_rad=config.fixed_joint5_rad,
            joint6_rad=config.joint6_rad,
            ik_position_tolerance=config.ik_position_tolerance,
            fine_align_max_joint_step_rad=config.fine_align_max_joint_step_rad,
            fine_align_lock_joint4=config.fine_align_lock_joint4,
            fine_align_max_step=config.fine_align_max_step,
        ),
        log_progress=log_progress,
        warn=warn,
    )


def create_prep_planner(config, joint4_lower, joint4_upper):
    return PrepTrajectoryPlanner(
        PrepTrajectoryConfig(
            prep_joint1_deg=config.prep_joint1_deg,
            prep_joint2_deg=config.prep_joint2_deg,
            prep_joint3_deg=config.prep_joint3_deg,
            prep_joint4_deg=config.prep_joint4_deg,
            duration_sec=config.prep_trajectory_duration_sec,
            j2_transition_sec=config.prep_j2_transition_sec,
            search_pan_min_deg=config.search_pan_min_deg,
            search_pan_max_deg=config.search_pan_max_deg,
            joint4_lower_rad=float(joint4_lower[0]),
            joint4_upper_rad=float(joint4_upper[0]),
            joint6_rad=config.joint6_rad,
        )
    )


def create_search_scanner(config):
    return SearchScanner(
        SearchScanConfig(
            prep_joint1_deg=config.prep_joint1_deg,
            prep_joint2_deg=config.prep_joint2_deg,
            prep_joint3_deg=config.prep_joint3_deg,
            prep_joint4_deg=config.prep_joint4_deg,
            search_pan_min_deg=config.search_pan_min_deg,
            search_pan_max_deg=config.search_pan_max_deg,
            search_pan_speed_deg_s=config.search_pan_speed_deg_s,
            joint6_rad=config.joint6_rad,
        )
    )


def create_press_motion_planner(
    config,
    ik_lower_bounds,
    ik_upper_bounds,
    joint4_lower,
    joint4_upper,
):
    return PressMotionPlanner(
        PressMotionConfig(
            press_joint2_delta_deg=config.press_joint2_delta_deg,
            press_joint3_delta_deg=config.press_joint3_delta_deg,
            press_joint4_delta_deg=config.press_joint4_delta_deg,
            joint6_rad=config.joint6_rad,
            ik_lower_bounds=tuple(float(value) for value in ik_lower_bounds),
            ik_upper_bounds=tuple(float(value) for value in ik_upper_bounds),
            joint4_lower_rad=float(joint4_lower[0]),
            joint4_upper_rad=float(joint4_upper[0]),
        )
    )
