# flake8: noqa
# Defaults for cup pickup, carry, and release.
#
# Units:
# - distances are meters
# - joint parameters ending in _deg are degrees
# - gripper parameters ending in _percent are percent open/closed command values

# Machine-specific /dev/v4l/by-id paths must be supplied at launch time.
# An empty value falls back to camera_index.
INNOMAKER_CAMERA_DEVICE = ""


CYLINDRICAL_VISION_PARAMS = {
    # Camera and debug output.
    "camera_device": INNOMAKER_CAMERA_DEVICE,
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "show_window": True,
    "publish_debug_images": True,
    "window_name": "Cup visual servo",
    "target_topic": "/target_joint_states",
    "real_joint_states_topic": "/real_joint_states",
    "command_topic": "/cup_servo_command",
    "auto_start": False,
    "skip_search_prep_on_start": False,

    # Detection backend. Use "hsv", "yolo", or "hybrid" (YOLO gate + HSV centroid).
    "detector_backend": "hsv",
    "yolo_model_path": "data/models/yolo/yolo11n.pt",
    "yolo_conf": 0.3,
    "yolo_class_name": "cup",
    "yolo_class_id": -1,
    "yolo_imgsz": 640,
    "yolo_device": "",
    "hybrid_roi_scale": 1.25,
    "hybrid_min_roi_padding_px": 12,
    "hybrid_hsv_yolo_min_overlap_ratio": 0.05,
    "hybrid_hsv_yolo_max_center_distance_ratio": 0.85,
    "hybrid_hsv_candidate_area_weight": 1.0,
    "hybrid_hsv_candidate_center_distance_weight": 1.25,

    # HSV color threshold for cup detection.
    "lower_h": 90,
    "lower_s": 50,
    "lower_v": 30,
    "upper_h": 130,
    "upper_s": 255,
    "upper_v": 255,
    "min_area": 500.0,
    "center_filter_alpha": 0.1,

    # Image-space visual servo targets.
    # target_center_x_ratio moves the desired cup center across the image width.
    # target_bbox_width controls final approach distance before closing gripper.
    "target_center_x_ratio": 0.55,
    # Fine-align pan correction threshold, using cup center x error in pixels.
    "center_tolerance_px": 40.0,
    # TRACKING_CENTER finish threshold, using cup center x error, not bbox width.
    "tracking_settle_tolerance_px": 40.0,
    "tracking_settle_cycles": 3,
    "target_bbox_width": 410.0,
    "bbox_tolerance_px": 12.0,
    "bbox_resume_tolerance_px": 35.0,

    # Search pan range in cup theta space. Actual base joint target is:
    # base_deg = theta_joint_sign * theta_deg + theta_joint_offset_deg.
    "start_theta_deg": 0.0,
    "theta_min_deg": -80.0,
    "theta_max_deg": 0.0,

    # Arm geometry, tool offset, and joint limits.
    "l1": 0.1159,
    "l2": 0.1350,
    "horizontal_tool_offset": 0.10,
    "z_tool_offset": -0.10,
    "theta_joint_sign": -1.0,
    "theta_joint_offset_deg": 0.0,
    "base_min_deg": -90.0,
    "base_max_deg": 90.0,
    "shoulder_min_deg": -100.0,
    "shoulder_max_deg": 100.0,
    "elbow_min_deg": -90.0,
    "elbow_max_deg": 100.0,
    "wrist_flex_min_deg": -104.0,
    "wrist_flex_max_deg": 100.0,

    # Search pose. search_joint4_deg is the camera head-down offset;
    # processed fine-align-prep ramps it back to level.
    "search_joint2_deg": -90.0,
    "search_joint3_deg": 98.0,
    "search_joint4_deg": 25.0,
    "search_prep_min_duration_sec": 3.0,
    "search_prep_head_down_duration_sec": 3.0,
    "search_prep_duration_scale": 1.15,
    "search_prep_trajectory_samples": 160,
    "search_prep_j1_max_vel_deg": 60.0,
    "search_prep_j2_max_vel_deg": 60.0,
    "search_prep_j3_max_vel_deg": 60.0,
    "search_prep_j4_max_vel_deg": 120.0,
    "search_prep_j1_max_acc_deg": 60.0,
    "search_prep_j2_max_acc_deg": 60.0,
    "search_prep_j3_max_acc_deg": 60.0,
    "search_prep_j4_max_acc_deg": 120.0,
    "fine_align_prep_trajectory_path": "data/trajectories/fine_align_prep/sessions/2026-06-15_8cm_base_raw/processed_template.json",
    "fine_align_prep_trajectory_duration_sec": 5.0,
    "fine_align_prep_shape_scale": 1.0,
    "fine_align_prep_z_offset_m": 0.02,

    # Wrist leveling calibration. Tune this if joint4 is not level when the
    # code says it should be horizontal.
    "joint_goal_tolerance_deg": 3.0,
    "horizontal_wrist_pitch_offset_deg": 5.0,
    "wrist_roll_deg": -90.0,
    "gripper_percent": 45.0,

    # Pan servo and cylindrical approach gains.
    "control_rate": 100.0,
    "theta_kp": 0.00008,
    "theta_max_step_deg": 1.08,
    "tracking_theta_max_step_deg": 0.30,
    "theta_sign": -1.0,
    "fine_align_centered_cycles": 2,
    "approach_radius_kp_m_per_px": 0.000008,
    "approach_max_radius_step_m": 0.027,

    # Lost-cup search behavior.
    "search_enabled": True,
    "search_step_deg": 0.70,
    "publish_while_lost": True,
    "tracking_lost_grace_sec": 1.5,
}


MOVE_CUP_PARAMS = {
    # Topics and timing.
    "command_topic": "/move_cup_command",
    "status_topic": "/move_cup_status",
    "target_topic": "/target_joint_states",
    "real_joint_states_topic": "/real_joint_states",
    "control_rate": 20.0,
    "joint_goal_tolerance_deg": 10.0,
    "settle_cycles": 10,
    "grip_settle_sec": 1.0,
    "release_settle_sec": 1.0,
    "release_pan_offset_deg": -9.0,
    "post_release_clearance_retract_m": 0.03,
    "post_release_clearance_duration_sec": 0.8,
    "target_check_joint_count": 4,
    "allow_timeout_advance": True,
    "phase_timeout_sec": 3.0,
    "timeout_advance_tolerance_deg": 12.0,

    # After gripping, lift only this much before moving to the endpoint.
    "lift_z_offset": 0.03,
    "lift_duration_sec": 1.2,
    "pre_pan_radius": 0.10,
    "pre_pan_retract_duration_sec": 1.5,
    "post_pan_extend_duration_sec": 1.2,
    "horizontal_move_duration_sec": 3.0,
    "lower_duration_sec": 1.2,
    "use_dynamic_carry_z": True,
    "carry_z": 0.05,

    # Move-cup endpoint. Tune these for the filter/dispenser location.
    "target_theta_deg": 0.0,
    "target_radius": 0.22,
    "target_z": 0.0,

    # Optional ArUco target scan. target_*_offset is the marker-frame offset
    # to the move-cup target frame. The marker supplies XY; release Z is always
    # carry_z - lift_z_offset. The marker is vertical; marker-frame +Z is the
    # marker normal in OpenCV pose coordinates.
    "camera_device": CYLINDRICAL_VISION_PARAMS["camera_device"],
    "camera_index": CYLINDRICAL_VISION_PARAMS["camera_index"],
    "frame_width": CYLINDRICAL_VISION_PARAMS["frame_width"],
    "frame_height": CYLINDRICAL_VISION_PARAMS["frame_height"],
    "calibration_path": "data/calibration/camera_calibration.yaml",
    "aruco_dictionary": "4X4_50",
    "target_marker_id": 1,
    "marker_length": 0.05,
    "base_frame": "base",
    "camera_frame": "camera_optical_frame",
    "tf_timeout_sec": 0.5,
    "target_base_topic": "/move_cup/target_base",
    "target_x_offset": 0.045,
    "target_y_offset": 0.0,
    "target_z_offset": 0.25,
    "target_lock_samples": 5,
    "target_lock_max_std_m": 0.01,
    "target_radius_margin_m": 0.01,
    "target_search_timeout_sec": 15.0,
    "target_search_theta_min_deg": CYLINDRICAL_VISION_PARAMS["theta_min_deg"],
    "target_search_theta_max_deg": CYLINDRICAL_VISION_PARAMS["theta_max_deg"],
    "target_search_step_deg": CYLINDRICAL_VISION_PARAMS["search_step_deg"],
    "search_joint2_deg": CYLINDRICAL_VISION_PARAMS["search_joint2_deg"],
    "search_joint3_deg": CYLINDRICAL_VISION_PARAMS["search_joint3_deg"],
    "search_joint4_deg": CYLINDRICAL_VISION_PARAMS["search_joint4_deg"],
    "search_prep_min_duration_sec": 3.0,
    "search_prep_head_down_duration_sec": 3.0,
    "search_prep_duration_scale": 1.15,
    "search_prep_trajectory_samples": 160,
    "search_prep_j1_max_vel_deg": 60.0,
    "search_prep_j2_max_vel_deg": 60.0,
    "search_prep_j3_max_vel_deg": 60.0,
    "search_prep_j4_max_vel_deg": 120.0,
    "search_prep_j1_max_acc_deg": 60.0,
    "search_prep_j2_max_acc_deg": 60.0,
    "search_prep_j3_max_acc_deg": 60.0,
    "search_prep_j4_max_acc_deg": 120.0,
    "return_search_prep_final_offset_deg": CYLINDRICAL_VISION_PARAMS["search_joint4_deg"],
    "aruco_debug_window": False,
    "aruco_debug_window_name": "Move Cup Target ArUco",

    # Arm geometry and joint limits for cylindrical IK. Move-cup targets the
    # wrist frame directly, so the cylindrical tool offsets stay at zero.
    "l1": 0.1159,
    "l2": 0.1350,
    "horizontal_tool_offset": 0.0,
    "z_tool_offset": 0.0,
    "theta_joint_sign": -1.0,
    "theta_joint_offset_deg": 0.0,
    "base_min_deg": -90.0,
    "base_max_deg": 90.0,
    "shoulder_min_deg": -100.0,
    "shoulder_max_deg": 100.0,
    "elbow_min_deg": -90.0,
    "elbow_max_deg": 100.0,
    "wrist_flex_min_deg": -104.0,
    "wrist_flex_max_deg": 100.0,
    "horizontal_wrist_pitch_offset_deg": 5.0,
    "wrist_roll_deg": -90.0,

    # Gripper values used while carrying and when releasing the cup.
    "hold_gripper_percent": 5.0,
    "release_gripper_percent": 50.0,

    # Pose to return to after releasing the cup.
    "return_joint2_deg": -90.0,
    "return_joint3_deg": 98.0,
}


RANDOM_PLACE_CUP_PARAMS = {
    # Topics and timing.
    "command_topic": "/random_place_cup_command",
    "status_topic": "/random_place_cup_status",
    "debug_topic": "/random_place_cup_debug",
    "target_topic": "/target_joint_states",
    "real_joint_states_topic": "/real_joint_states",
    "control_rate": 20.0,
    "joint_goal_tolerance_deg": 6.0,
    "settle_cycles": 10,
    "grip_settle_sec": 1.0,
    "release_settle_sec": 1.0,
    "target_check_joint_count": 4,
    "allow_timeout_advance": True,
    "phase_timeout_sec": 3.0,
    "timeout_advance_tolerance_deg": 12.0,

    # Training placement region in table coordinates. Forward is the theta=0
    # direction from the shoulder pan axis; lateral is left/right on the table.
    "table_forward_min": 0.0,
    "table_forward_max": 0.22,
    "table_lateral_min": -0.27,
    "table_lateral_max": 0.27,
    "radius_tip_min": 0.30,
    "radius_tip_max": 0.40,
    "theta_min_deg": -90.0,
    "theta_max_deg": 90.0,
    "place_z": 0.01,
    "max_sample_attempts": 200,
    "random_seed": 0,

    # Carry and return motion.
    "lift_z_offset": 0.05,
    "lift_duration_sec": 1.2,
    "pre_pan_radius": 0.22,
    "pre_pan_retract_duration_sec": 1.5,
    "post_pan_extend_duration_sec": 1.2,
    "horizontal_move_duration_sec": 3.0,
    "lower_duration_sec": 1.2,
    "use_dynamic_carry_z": True,
    "carry_z": 0.03,
    "return_home_z": 0.05,
    "return_home_lift_duration_sec": 1.0,
    "return_home_speed_scale": 2.0,
    "return_pan_deg": 0.0,
    "return_joint2_deg": -90.0,
    "return_joint3_deg": 98.0,

    # Arm geometry, tool offset, and joint limits for cylindrical IK.
    "l1": 0.1159,
    "l2": 0.1350,
    "horizontal_tool_offset": 0.10,
    "z_tool_offset": -0.10,
    "theta_joint_sign": -1.0,
    "theta_joint_offset_deg": 0.0,
    "base_min_deg": -90.0,
    "base_max_deg": 90.0,
    "shoulder_min_deg": -100.0,
    "shoulder_max_deg": 100.0,
    "elbow_min_deg": -90.0,
    "elbow_max_deg": 100.0,
    "wrist_flex_min_deg": -104.0,
    "wrist_flex_max_deg": 100.0,
    "horizontal_wrist_pitch_offset_deg": 5.0,
    "wrist_roll_deg": -90.0,

    # Gripper values used while carrying and when releasing the cup.
    "hold_gripper_percent": 12.0,
    "release_gripper_percent": 50.0,
}


FINE_ALIGN_TRAINING_LOOP_PARAMS = {
    "command_topic": "/fine_align_training/command",
    "status_topic": "/fine_align_training/status",
    "task_status_topic": "/task_manager_status",
    "reset_command_topic": "/random_place_cup_command",
    "reset_status_topic": "/random_place_cup_status",
    "grasp_cup_action_name": "/grasp_cup",
    "grasp_timeout_sec": 60.0,
    "reset_timeout_sec": 45.0,
    "auto_start": True,
    "max_episodes": 0,
    "settle_sec": 0.75,
    "reset_start_delay_sec": 0.5,
    "continue_on_grasp_failure": False,
    "continue_on_reset_failure": False,
}


TASK_MANAGER_PARAMS = {
    "move_cup_command_topic": "/move_cup_command",
    "move_cup_status_topic": "/move_cup_status",
    "scan_move_target_before_grasp": False,
    "move_cup_scan_command": "scan",
    "grasp_cup_action_name": "/grasp_cup",
    "move_cup_action_name": "/move_cup",
    "grasp_timeout_sec": 60.0,
    "move_timeout_sec": 45.0,
    "press_button_command_topic": "/press_button_command",
    "press_button_status_topic": "/press_button_status",
    "task_status_topic": "/task_manager_status",
    "auto_start_grasp": True,
    "auto_start_move": True,
    "auto_start_press_button": False,
}


PRESS_BUTTON_PARAMS = {
    "target_topic": "/target_joint_states",
    "real_joint_states_topic": "/real_joint_states",
    "command_topic": "/press_button_command",
    "status_topic": "/press_button_status",
    "control_rate": 20.0,
    "settle_cycles": 8,
    "joint_goal_tolerance_deg": 5.0,
    "camera_device": INNOMAKER_CAMERA_DEVICE,
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "calibration_path": "data/calibration/camera_calibration.yaml",
    "aruco_dictionary": "4X4_50",
    "target_marker_id": 0,
    "marker_length": 0.05,
    "base_frame": "base",
    "camera_frame": "camera_optical_frame",
    "gripper_frame": "grippertip",
    "urdf_path": "",
    "tf_timeout_sec": 0.5,
    "target_x_offset": -0.053,
    "target_y_offset": -0.065,
    "target_z_offset": -0.01,
    "prep_joint1_deg": 0.0,
    "prep_joint2_deg": 0.0,
    "prep_joint3_deg": -45.0,
    "prep_joint4_deg": 65.0,
    "prep_trajectory_duration_sec": 5.0,
    "prep_j2_transition_sec": 3.0,
    "search_pan_min_deg": 0.0,
    "search_pan_max_deg": 90.0,
    "search_pan_speed_deg_s": 15.0,
    "move_to_marker_speed_scale": 0.25,
    "align_wrist_pitch_deg": 59.4,
    "ik_joint4_min_deg": 30.0,
    "ik_joint4_max_deg": 60.0,
    "track_marker_during_move": True,
    "move_goal_update_period_sec": 0.5,
    "fine_align_enable": True,
    "fine_align_start_distance": 0.03,
    "fine_align_tolerance": 0.01,
    "fine_align_max_step": 0.006,
    "fine_align_speed_scale": 0.25,
    "fine_align_max_joint_step_deg": 8.0,
    "fine_align_lock_joint4": False,
    "fine_align_use_ik_service": True,
    "fine_align_ik_service_name": "/solve_grippertip_ik",
    "fine_align_ik_service_timeout_sec": 0.5,
    "fine_align_ik_service_fallback": True,
    "fine_align_max_cycles": 30,
    "fine_align_integral_enable": True,
    "fine_align_max_command_ahead": 0.06,
    "fine_align_marker_samples": 5,
    "fine_align_sample_wait_sec": 0.2,
    "fine_align_settle_cycles": 8,
    "fine_align_move_timeout_sec": 2.0,
    "fine_align_joint_goal_tolerance_deg": 0.10,
    "press_wait_sec": 1.0,
    "press_joint2_delta_deg": 10.0,
    "press_joint3_delta_deg": 15.0,
    "press_joint4_delta_deg": 10.0,
    "press_motion_speed_scale": 2.0,
    "press_motion_settle_cycles": 5,
    "press_repeat_wait_sec": 2.0,
    "press_cycles": 2,
    "fixed_joint5_deg": -90.0,
    "ik_position_tolerance": 0.01,
    "joint6_percent": 0.0,
    "debug_window": True,
    "debug_window_name": "Press Button",
    "publish_debug_image": True,
    "debug_image_topic": "/press_button/debug_image",
    "lock_wait_sec": 1.0,
    "auto_start": True,
}


ARUCO_FOLLOWER_PARAMS = {
    "command_topic": "/aruco_follower_command",
    "status_topic": "/aruco_follower_status",
    "target_topic": "/target_joint_states",
    "real_joint_states_topic": "/real_joint_states",
    "debug_image_topic": "/aruco_follower/image",
    "debug_marker_topic": "/aruco_follower/markers",

    "camera_device": INNOMAKER_CAMERA_DEVICE,
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "show_window": True,
    "window_name": "ArUco follower",
    "publish_debug_image": True,
    "debug_window": True,

    "calibration_path": PRESS_BUTTON_PARAMS["calibration_path"],
    "aruco_dictionary": PRESS_BUTTON_PARAMS["aruco_dictionary"],
    "target_marker_id": 2,
    "marker_length": PRESS_BUTTON_PARAMS["marker_length"],
    "base_frame": PRESS_BUTTON_PARAMS["base_frame"],
    "camera_frame": PRESS_BUTTON_PARAMS["camera_frame"],
    "gripper_frame": PRESS_BUTTON_PARAMS["gripper_frame"],
    "tf_timeout_sec": PRESS_BUTTON_PARAMS["tf_timeout_sec"],

    # The marker is vertical. OpenCV marker +Z is the marker normal, so +5 cm
    # means a point in front of the marker face rather than inside the surface.
    "target_x_offset": 0.0,
    "target_y_offset": 0.0,
    "target_z_offset": 0.05,

    "control_rate": 20.0,
    "auto_start": True,
    "align_tolerance_m": 0.01,
    "max_xy_step_m": 0.005,
    "min_command_period_sec": 0.08,
    "hold_gripper_percent": MOVE_CUP_PARAMS["hold_gripper_percent"],
    "carry_z": MOVE_CUP_PARAMS["carry_z"],
    "pre_pan_radius": MOVE_CUP_PARAMS["pre_pan_radius"],
    "prep_joint_goal_tolerance_deg": MOVE_CUP_PARAMS["joint_goal_tolerance_deg"],
    "prep_settle_cycles": MOVE_CUP_PARAMS["settle_cycles"],
    "prep_target_check_joint_count": MOVE_CUP_PARAMS["target_check_joint_count"],

    "l1": MOVE_CUP_PARAMS["l1"],
    "l2": MOVE_CUP_PARAMS["l2"],
    "horizontal_tool_offset": MOVE_CUP_PARAMS["horizontal_tool_offset"],
    "z_tool_offset": MOVE_CUP_PARAMS["z_tool_offset"],
    "theta_joint_sign": MOVE_CUP_PARAMS["theta_joint_sign"],
    "theta_joint_offset_deg": MOVE_CUP_PARAMS["theta_joint_offset_deg"],
    "base_min_deg": MOVE_CUP_PARAMS["base_min_deg"],
    "base_max_deg": MOVE_CUP_PARAMS["base_max_deg"],
    "shoulder_min_deg": MOVE_CUP_PARAMS["shoulder_min_deg"],
    "shoulder_max_deg": MOVE_CUP_PARAMS["shoulder_max_deg"],
    "elbow_min_deg": MOVE_CUP_PARAMS["elbow_min_deg"],
    "elbow_max_deg": MOVE_CUP_PARAMS["elbow_max_deg"],
    "wrist_flex_min_deg": MOVE_CUP_PARAMS["wrist_flex_min_deg"],
    "wrist_flex_max_deg": MOVE_CUP_PARAMS["wrist_flex_max_deg"],
    "horizontal_wrist_pitch_offset_deg": MOVE_CUP_PARAMS[
        "horizontal_wrist_pitch_offset_deg"
    ],
    "wrist_roll_deg": MOVE_CUP_PARAMS["wrist_roll_deg"],
}
