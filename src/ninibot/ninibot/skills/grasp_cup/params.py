import math
from dataclasses import dataclass
from dataclasses import fields

from ninibot.skills.common.math_utils import gripper_percent_to_rad
from vision_controller.aruco_detection import normalize_camera_source


@dataclass(frozen=True)
class GraspCupParams:
    camera_device: str
    camera_index: int
    camera_source: object
    frame_width: int
    frame_height: int
    show_window: bool
    publish_debug_images: bool
    window_name: str
    target_topic: str
    real_joint_states_topic: str
    command_topic: str
    skip_search_prep_on_start: bool
    auto_start: bool
    detector_backend: str
    center_filter_alpha: float
    target_center_x_ratio: float
    center_tolerance_px: float
    tracking_settle_tolerance_px: float
    tracking_settle_cycles: int
    target_bbox_width: float
    bbox_tolerance_px: float
    bbox_resume_tolerance_px: float
    theta: float
    theta_min: float
    theta_max: float
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
    search_joint2_deg: float
    search_joint3_deg: float
    search_joint4_deg: float
    search_prep_min_duration_sec: float
    search_prep_head_down_duration_sec: float
    search_prep_duration_scale: float
    search_prep_trajectory_samples: int
    search_prep_max_vel_deg: list
    search_prep_max_acc_deg: list
    fine_align_prep_trajectory_path: str
    fine_align_prep_trajectory_duration_sec: float
    fine_align_prep_shape_scale: float
    fine_align_prep_z_offset_m: float
    joint_goal_tolerance: float
    horizontal_wrist_pitch_offset_deg: float
    horizontal_wrist_pitch_offset: float
    wrist_roll_deg: float
    gripper_percent: float
    gripper_rad: float
    joint_names: list
    theta_kp: float
    theta_max_step: float
    tracking_theta_max_step: float
    theta_sign: float
    fine_align_centered_cycles: int
    approach_radius_kp: float
    approach_max_radius_step: float
    search_enabled: bool
    search_step: float
    publish_while_lost: bool
    tracking_lost_grace_sec: float
    control_rate: float
    lower_h: int
    lower_s: int
    lower_v: int
    upper_h: int
    upper_s: int
    upper_v: int
    min_area: float
    yolo_model_path: str
    yolo_conf: float
    yolo_class_name: str
    yolo_class_id: int
    yolo_imgsz: int
    yolo_device: str
    hybrid_roi_scale: float
    hybrid_min_roi_padding_px: int
    hybrid_hsv_yolo_min_overlap_ratio: float
    hybrid_hsv_yolo_max_center_distance_ratio: float
    hybrid_hsv_candidate_area_weight: float
    hybrid_hsv_candidate_center_distance_weight: float

    @classmethod
    def from_node(cls, node):
        camera_device = str(node.get_parameter("camera_device").value).strip()
        camera_index = int(node.get_parameter("camera_index").value)
        camera_source = normalize_camera_source(
            camera_device if camera_device else camera_index
        )
        gripper_percent = float(node.get_parameter("gripper_percent").value)
        horizontal_wrist_pitch_offset_deg = float(
            node.get_parameter("horizontal_wrist_pitch_offset_deg").value
        )

        return cls(
            camera_device=camera_device,
            camera_index=camera_index,
            camera_source=camera_source,
            frame_width=int(node.get_parameter("frame_width").value),
            frame_height=int(node.get_parameter("frame_height").value),
            show_window=bool(node.get_parameter("show_window").value),
            publish_debug_images=bool(
                node.get_parameter("publish_debug_images").value
            ),
            window_name=node.get_parameter("window_name").value,
            target_topic=node.get_parameter("target_topic").value,
            real_joint_states_topic=node.get_parameter(
                "real_joint_states_topic"
            ).value,
            command_topic=node.get_parameter("command_topic").value,
            skip_search_prep_on_start=bool(
                node.get_parameter("skip_search_prep_on_start").value
            ),
            auto_start=bool(node.get_parameter("auto_start").value),
            detector_backend=str(
                node.get_parameter("detector_backend").value
            ).strip().lower(),
            center_filter_alpha=float(
                node.get_parameter("center_filter_alpha").value
            ),
            target_center_x_ratio=float(
                node.get_parameter("target_center_x_ratio").value
            ),
            center_tolerance_px=float(
                node.get_parameter("center_tolerance_px").value
            ),
            tracking_settle_tolerance_px=float(
                node.get_parameter("tracking_settle_tolerance_px").value
            ),
            tracking_settle_cycles=max(
                1,
                int(node.get_parameter("tracking_settle_cycles").value),
            ),
            target_bbox_width=float(
                node.get_parameter("target_bbox_width").value
            ),
            bbox_tolerance_px=float(node.get_parameter("bbox_tolerance_px").value),
            bbox_resume_tolerance_px=float(
                node.get_parameter("bbox_resume_tolerance_px").value
            ),
            theta=math.radians(
                float(node.get_parameter("start_theta_deg").value)
            ),
            theta_min=math.radians(
                float(node.get_parameter("theta_min_deg").value)
            ),
            theta_max=math.radians(
                float(node.get_parameter("theta_max_deg").value)
            ),
            l1=float(node.get_parameter("l1").value),
            l2=float(node.get_parameter("l2").value),
            horizontal_tool_offset=float(
                node.get_parameter("horizontal_tool_offset").value
            ),
            z_tool_offset=float(node.get_parameter("z_tool_offset").value),
            theta_joint_sign=float(
                node.get_parameter("theta_joint_sign").value
            ),
            theta_joint_offset_deg=float(
                node.get_parameter("theta_joint_offset_deg").value
            ),
            base_min_deg=float(node.get_parameter("base_min_deg").value),
            base_max_deg=float(node.get_parameter("base_max_deg").value),
            shoulder_min_deg=float(node.get_parameter("shoulder_min_deg").value),
            shoulder_max_deg=float(node.get_parameter("shoulder_max_deg").value),
            elbow_min_deg=float(node.get_parameter("elbow_min_deg").value),
            elbow_max_deg=float(node.get_parameter("elbow_max_deg").value),
            wrist_flex_min_deg=float(
                node.get_parameter("wrist_flex_min_deg").value
            ),
            wrist_flex_max_deg=float(
                node.get_parameter("wrist_flex_max_deg").value
            ),
            search_joint2_deg=float(
                node.get_parameter("search_joint2_deg").value
            ),
            search_joint3_deg=float(
                node.get_parameter("search_joint3_deg").value
            ),
            search_joint4_deg=float(
                node.get_parameter("search_joint4_deg").value
            ),
            search_prep_min_duration_sec=max(
                0.1,
                float(node.get_parameter("search_prep_min_duration_sec").value),
            ),
            search_prep_head_down_duration_sec=max(
                0.1,
                float(
                    node.get_parameter(
                        "search_prep_head_down_duration_sec"
                    ).value
                ),
            ),
            search_prep_duration_scale=max(
                1.0,
                float(node.get_parameter("search_prep_duration_scale").value),
            ),
            search_prep_trajectory_samples=max(
                16,
                int(node.get_parameter("search_prep_trajectory_samples").value),
            ),
            search_prep_max_vel_deg=[
                float(node.get_parameter("search_prep_j1_max_vel_deg").value),
                float(node.get_parameter("search_prep_j2_max_vel_deg").value),
                float(node.get_parameter("search_prep_j3_max_vel_deg").value),
                float(node.get_parameter("search_prep_j4_max_vel_deg").value),
            ],
            search_prep_max_acc_deg=[
                float(node.get_parameter("search_prep_j1_max_acc_deg").value),
                float(node.get_parameter("search_prep_j2_max_acc_deg").value),
                float(node.get_parameter("search_prep_j3_max_acc_deg").value),
                float(node.get_parameter("search_prep_j4_max_acc_deg").value),
            ],
            fine_align_prep_trajectory_path=str(
                node.get_parameter("fine_align_prep_trajectory_path").value
            ).strip(),
            fine_align_prep_trajectory_duration_sec=float(
                node.get_parameter(
                    "fine_align_prep_trajectory_duration_sec"
                ).value
            ),
            fine_align_prep_shape_scale=float(
                node.get_parameter("fine_align_prep_shape_scale").value
            ),
            fine_align_prep_z_offset_m=float(
                node.get_parameter("fine_align_prep_z_offset_m").value
            ),
            joint_goal_tolerance=math.radians(
                float(node.get_parameter("joint_goal_tolerance_deg").value)
            ),
            horizontal_wrist_pitch_offset_deg=horizontal_wrist_pitch_offset_deg,
            horizontal_wrist_pitch_offset=math.radians(
                horizontal_wrist_pitch_offset_deg
            ),
            wrist_roll_deg=float(node.get_parameter("wrist_roll_deg").value),
            gripper_percent=gripper_percent,
            gripper_rad=gripper_percent_to_rad(gripper_percent),
            joint_names=["1", "2", "3", "4", "5", "6"],
            theta_kp=float(node.get_parameter("theta_kp").value),
            theta_max_step=math.radians(
                float(node.get_parameter("theta_max_step_deg").value)
            ),
            tracking_theta_max_step=math.radians(
                float(node.get_parameter("tracking_theta_max_step_deg").value)
            ),
            theta_sign=float(node.get_parameter("theta_sign").value),
            fine_align_centered_cycles=max(
                1,
                int(node.get_parameter("fine_align_centered_cycles").value),
            ),
            approach_radius_kp=float(
                node.get_parameter("approach_radius_kp_m_per_px").value
            ),
            approach_max_radius_step=float(
                node.get_parameter("approach_max_radius_step_m").value
            ),
            search_enabled=bool(node.get_parameter("search_enabled").value),
            search_step=math.radians(
                float(node.get_parameter("search_step_deg").value)
            ),
            publish_while_lost=bool(
                node.get_parameter("publish_while_lost").value
            ),
            tracking_lost_grace_sec=max(
                0.0,
                float(node.get_parameter("tracking_lost_grace_sec").value),
            ),
            control_rate=float(node.get_parameter("control_rate").value),
            lower_h=int(node.get_parameter("lower_h").value),
            lower_s=int(node.get_parameter("lower_s").value),
            lower_v=int(node.get_parameter("lower_v").value),
            upper_h=int(node.get_parameter("upper_h").value),
            upper_s=int(node.get_parameter("upper_s").value),
            upper_v=int(node.get_parameter("upper_v").value),
            min_area=float(node.get_parameter("min_area").value),
            yolo_model_path=str(node.get_parameter("yolo_model_path").value),
            yolo_conf=float(node.get_parameter("yolo_conf").value),
            yolo_class_name=str(node.get_parameter("yolo_class_name").value),
            yolo_class_id=int(node.get_parameter("yolo_class_id").value),
            yolo_imgsz=int(node.get_parameter("yolo_imgsz").value),
            yolo_device=str(node.get_parameter("yolo_device").value),
            hybrid_roi_scale=float(node.get_parameter("hybrid_roi_scale").value),
            hybrid_min_roi_padding_px=int(
                node.get_parameter("hybrid_min_roi_padding_px").value
            ),
            hybrid_hsv_yolo_min_overlap_ratio=float(
                node.get_parameter("hybrid_hsv_yolo_min_overlap_ratio").value
            ),
            hybrid_hsv_yolo_max_center_distance_ratio=float(
                node.get_parameter(
                    "hybrid_hsv_yolo_max_center_distance_ratio"
                ).value
            ),
            hybrid_hsv_candidate_area_weight=float(
                node.get_parameter("hybrid_hsv_candidate_area_weight").value
            ),
            hybrid_hsv_candidate_center_distance_weight=float(
                node.get_parameter(
                    "hybrid_hsv_candidate_center_distance_weight"
                ).value
            ),
        )

    def apply_to(self, target):
        for field in fields(self):
            setattr(target, field.name, getattr(self, field.name))
