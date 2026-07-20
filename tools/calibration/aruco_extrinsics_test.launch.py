# flake8: noqa
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import TimerAction
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

try:
    from ninibot.task_params import PRESS_BUTTON_PARAMS
except Exception:
    PRESS_BUTTON_PARAMS = {
        "camera_device": "",
        "camera_index": 0,
        "target_x_offset": -0.04,
        "target_y_offset": -0.06,
        "target_z_offset": 0.02,
    }


def arg(name, default, description):
    return DeclareLaunchArgument(name, default_value=str(default), description=description)


def generate_launch_description():
    camera_device = LaunchConfiguration("camera_device")
    camera_index = LaunchConfiguration("camera_index")

    urdf_path = PathJoinSubstitution([
        FindPackageShare("arm_controller"),
        "urdf",
        "so101.urdf",
    ])
    project_root = EnvironmentVariable(
        "NINIBOT_ROOT",
        default_value=EnvironmentVariable("PWD"),
    )
    calibration_path = PathJoinSubstitution([
        project_root,
        "data",
        "calibration",
        "camera_calibration.yaml",
    ])
    checker_path = PathJoinSubstitution([
        project_root,
        "tools",
        "calibration",
        "check_aruco_extrinsics.py",
    ])
    pose_recorder_path = PathJoinSubstitution([
        project_root,
        "tools",
        "calibration",
        "press_button_pose_recorder.py",
    ])

    joint_states_topic = "/joint_states"
    base_frame = "base"
    camera_frame = "camera_optical_frame"
    gripper_frame = "grippertip"
    marker_base_topic = "/aruco_extrinsics/marker_base"

    checker_cmd = [
        "/usr/bin/python3",
        "-u",
        "-B",
        checker_path,
        "--calibration",
        calibration_path,
        "--marker-length",
        "0.05",
        "--dictionary",
        "4X4_50",
        "--target-marker-id",
        "0",
        "--base-frame",
        base_frame,
        "--camera-frame",
        camera_frame,
        "--gripper-frame",
        gripper_frame,
        "--debug-image-topic",
        "/aruco_extrinsics/debug_image",
        "--marker-base-topic",
        marker_base_topic,
        "--camera-device",
        camera_device,
        "--camera-id",
        camera_index,
    ]

    return LaunchDescription([
        arg("camera_device", PRESS_BUTTON_PARAMS.get("camera_device", ""), "Stable OpenCV camera device path; empty string falls back to camera_index"),
        arg("camera_index", PRESS_BUTTON_PARAMS.get("camera_index", 0), "OpenCV camera index used when camera_device is empty"),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            arguments=[urdf_path],
            remappings=[("joint_states", joint_states_topic)],
            output="screen",
        ),

        Node(
            package="arm_controller",
            executable="read_arm",
            name="read_arm",
            output="screen",
            parameters=[{
                "publish_rate": 30.0,
                "disable_torque": True,
                "publish_topic": joint_states_topic,
            }],
        ),

        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=checker_cmd,
                    output="screen",
                ),
            ],
        ),

        TimerAction(
            period=5.5,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "/usr/bin/python3",
                        "-u",
                        "-B",
                        pose_recorder_path,
                        "--marker-base-topic",
                        marker_base_topic,
                        "--joint-states-topic",
                        joint_states_topic,
                        "--base-frame",
                        base_frame,
                        "--gripper-frame",
                        gripper_frame,
                        "--target-x-offset",
                        str(PRESS_BUTTON_PARAMS["target_x_offset"]),
                        "--target-y-offset",
                        str(PRESS_BUTTON_PARAMS["target_y_offset"]),
                        "--target-z-offset",
                        str(PRESS_BUTTON_PARAMS["target_z_offset"]),
                        "--print-every",
                        "0.5",
                    ],
                    output="screen",
                ),
            ],
        ),
    ])
