# flake8: noqa
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from ninibot.task_params import PRESS_BUTTON_PARAMS


def arg(name, default, description):
    return DeclareLaunchArgument(name, default_value=default, description=description)


def launch_default(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def config_path(filename):
    return PathJoinSubstitution([
        FindPackageShare("ninibot"),
        "config",
        filename,
    ])


def generate_launch_description():
    port = LaunchConfiguration("port")
    dry_run = LaunchConfiguration("dry_run")
    camera_device = LaunchConfiguration("camera_device")
    camera_index = LaunchConfiguration("camera_index")
    show_window = LaunchConfiguration("show_window")
    calibration_path = LaunchConfiguration("calibration_path")

    urdf_path = PathJoinSubstitution([
        FindPackageShare("arm_controller"),
        "urdf",
        "so101.urdf",
    ])
    default_calibration_path = PathJoinSubstitution([
        EnvironmentVariable("NINIBOT_ROOT", default_value=EnvironmentVariable("PWD")),
        "data",
        "calibration",
        "camera_calibration.yaml",
    ])
    press_button_overrides = {
        "camera_device": camera_device,
        "camera_index": ParameterValue(camera_index, value_type=int),
        "calibration_path": calibration_path,
        "debug_window": ParameterValue(show_window, value_type=bool),
        "urdf_path": urdf_path,
    }
    motor_overrides = {
        "port": port,
        "dry_run": ParameterValue(dry_run, value_type=bool),
        "torque_limit": 1000,
        "torque_limit_motors": "shoulder_lift,elbow_flex,wrist_flex",
    }

    return LaunchDescription([
        arg("port", "/dev/ttyACM0", "SO101 serial port"),
        arg("dry_run", "True", "Set False only after validating the complete motion in RViz"),
        arg("calibration_path", default_calibration_path, "Camera calibration YAML; defaults to $NINIBOT_ROOT/data/... or $PWD/data/..."),
        arg("camera_device", launch_default(PRESS_BUTTON_PARAMS["camera_device"]), "Stable OpenCV camera device path; empty string falls back to camera_index"),
        arg("camera_index", launch_default(PRESS_BUTTON_PARAMS["camera_index"]), "OpenCV camera index"),
        arg("show_window", launch_default(PRESS_BUTTON_PARAMS["debug_window"]), "Show OpenCV debug window"),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            arguments=[urdf_path],
            remappings=[("joint_states", "/real_joint_states")],
            output="screen",
        ),

        Node(
            package="ninibot",
            executable="press_button_node",
            name="press_button_node",
            output="screen",
            parameters=[
                dict(PRESS_BUTTON_PARAMS),
                press_button_overrides,
            ],
        ),

        Node(
            package="arm_controller",
            executable="grippertip_ik_service",
            name="grippertip_ik_service",
            output="screen",
            parameters=[{
                "urdf_path": urdf_path,
                "base_frame": PRESS_BUTTON_PARAMS["base_frame"],
                "gripper_frame": PRESS_BUTTON_PARAMS["gripper_frame"],
            }],
        ),

        Node(
            package="arm_controller",
            executable="motor_driver",
            name="motor_driver",
            output="screen",
            parameters=[config_path("motor.yaml"), motor_overrides],
        ),
    ])
