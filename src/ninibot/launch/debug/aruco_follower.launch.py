# flake8: noqa
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from ninibot.task_params import (
    ARUCO_FOLLOWER_PARAMS,
)


def arg(name, default, description):
    return DeclareLaunchArgument(name, default_value=default, description=description)


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

    follower_overrides = {
        "camera_device": camera_device,
        "camera_index": ParameterValue(camera_index, value_type=int),
        "calibration_path": calibration_path,
        "auto_start": True,
    }
    motor_overrides = {
        "port": port,
        "dry_run": ParameterValue(dry_run, value_type=bool),
        "speed_scale": 0.35,
        "accel_scale": 0.35,
        "torque_limit": -1,
        "torque_limit_motors": "shoulder_lift,elbow_flex,wrist_flex",
    }

    return LaunchDescription([
        arg("port", "/dev/ttyACM0", "SO101 serial port"),
        arg("dry_run", "True", "Set False only after validating the complete motion in RViz"),
        arg("camera_device", ARUCO_FOLLOWER_PARAMS["camera_device"], "Stable OpenCV camera device path; empty string falls back to camera_index"),
        arg("camera_index", str(ARUCO_FOLLOWER_PARAMS["camera_index"]), "OpenCV camera index"),
        arg("calibration_path", default_calibration_path, "Camera calibration YAML; defaults to $NINIBOT_ROOT/data/... or $PWD/data/..."),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="real_robot_state_publisher",
            arguments=[urdf_path],
            remappings=[("joint_states", "/real_joint_states")],
            output="screen",
        ),

        Node(
            package="arm_controller",
            executable="motor_driver",
            name="motor_driver",
            output="screen",
            parameters=[config_path("motor.yaml"), motor_overrides],
        ),

        Node(
            package="ninibot",
            executable="aruco_follower_node",
            name="aruco_follower_node",
            output="screen",
            parameters=[
                dict(ARUCO_FOLLOWER_PARAMS),
                config_path("aruco_follower.yaml"),
                follower_overrides,
            ],
        ),
    ])
