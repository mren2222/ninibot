# flake8: noqa
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from ninibot.task_params import CYLINDRICAL_VISION_PARAMS


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
    detector_backend = LaunchConfiguration("detector_backend")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    yolo_conf = LaunchConfiguration("yolo_conf")
    auto_start = LaunchConfiguration("auto_start")
    skip_search_prep_on_start = LaunchConfiguration("skip_search_prep_on_start")

    urdf_path = PathJoinSubstitution([
        FindPackageShare("arm_controller"),
        "urdf",
        "so101.urdf",
    ])
    python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    yolo_site_packages = PathJoinSubstitution([
        EnvironmentVariable("NINIBOT_ROOT", default_value=EnvironmentVariable("PWD")),
        ".venv-yolo",
        "lib",
        python_tag,
        "site-packages",
    ])

    camera_overrides = {
        "camera_device": camera_device,
        "camera_index": ParameterValue(camera_index, value_type=int),
        "show_window": ParameterValue(show_window, value_type=bool),
        "detector_backend": detector_backend,
        "yolo_model_path": yolo_model_path,
        "yolo_conf": ParameterValue(yolo_conf, value_type=float),
        "auto_start": ParameterValue(auto_start, value_type=bool),
        "skip_search_prep_on_start": ParameterValue(
            skip_search_prep_on_start,
            value_type=bool,
        ),
    }
    motor_overrides = {
        "port": port,
        "dry_run": ParameterValue(dry_run, value_type=bool),
        "torque_limit": 1000,
        "torque_limit_motors": "shoulder_lift,elbow_flex,wrist_flex,gripper",
        "slow_status_topics": "/move_cup_status",
        "slow_statuses": "LIFTING,RETRACTING_RADIUS",
        "slow_status_speed_scale": 0.45,
        "slow_status_accel_scale": 0.45,
        "slow_status_timeout_sec": 0.5,
    }

    return LaunchDescription([
        arg("port", "/dev/ttyACM0", "SO101 serial port"),
        arg("dry_run", "True", "Set False only after validating the complete motion in RViz"),
        arg("camera_device", launch_default(CYLINDRICAL_VISION_PARAMS["camera_device"]), "Stable OpenCV camera device path; empty string falls back to camera_index"),
        arg("camera_index", launch_default(CYLINDRICAL_VISION_PARAMS["camera_index"]), "OpenCV camera index"),
        arg("show_window", launch_default(CYLINDRICAL_VISION_PARAMS["show_window"]), "Show OpenCV debug window"),
        arg("detector_backend", launch_default(CYLINDRICAL_VISION_PARAMS["detector_backend"]), "Cup detector backend: hsv, yolo, or hybrid"),
        arg("yolo_model_path", launch_default(CYLINDRICAL_VISION_PARAMS["yolo_model_path"]), "YOLO model path, for example data/models/yolo/yolo11n.pt"),
        arg("yolo_conf", launch_default(CYLINDRICAL_VISION_PARAMS["yolo_conf"]), "YOLO confidence threshold"),
        arg("auto_start", "True", "Start cup visual servo immediately"),
        arg("skip_search_prep_on_start", "False", "Skip SEARCH_PREP when the cup servo starts"),

        SetEnvironmentVariable(
            name="PYTHONPATH",
            value=[
                yolo_site_packages,
                ":",
                EnvironmentVariable("PYTHONPATH", default_value=""),
            ],
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="real_robot_state_publisher",
            arguments=[urdf_path],
            remappings=[("joint_states", "/real_joint_states")],
            output="screen",
        ),

        Node(
            package="ninibot",
            executable="grasp_cup_node",
            name="grasp_cup_node",
            output="screen",
            prefix="python3",
            parameters=[
                dict(CYLINDRICAL_VISION_PARAMS),
                camera_overrides,
            ],
        ),

        Node(
            package="ninibot",
            executable="grasp_cup_action_server",
            name="grasp_cup_action_server",
            output="screen",
        ),

        Node(
            package="arm_controller",
            executable="motor_driver",
            name="motor_driver",
            output="screen",
            parameters=[config_path("motor.yaml"), motor_overrides],
        ),
    ])
