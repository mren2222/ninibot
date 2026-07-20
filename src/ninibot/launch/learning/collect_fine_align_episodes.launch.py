# flake8: noqa
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from ninibot.task_params import CYLINDRICAL_VISION_PARAMS
from ninibot.task_params import FINE_ALIGN_TRAINING_LOOP_PARAMS
from ninibot.task_params import RANDOM_PLACE_CUP_PARAMS


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
    max_episodes = LaunchConfiguration("max_episodes")
    start_episode_logger = LaunchConfiguration("start_episode_logger")
    episode_root = LaunchConfiguration("episode_root")
    save_images = LaunchConfiguration("save_images")

    urdf_path = PathJoinSubstitution([
        FindPackageShare("arm_controller"),
        "urdf",
        "so101.urdf",
    ])

    camera_overrides = {
        "camera_device": camera_device,
        "camera_index": ParameterValue(camera_index, value_type=int),
        "show_window": ParameterValue(show_window, value_type=bool),
        "detector_backend": detector_backend,
        "yolo_model_path": yolo_model_path,
        "yolo_conf": ParameterValue(yolo_conf, value_type=float),
    }
    motor_overrides = {
        "port": port,
        "dry_run": ParameterValue(dry_run, value_type=bool),
    }
    training_loop_overrides = {
        "max_episodes": ParameterValue(max_episodes, value_type=int),
    }

    return LaunchDescription([
        arg("port", "/dev/ttyACM0", "SO101 serial port"),
        arg("dry_run", "True", "Set False only for a supervised real-arm collection run"),
        arg("camera_device", launch_default(CYLINDRICAL_VISION_PARAMS["camera_device"]), "Stable OpenCV camera device path; empty string falls back to camera_index"),
        arg("camera_index", launch_default(CYLINDRICAL_VISION_PARAMS["camera_index"]), "OpenCV camera index"),
        arg("show_window", launch_default(CYLINDRICAL_VISION_PARAMS["show_window"]), "Show OpenCV debug window"),
        arg("detector_backend", launch_default(CYLINDRICAL_VISION_PARAMS["detector_backend"]), "Cup detector backend: hsv or yolo"),
        arg("yolo_model_path", launch_default(CYLINDRICAL_VISION_PARAMS["yolo_model_path"]), "YOLO model path"),
        arg("yolo_conf", launch_default(CYLINDRICAL_VISION_PARAMS["yolo_conf"]), "YOLO confidence threshold"),
        arg("max_episodes", launch_default(FINE_ALIGN_TRAINING_LOOP_PARAMS["max_episodes"]), "Number of fine-align episodes to run; 0 means unlimited"),
        arg("start_episode_logger", "False", "Record fine-align episodes"),
        arg("episode_root", "data/episodes", "Episode logger output directory"),
        arg("save_images", "False", "Save debug images in episode logger"),

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
            parameters=[
                dict(CYLINDRICAL_VISION_PARAMS),
                camera_overrides,
            ],
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
            executable="grasp_cup_action_server",
            name="grasp_cup_action_server",
            output="screen",
        ),

        Node(
            package="ninibot",
            executable="random_place_cup_node",
            name="random_place_cup_node",
            output="screen",
            parameters=[
                dict(RANDOM_PLACE_CUP_PARAMS),
            ],
        ),

        Node(
            package="ninibot",
            executable="fine_align_episode_collector",
            name="fine_align_episode_collector",
            output="screen",
            parameters=[
                dict(FINE_ALIGN_TRAINING_LOOP_PARAMS),
                training_loop_overrides,
            ],
        ),

        Node(
            package="ninibot",
            executable="episode_logger",
            name="episode_logger",
            output="screen",
            parameters=[{
                "episode_root": episode_root,
                "detector_backend": detector_backend,
                "task_name": "fine_align_training",
                "auto_start_on_task": True,
                "auto_stop_on_task": True,
                "save_images": ParameterValue(save_images, value_type=bool),
            }],
            condition=IfCondition(start_episode_logger),
        ),
    ])
