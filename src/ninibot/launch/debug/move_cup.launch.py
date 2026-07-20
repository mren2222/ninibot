# flake8: noqa
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from ninibot.task_params import MOVE_CUP_PARAMS


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

    urdf_path = PathJoinSubstitution([
        FindPackageShare("arm_controller"),
        "urdf",
        "so101.urdf",
    ])
    motor_overrides = {
        "port": port,
        "dry_run": ParameterValue(dry_run, value_type=bool),
    }

    return LaunchDescription([
        arg("port", "/dev/ttyACM0", "SO101 serial port"),
        arg("dry_run", "True", "Set False only after validating the complete motion in RViz"),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            arguments=[urdf_path],
            remappings=[("joint_states", "/real_joint_states")],
            output="screen",
        ),

        Node(
            package="ninibot",
            executable="move_cup_node",
            name="move_cup_node",
            output="screen",
            parameters=[
                dict(MOVE_CUP_PARAMS),
            ],
        ),

        Node(
            package="ninibot",
            executable="move_cup_action_server",
            name="move_cup_action_server",
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
