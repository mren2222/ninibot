from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    device = LaunchConfiguration("device")
    config_file = PathJoinSubstitution([
        FindPackageShare("audio_controller"),
        "config",
        "water_flow.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "device",
            default_value="",
            description="sounddevice microphone device id/name; empty uses default input.",
        ),
        Node(
            package="audio_controller",
            executable="water_flow_monitor",
            name="water_flow_monitor_node",
            output="screen",
            parameters=[
                config_file,
                {"device": device},
            ],
        ),
    ])
