# flake8: noqa
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def arg(name, default, description):
    return DeclareLaunchArgument(name, default_value=default, description=description)


def generate_launch_description():
    port = LaunchConfiguration("port")

    urdf_path = PathJoinSubstitution([
        FindPackageShare("arm_controller"),
        "urdf",
        "so101.urdf",
    ])
    rviz_config = PathJoinSubstitution([
        FindPackageShare("arm_controller"),
        "rviz",
        "gripper_pose_debug.rviz",
    ])

    return LaunchDescription([
        arg("port", "/dev/ttyACM0", "SO101 serial port"),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            arguments=[urdf_path],
            output="screen",
        ),

        Node(
            package="arm_controller",
            executable="read_arm",
            name="read_arm",
            output="screen",
            parameters=[{
                "port": port,
            }],
        ),

        Node(
            package="arm_controller",
            executable="gripper_pose_debug",
            name="gripper_pose_debug_node",
            output="screen",
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            output="screen",
        ),
    ])
