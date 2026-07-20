# flake8: noqa
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    urdf_path = PathJoinSubstitution([
        FindPackageShare("arm_controller"),
        "urdf",
        "so101.urdf",
    ])

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            arguments=[urdf_path],
            remappings=[("joint_states", "/real_joint_states")],
            output="screen",
        ),

        Node(
            package="arm_controller",
            executable="point_ik",
            name="point_ik",
            output="screen",
        ),

        Node(
            package="arm_controller",
            executable="motor_driver",
            name="motor_driver",
            output="screen",
            parameters=[{
                "port": "/dev/ttyACM0",
                "dry_run": False,
            }],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
        ),
    ])
