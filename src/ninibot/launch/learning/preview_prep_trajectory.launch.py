from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def arg(name, default, description):
    return DeclareLaunchArgument(name, default_value=default, description=description)


def generate_launch_description():
    trajectory_path = LaunchConfiguration("trajectory_path")

    urdf_path = PathJoinSubstitution([
        FindPackageShare("arm_controller"),
        "urdf",
        "so101.urdf",
    ])
    default_trajectory_path = PathJoinSubstitution([
        EnvironmentVariable("NINIBOT_ROOT", default_value=EnvironmentVariable("PWD")),
        "data",
        "trajectories",
        "fine_align_prep",
        "sessions",
        "2026-06-15_8cm_base_raw",
        "processed_template.json",
    ])
    rviz_config = PathJoinSubstitution([
        FindPackageShare("ninibot"),
        "rviz",
        "fine_align_prep_preview.rviz",
    ])

    return LaunchDescription([
        arg("trajectory_path", default_trajectory_path, "Processed fine-align-prep trajectory JSON"),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="builtin_fine_align_prep_state_publisher",
            arguments=[urdf_path],
            parameters=[{"frame_prefix": "builtin_/"}],
            remappings=[("joint_states", "/fine_align_prep_builtin_joint_states")],
            output="screen",
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="processed_fine_align_prep_state_publisher",
            arguments=[urdf_path],
            parameters=[{"frame_prefix": "processed_/"}],
            remappings=[("joint_states", "/fine_align_prep_processed_joint_states")],
            output="screen",
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="builtin_preview_offset",
            arguments=[
                "--x", "0.0",
                "--y", "-0.28",
                "--z", "0.0",
                "--roll", "0.0",
                "--pitch", "0.0",
                "--yaw", "0.0",
                "--frame-id", "base",
                "--child-frame-id", "builtin_/base",
            ],
            output="screen",
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="processed_preview_offset",
            arguments=[
                "--x", "0.0",
                "--y", "0.28",
                "--z", "0.0",
                "--roll", "0.0",
                "--pitch", "0.0",
                "--yaw", "0.0",
                "--frame-id", "base",
                "--child-frame-id", "processed_/base",
            ],
            output="screen",
        ),

        Node(
            package="ninibot",
            executable="preview_prep_trajectory",
            name="preview_prep_trajectory",
            output="screen",
            parameters=[{
                "trajectory_path": trajectory_path,
            }],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            output="screen",
        ),
    ])
