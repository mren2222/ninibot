from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def arg(name, default, description):
    return DeclareLaunchArgument(name, default_value=default, description=description)


def generate_launch_description():
    policy_model_path = LaunchConfiguration("policy_model_path")
    dry_run = LaunchConfiguration("dry_run")

    urdf_path = PathJoinSubstitution([
        FindPackageShare("arm_controller"),
        "urdf",
        "so101.urdf",
    ])
    default_policy_model_path = PathJoinSubstitution([
        EnvironmentVariable("NINIBOT_ROOT", default_value=EnvironmentVariable("PWD")),
        "data",
        "models",
        "structured_policy_fine_align",
        "policy.pt",
    ])
    rviz_config = PathJoinSubstitution([
        FindPackageShare("ninibot"),
        "rviz",
        "grasp_policy_preview.rviz",
    ])

    return LaunchDescription([
        arg("policy_model_path", default_policy_model_path, "Structured policy .pt model path"),
        arg("dry_run", "True", "Keep the included manipulation chain in RViz-only mode"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("ninibot"),
                    "launch",
                    "demos",
                    "grasp_and_move_cup_demo.launch.py",
                ])
            ]),
            launch_arguments={
                "detector_backend": "hsv",
                "dry_run": dry_run,
            }.items(),
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="policy_robot_state_publisher",
            arguments=[urdf_path],
            parameters=[{"frame_prefix": "policy_/"}],
            remappings=[("joint_states", "/policy_joint_states")],
            output="screen",
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="policy_base_offset",
            arguments=[
                "--x", "0.0",
                "--y", "0.35",
                "--z", "0.0",
                "--roll", "0.0",
                "--pitch", "0.0",
                "--yaw", "0.0",
                "--frame-id", "base",
                "--child-frame-id", "policy_/base",
            ],
            output="screen",
        ),

        Node(
            package="ninibot",
            executable="policy_preview",
            name="policy_preview",
            output="screen",
            parameters=[{
                "model_path": policy_model_path,
                "inactive_sync_source": "real",
                "active_start_source": "real",
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
