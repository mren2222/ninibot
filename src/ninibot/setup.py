from glob import glob
import os

from setuptools import find_packages, setup


package_name = "ninibot"


def package_files(directory, pattern):
    data_files = []
    for path in sorted(glob(os.path.join(directory, "**", pattern), recursive=True)):
        target_dir = os.path.join("share", package_name, os.path.dirname(path))
        data_files.append((target_dir, [path]))
    return data_files


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ]
    + package_files("rviz", "*.rviz")
    + package_files("launch", "*.launch.py")
    + package_files("config", "*.yaml"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mren2222",
    maintainer_email="182375090+mren2222@users.noreply.github.com",
    description="Top-level Ninibot skills, orchestration, policies, launch, and configuration.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "grasp_cup_node = ninibot.skills.grasp_cup_node:main",
            "aruco_follower_node = ninibot.skills.aruco_follower_node:main",
            "move_cup_node = ninibot.skills.move_cup_node:main",
            "random_place_cup_node = ninibot.skills.random_place_cup_node:main",
            "press_button_node = ninibot.skills.press_button_node:main",
            "shutdown_on_task_status = ninibot.skills.shutdown_on_task_status:main",
            "grasp_cup_action_server = ninibot.actions.grasp_cup_action_server:main",
            "move_cup_action_server = ninibot.actions.move_cup_action_server:main",
            "full_demo_manager = ninibot.orchestration.full_demo_manager:main",
            "fine_align_episode_collector = ninibot.orchestration.fine_align_episode_collector:main",
            "episode_loop_node = ninibot.orchestration.episode_loop_node:main",
            "episode_logger = ninibot.logging.episode_logger:main",
            "policy_preview = ninibot.policies.policy_preview:main",
            "fine_align_policy_gate = ninibot.policies.fine_align_policy_gate:main",
            "record_prep_trajectory = ninibot.trajectories.record_prep_trajectory:main",
            "process_prep_trajectory = ninibot.trajectories.process_prep_trajectory:main",
            "preview_prep_trajectory = ninibot.trajectories.preview_prep_trajectory:main",
        ],
    },
)
