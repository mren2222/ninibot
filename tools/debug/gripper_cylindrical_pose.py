#!/usr/bin/env python3
# flake8: noqa
import argparse
import contextlib
import os
import pathlib
import sys
import time


DEFAULT_PORT = "/dev/ttyACM0"


def add_workspace_packages_to_path():
    workspace = pathlib.Path(__file__).resolve().parents[2]
    for relative in ("src/arm_controller", "src/vision_controller", "src/ninibot"):
        path = workspace / relative
        if path.exists():
            sys.path.insert(0, str(path))


add_workspace_packages_to_path()

from arm_controller.kinematics.cylindrical import CylindricalArmKinematics
from ninibot.task_params import CYLINDRICAL_VISION_PARAMS


def build_kinematics():
    params = CYLINDRICAL_VISION_PARAMS
    return CylindricalArmKinematics(
        l1=params["l1"],
        l2=params["l2"],
        horizontal_tool_offset=params["horizontal_tool_offset"],
        z_tool_offset=params["z_tool_offset"],
        theta_joint_sign=params["theta_joint_sign"],
        theta_joint_offset_deg=params["theta_joint_offset_deg"],
        base_min_deg=params["base_min_deg"],
        base_max_deg=params["base_max_deg"],
        shoulder_min_deg=params["shoulder_min_deg"],
        shoulder_max_deg=params["shoulder_max_deg"],
        elbow_min_deg=params["elbow_min_deg"],
        elbow_max_deg=params["elbow_max_deg"],
        wrist_flex_min_deg=params["wrist_flex_min_deg"],
        wrist_flex_max_deg=params["wrist_flex_max_deg"],
        horizontal_wrist_pitch_offset_deg=params["horizontal_wrist_pitch_offset_deg"],
    )


def connect_robot(port, disable_torque):
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    robot = SOFollower(SOFollowerRobotConfig(port=port))
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            robot.connect(calibrate=False)
            if disable_torque:
                robot.bus.disable_torque()
    return robot


def observation_to_joint_degs(obs):
    return (
        float(obs["shoulder_pan.pos"]),
        float(obs["shoulder_lift.pos"]),
        float(obs["elbow_flex.pos"]),
    )


def format_pose(pose):
    return f"r={pose.radius:.4f} theta={pose.theta_deg:+.2f} z={pose.z:.4f}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read SO101 motors directly and print gripper cylindrical pose."
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--keep-torque",
        action="store_true",
        help="Do not disable torque after connecting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    kinematics = build_kinematics()
    robot = connect_robot(args.port, disable_torque=not args.keep_torque)
    period = 1.0 / max(args.rate, 1e-6)

    try:
        while True:
            obs = robot.get_observation()
            base_deg, shoulder_deg, elbow_deg = observation_to_joint_degs(obs)
            pose = kinematics.cylindrical_from_joint_degs(
                base_deg,
                shoulder_deg,
                elbow_deg,
            )
            print(format_pose(pose), flush=True)
            if args.once:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        with contextlib.suppress(Exception):
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull):
                    robot.disconnect()


if __name__ == "__main__":
    main()
