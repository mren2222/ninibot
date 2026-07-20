#!/usr/bin/env python3
"""Standalone SO100/SO101 calibration workflow for local experiments.

This keeps the important calibration math in one editable place while still
using LeRobot's Feetech bus for serial communication.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from lerobot.motors import MotorCalibration
    from lerobot.motors.feetech import OperatingMode
    from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO100Follower
    from lerobot.robots.so_follower.so_follower import SO101Follower
    from lerobot.utils.utils import enter_pressed
    from lerobot.utils.utils import move_cursor_up
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import LeRobot. Install it in the active environment or set "
        "PYTHONPATH to the checked-out lerobot/src directory."
    ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or inspect SO100/SO101 Feetech calibration."
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot", choices=["so100", "so101"], default="so101")
    parser.add_argument(
        "--id",
        dest="robot_id",
        default=None,
        help="Calibration file id. If omitted, LeRobot uses None.json.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Optional directory for calibration JSON files.",
    )
    parser.add_argument(
        "--mode",
        choices=["calibrate", "show", "write-existing"],
        default="calibrate",
    )
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="Run a new calibration even if a calibration file already exists.",
    )
    parser.add_argument(
        "--full-turn-motor",
        default="wrist_roll",
        help="Motor with known full 0..max encoder range. Use '' to disable.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Write calibration to motors but do not save the JSON file.",
    )
    return parser


def make_robot(args):
    config_class = SO101FollowerConfig if args.robot == "so101" else SO100FollowerConfig
    robot_class = SO101Follower if args.robot == "so101" else SO100Follower
    config = config_class(
        port=args.port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir,
    )
    return robot_class(config)


def motor_model(bus, motor: str) -> str:
    return bus._get_motor_model(motor)


def motor_max_position(bus, motor: str) -> int:
    model = motor_model(bus, motor)
    return int(bus.model_resolution_table[model] - 1)


def print_calibration(label: str, calibration: dict[str, MotorCalibration]) -> None:
    print(f"\n{label}:")
    if not calibration:
        print("  <empty>")
        return
    for motor, cal in calibration.items():
        print(
            f"  {motor:<14} id={cal.id:<2} "
            f"homing_offset={cal.homing_offset:<6} "
            f"range_min={cal.range_min:<6} range_max={cal.range_max:<6} "
            f"drive_mode={cal.drive_mode}"
        )


def set_half_turn_homings_local(bus, motors: list[str]) -> dict[str, int]:
    """Set current raw encoder position to the half-turn center."""
    print("\nResetting calibration registers before computing homing offsets...")
    bus.reset_calibration(motors)

    positions = bus.sync_read("Present_Position", motors, normalize=False)
    homing_offsets = {}
    print("\nComputed homing offsets:")
    for motor in motors:
        pos = int(positions[motor])
        half_turn = int(motor_max_position(bus, motor) / 2)
        offset = pos - half_turn
        homing_offsets[motor] = offset
        print(
            f"  {motor:<14} raw_pos={pos:<6} half_turn={half_turn:<6} "
            f"homing_offset={offset}"
        )

    for motor, offset in homing_offsets.items():
        bus.write("Homing_Offset", motor, offset, normalize=False)

    return homing_offsets


def record_ranges_local(bus, motors: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Record raw encoder min/max while the user moves joints by hand."""
    start_positions = bus.sync_read("Present_Position", motors, normalize=False)
    mins = {motor: int(value) for motor, value in start_positions.items()}
    maxes = mins.copy()

    print("\nMove the listed joints through their full range. Press ENTER to stop.")
    user_pressed_enter = False
    while not user_pressed_enter:
        positions = bus.sync_read("Present_Position", motors, normalize=False)
        positions = {motor: int(value) for motor, value in positions.items()}
        mins = {motor: min(positions[motor], mins[motor]) for motor in motors}
        maxes = {motor: max(positions[motor], maxes[motor]) for motor in motors}

        print("\n-------------------------------------------")
        print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
        for motor in motors:
            print(f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")

        if enter_pressed():
            user_pressed_enter = True
        if not user_pressed_enter:
            move_cursor_up(len(motors) + 3)

    stuck = [motor for motor in motors if mins[motor] == maxes[motor]]
    if stuck:
        raise RuntimeError(f"These motors never moved, so min=max: {stuck}")

    return mins, maxes


def build_calibration(bus, homing_offsets, range_mins, range_maxes):
    calibration = {}
    for motor, motor_info in bus.motors.items():
        calibration[motor] = MotorCalibration(
            id=motor_info.id,
            drive_mode=0,
            homing_offset=int(homing_offsets[motor]),
            range_min=int(range_mins[motor]),
            range_max=int(range_maxes[motor]),
        )
    return calibration


def run_new_calibration(robot, args) -> None:
    bus = robot.bus
    motors = list(bus.motors)
    full_turn_motor = args.full_turn_motor.strip() or None

    print(f"\nRunning calibration for {args.robot} on {args.port}")
    print(f"Calibration file: {robot.calibration_fpath}")
    print("Disabling torque so you can move the arm by hand...")
    bus.disable_torque()
    for motor in motors:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    input("\nMove the robot to the middle of its range of motion, then press ENTER...")
    homing_offsets = set_half_turn_homings_local(bus, motors)

    range_motors = list(motors)
    if full_turn_motor:
        if full_turn_motor not in motors:
            raise ValueError(f"Unknown full-turn motor: {full_turn_motor}")
        range_motors = [motor for motor in motors if motor != full_turn_motor]

    range_mins, range_maxes = record_ranges_local(bus, range_motors)
    if full_turn_motor:
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = motor_max_position(bus, full_turn_motor)
        print(
            f"\nUsing full-turn range for {full_turn_motor}: "
            f"{range_mins[full_turn_motor]}..{range_maxes[full_turn_motor]}"
        )

    calibration = build_calibration(bus, homing_offsets, range_mins, range_maxes)
    print_calibration("New calibration", calibration)

    print("\nWriting calibration to motor registers...")
    bus.write_calibration(calibration)
    robot.calibration = calibration
    if not args.no_save:
        robot._save_calibration()
        print("Saved calibration to", robot.calibration_fpath)
    else:
        print("Skipped saving calibration JSON because --no-save was set.")


def run(args) -> int:
    robot = make_robot(args)
    print(f"Calibration file path: {robot.calibration_fpath}")

    if args.mode == "show":
        print_calibration("Loaded file calibration", robot.calibration)
        return 0

    if args.mode == "write-existing":
        if not robot.calibration:
            print("No calibration file loaded; nothing to write.")
            return 1
        robot.bus.connect()
        try:
            print_calibration("Writing existing calibration", robot.calibration)
            robot.bus.write_calibration(robot.calibration)
        finally:
            robot.bus.disconnect()
        return 0

    robot.bus.connect()
    try:
        if robot.calibration and not args.force_new:
            print_calibration("Existing calibration file", robot.calibration)
            choice = input(
                "\nPress ENTER to write this file to the motors, "
                "or type 'c' then ENTER to run a new calibration: "
            )
            if choice.strip().lower() != "c":
                robot.bus.write_calibration(robot.calibration)
                print("Existing calibration written to motors.")
                return 0

        run_new_calibration(robot, args)
    finally:
        robot.bus.disconnect()

    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
